"""Live plotly view comparing a ground-truth and a predicted flux volume.

``LiveReconstructionView.render`` min-max normalises both volumes per field, takes log10 (floored
at ``log_floor``) and builds one figure: ground truth in the top row, prediction in the bottom row,
each with three slices (xy, yz, xz) and a 3D isosurface (through ``slice_center``, default the
ground-truth peak voxel, which is marked as the focal spot; optional geometry layers are drawn as
contours in the slices and as grey surfaces in 3D), on a
shared colour range of [-6, 0] decades.

The view serves itself: on the first render a daemon HTTP server starts on 127.0.0.1:``port``
(an OS-chosen port if that one is taken; ``self.url`` holds the result). ``GET /`` returns a page
that polls ``GET /figure.json`` every ``refresh_s`` seconds and updates the plot in place with
``Plotly.react`` (``uirevision`` keeps the 3D camera), and shows "waiting for the next run" while
the server is unreachable, so one tab survives a chain of runs that share the port. The first
render opens that URL in the browser if ``open_browser`` is set and a display exists. Plotly.js
comes from the CDN, so the browser needs internet access.

A static snapshot of the same figure is written to every path in ``out_paths`` through a temporary
file and ``os.replace``. The class knows nothing about batches, datamodules or networks.
"""
import json
import os
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import torch
from torch import Tensor

C_MIN, C_MAX = -6.0, 0.0
PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>live reconstruction</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>body{font-family:sans-serif;margin:8px}#status{color:#555;margin:4px 0}</style></head>
<body><div id="status">waiting for the next run&hellip;</div><div id="plot"></div>
<script>
const REFRESH = __REFRESH__;
async function tick() {
  const st = document.getElementById("status");
  try {
    const r = await fetch("/figure.json", {cache: "no-store"});
    const f = await r.json();
    if (f.data) {
      f.layout.uirevision = "keep";
      await Plotly.react("plot", f.data, f.layout);
      st.textContent = f.status || "";
    } else { st.textContent = "waiting for the first render\u2026"; }
  } catch (e) { st.textContent = "waiting for the next run\u2026"; }
}
tick(); setInterval(tick, REFRESH * 1000);
</script></body></html>"""

CAMERA = dict(eye=dict(x=1.6, y=1.6, z=1.0))


class LiveReconstructionView:
    def __init__(self, out_paths: list[str], refresh_s: int = 20, open_browser: bool = False,
                 log_floor: float = 1e-7, iso_stride: int = 2, port: int = 8765, log=print):
        self.out_paths = [out_paths] if isinstance(out_paths, str) else list(out_paths)
        self.refresh_s = refresh_s
        self.open_browser = open_browser
        self.log_floor = log_floor
        self.iso_stride = iso_stride
        self.port = port
        self.url = None
        self._log = log
        self._opened = False
        self._lock = threading.Lock()
        self._payload = b"{}"
        self._server = None

    def _start_server(self):
        view = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path.split("?")[0] == "/figure.json":
                    with view._lock:
                        body, ctype = view._payload, "application/json"
                elif self.path.split("?")[0] == "/":
                    body = PAGE.replace("__REFRESH__", str(view.refresh_s)).encode()
                    ctype = "text/html; charset=utf-8"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        try:
            self._server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        except OSError:
            self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}/"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        self._log(f"[live view] serving {self.url}")

    def close(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def _log_volume(self, t: Tensor) -> np.ndarray:
        t = t.detach().float()
        v = t - t.min()
        v = v / v.max().clamp_min(1e-12)
        return torch.log10(v.clamp_min(self.log_floor)).cpu().numpy()

    @staticmethod
    def _mask(v) -> np.ndarray:
        v = v.detach().cpu().numpy() if isinstance(v, Tensor) else np.asarray(v)
        return np.squeeze(v) > 0

    def _figure(self, gt: np.ndarray, pred: np.ndarray, title: str, geometry: dict, center, marker) -> go.Figure:
        i, j, k = (int(c) for c in center)
        mi, mj, mk = (int(c) for c in marker)
        pos = (f"z = {k}", f"x = {i}", f"y = {j}")
        names = ("xy", "yz", "xz")
        titles = [f"{r} {n}, {p}" if c < 3 else f"{r} 3D" for r in ("GT", "pred")
                  for c, (n, p) in enumerate(list(zip(names, pos)) + [("3D", "")])]
        fig = make_subplots(rows=2, cols=4, subplot_titles=titles,
                            specs=[[{}, {}, {}, {"type": "scene"}]] * 2,
                            column_widths=[1, 1, 1, 1.5], horizontal_spacing=0.03, vertical_spacing=0.1)
        s = self.iso_stride
        x3, y3, z3 = np.meshgrid(*[np.arange(0, n, s) for n in gt.shape], indexing="ij")
        grid = dict(x=x3.ravel(), y=y3.ravel(), z=z3.ravel())
        no_caps = dict(x_show=False, y_show=False, z_show=False)
        # (slice index, in-plane marker coordinates (col, row), marker lies in this slice)
        planes = ((lambda v: v[:, :, k].T, (mi, mj), abs(mk - k) <= 1),
                  (lambda v: v[i, :, :].T, (mj, mk), abs(mi - i) <= 1),
                  (lambda v: v[:, j, :].T, (mi, mk), abs(mj - j) <= 1))
        for row, vol in ((1, gt), (2, pred)):
            for col, (cut, (mx, my), hit) in enumerate(planes, start=1):
                fig.add_trace(go.Heatmap(z=cut(vol), zmin=C_MIN, zmax=C_MAX, colorscale="Viridis",
                                         showscale=(row == 1 and col == 1),
                                         colorbar=dict(title="log10", len=0.9)), row=row, col=col)
                for name, m in geometry.items():
                    color = "white" if name == "patient" else "lightgrey"
                    fig.add_trace(go.Contour(z=cut(m).astype(np.float32), showscale=False, hoverinfo="skip",
                                             contours=dict(coloring="lines", start=0.5, end=0.5, size=1),
                                             colorscale=[[0, color], [1, color]], line=dict(width=2)),
                              row=row, col=col)
                if hit:
                    fig.add_trace(go.Scatter(x=[mx], y=[my], mode="markers+text", text=["Fokus"],
                                             textposition="top right", showlegend=False,
                                             marker=dict(symbol="x", color="red", size=9),
                                             textfont=dict(color="red")), row=row, col=col)
            fig.add_trace(go.Isosurface(**grid, value=vol[::s, ::s, ::s].ravel(), isomin=-4, isomax=-1,
                                        surface_count=4, opacity=0.08, caps=no_caps, colorscale="Viridis",
                                        cmin=C_MIN, cmax=C_MAX, showscale=False), row=row, col=4)
            for name, m in geometry.items():
                c = "dimgrey" if name == "patient" else "lightgrey"
                fig.add_trace(go.Isosurface(**grid, value=m[::s, ::s, ::s].astype(np.float32).ravel(),
                                            isomin=0.5, isomax=0.5, surface_count=1, caps=no_caps,
                                            opacity=0.6 if name == "patient" else 0.2,
                                            colorscale=[[0, c], [1, c]],
                                            cmin=0, cmax=1, showscale=False, hoverinfo="skip"), row=row, col=4)
            fig.add_trace(go.Scatter3d(x=[mi], y=[mj], z=[mk], mode="markers", showlegend=False,
                                       marker=dict(color="red", size=6), hovertext="Fokus"), row=row, col=4)
        fig.update_layout(title=title, height=620, margin=dict(l=30, r=10, t=80, b=20),
                          scene=dict(camera=CAMERA), scene2=dict(camera=CAMERA))
        return fig

    def render(self, gt_flux: Tensor, pred_flux: Tensor, title: str, status: str = "", geometry: dict | None = None,
               slice_center=None, marker=None) -> int:
        """gt_flux, pred_flux: [D, H, W]. geometry: optional {layer name: occupancy [D, H, W]}, drawn as
        contours in the slices and as grey surfaces in the 3D panels (``patient`` white/dark, the rest
        light). slice_center: voxel triple the slices pass through (default: the GT peak). marker: voxel
        marked as the focal spot (default: the GT peak). Publishes the figure to the server and the snapshot
        files; returns the snapshot size in bytes."""
        gt = self._log_volume(gt_flux)
        peak = np.unravel_index(int(np.argmax(gt)), gt.shape)
        geometry = {n: self._mask(m) for n, m in (geometry or {}).items()}
        fig = self._figure(gt, self._log_volume(pred_flux), title, geometry,
                           peak if slice_center is None else slice_center, peak if marker is None else marker)
        if self._server is None:
            self._start_server()
        payload = json.loads(pio.to_json(fig))
        payload["status"] = f"updated {time.strftime('%H:%M:%S')} — {status or title}"
        with self._lock:
            self._payload = json.dumps(payload).encode()
        html = fig.to_html(include_plotlyjs="cdn", full_html=True)
        for path in self.out_paths:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                f.write(html)
            os.replace(tmp, path)
        if self.open_browser and not self._opened and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            self._opened = True
            try:
                webbrowser.open(self.url)
            except Exception:
                pass
        return len(html)
