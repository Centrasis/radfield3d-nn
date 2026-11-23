"""Convert old flat PBRFNet config JSONs to the new dict-based format.

Mapping (flat -> nested dict):
  location_encoding_dims=N                  -> location_encoding_params {sinusoidal, pos_enc_dim=N}
  direction_encoding_dims=N                 -> direction_encoding_params {spherical_harmonics, degree=N}
  in_spectra_dim / encoded_spectra_dims /
    use_spectra_encoding                    -> spectra_encoding_params {simple|projector, ...}
  conditioning / use_beam_shape             -> conditioning_params {type, use_beam_shape}
  learning_rate / max_lr /
    randomize_voxel_location_in_training /
    voxels_centered_around_origin           -> training_params {...}

An already-present *_params dict in the source is preserved (e.g. the RFF location config).
hyperparameter_space is regenerated as single-element lists mirroring the new parameters (the
existing spaces are all degenerate single-choice).
"""
import json
import sys
from pathlib import Path

FLAT_KEYS = {
    "location_encoding_dims", "direction_encoding_dims",
    "in_spectra_dim", "encoded_spectra_dims", "use_spectra_encoding",
    "conditioning", "use_beam_shape",
    "learning_rate", "max_lr",
    "randomize_voxel_location_in_training", "voxels_centered_around_origin",
}


def convert_parameters(p: dict) -> dict:
    out = {k: v for k, v in p.items() if k not in FLAT_KEYS}

    # location / direction encoders (preserve an explicit *_params dict if already present)
    if "location_encoding_params" not in out and "location_encoding_dims" in p:
        out["location_encoding_params"] = {
            "type": "sinusoidal", "pos_enc_dim": p["location_encoding_dims"], "append_input": True}
    if "direction_encoding_params" not in out and "direction_encoding_dims" in p:
        out["direction_encoding_params"] = {
            "type": "spherical_harmonics", "degree": p["direction_encoding_dims"], "append_input": True}

    # spectra encoder
    if "spectra_encoding_params" not in out and "in_spectra_dim" in p:
        in_dim = p["in_spectra_dim"]
        if p.get("use_spectra_encoding", False):
            out["spectra_encoding_params"] = {
                "type": "simple", "in_spectra_dim": in_dim,
                "encoded_spectra_dims": p.get("encoded_spectra_dims", in_dim)}
        else:
            out["spectra_encoding_params"] = {
                "type": "projector", "in_spectra_dim": in_dim, "out_spectra_dim": in_dim}

    # conditioning (fold use_beam_shape)
    if "conditioning_params" not in out and "conditioning" in p:
        cp = {"type": p["conditioning"]}
        if "use_beam_shape" in p:
            cp["use_beam_shape"] = p["use_beam_shape"]
        out["conditioning_params"] = cp

    # training params (only include keys actually present)
    tp = {k: p[k] for k in ("learning_rate", "randomize_voxel_location_in_training",
                            "voxels_centered_around_origin", "max_lr") if k in p}
    if "training_params" not in out and tp:
        out["training_params"] = tp

    return out


def main(paths):
    for path in paths:
        path = Path(path)
        cfg = json.loads(path.read_text())
        cfg["parameters"] = convert_parameters(cfg.get("parameters", {}))
        # regenerate the (degenerate) hyperparameter space from the new parameters
        if "hyperparameter_space" in cfg:
            cfg["hyperparameter_space"] = {k: [v] for k, v in cfg["parameters"].items()}
        path.write_text(json.dumps(cfg, indent=2) + "\n")
        print(f"converted {path.name}: {sorted(k for k in cfg['parameters'] if k.endswith('params'))}")


if __name__ == "__main__":
    main(sys.argv[1:])
