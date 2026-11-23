"""Extract a dataset zip to a directory, skipping fields listed as unusable.

Training from the archive re-inflates each field several times per sample (the loader's buffer
cache holds one entry), which measured 4.4x slower per field than reading it from disk.
"""
import argparse, json, os, shutil, sys, zipfile
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _extract(args):
    zip_path, names, out_dir = args
    done = 0
    with zipfile.ZipFile(zip_path) as z:
        for n in names:
            dst = os.path.join(out_dir, os.path.basename(n))
            if os.path.exists(dst) and os.path.getsize(dst) == z.getinfo(n).file_size:
                done += 1
                continue
            tmp = dst + ".part"
            with z.open(n) as src, open(tmp, "wb") as out:
                shutil.copyfileobj(src, out, 1 << 22)
            os.replace(tmp, dst)
            done += 1
    return done


def main():
    ap = argparse.ArgumentParser(description="Extract a RadField3D dataset zip into a directory.")
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True, help="Dataset directory; fields land in <out>/fields.")
    ap.add_argument("--exclude_files", default=None, help="JSON with a 'fields' list to skip.")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    from radfield3dnn.datasets import list_dataset_field_names
    names = list_dataset_field_names(a.zip)
    if a.exclude_files:
        bad = set(json.load(open(a.exclude_files))["fields"])
        names = [n for n in names if n not in bad]
        print(f"skipping {len(bad)} excluded fields")
    fields_dir = os.path.join(a.out, "fields")
    os.makedirs(fields_dir, exist_ok=True)
    print(f"extracting {len(names)} fields to {fields_dir} with {a.workers} workers", flush=True)

    chunks = [(a.zip, names[i::a.workers], fields_dir) for i in range(a.workers)]
    total = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for got in ex.map(_extract, chunks):
            total += got
            print(f"  {total}/{len(names)}", flush=True)
    print(f"done: {total} fields in {fields_dir}")


if __name__ == "__main__":
    main()
