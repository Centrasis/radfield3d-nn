from RadFiled3D.RadFiled3D import FieldStore
import zipfile
from rich.progress import Progress
from rich import print
from pathlib import Path
import os


def test_file(file: str, zip_ref: zipfile.ZipFile, progressbar: Progress, task):
    progressbar.advance(task)
    try:
        with zip_ref.open(file) as f:
            field = FieldStore.load_from_buffer(f.read())
        return None
    except zipfile.BadZipFile:
        return test_file(file, zip_ref, progressbar, task)
    except zipfile.LargeZipFile:
        return test_file(file, zip_ref, progressbar, task)
    except Exception as e:
        print(e)
        return file
    

def write_file(file: str, zip_ref: zipfile.ZipFile, new_zip_ref: zipfile.ZipFile, progressbar: Progress, task):
    data = None
    with zip_ref.open(file) as f:
        data = f.read()

    new_zip_ref.writestr(file, data)
    progressbar.advance(task)


if __name__ == "__main__":
    base_path = Path(r"\\hpc.isc.pad.ptb.de\user\lehner04\Promotion\Datasets")
    #base_path = Path(r"C:\Users\lehner04\Documents\Datasets")
    path = base_path / "Cylinder-Multi-Spectra.zip"
    new_file = base_path / "Cylinder-Multi-Spectra.filtered.zip"
    bad_files_count = 0
    files_count = 0
    bad_files = []
    print("[yellow]Reading file list...")
    with zipfile.ZipFile(path, 'r') as zip_ref:
        all_files = [f for f in zip_ref.namelist() if f.endswith(".rf")]

    if os.path.exists(new_file):
        os.remove(new_file)
    
    print("[green]Success!")
    with Progress() as progress:
        task = progress.add_task("[green]Testing dataset...", total=len(all_files))
        with zipfile.ZipFile(path, 'r') as zip_ref:
            with zipfile.ZipFile(new_file, 'w', compression=zipfile.ZIP_DEFLATED) as new_zip_ref:
                for file in all_files:
                    res = test_file(file, zip_ref, progress, task)
                    if res is not None:
                        bad_files_count += 1
                        bad_files.append(res)
                    else:
                        files_count += 1
                        while True:
                            try:
                                write_file(file, zip_ref, new_zip_ref, progress, task)
                                break
                            except Exception as e:
                                print(e)
                                continue
    print(f"[green]Success! {files_count} files are valid and {bad_files_count} files are invalid.")
    print(f"Invalid file ratio: {bad_files_count / (files_count + bad_files_count) * 100:.2f}%")
