"""Preflight scan for the patient-translation metadata a translation dataset requires.

RadField3DTranslationDataset reads ``metadata.simulation.patient_translation`` from every field and
raises when it is absent. A dataset that mixes fields from generation runs with and without patient
translation therefore fails at a RANDOM step, inside a DataLoader worker, once training has already
been running for minutes — the traceback names the field type but not which files are at fault.

This module answers that question up front: it reads only the METADATA of each field (cheap next to
a full decode), so a run either starts knowing the dataset is uniform or reports exactly how many
fields and which ones are missing the entry.
"""
import os
from typing import Iterable

from RadFiled3D.utils import FieldStore


def field_has_translation(path: str) -> bool:
    """True when this field carries a patient_translation entry in its dynamic metadata.

    A field that cannot be read at all counts as MISSING rather than raising: an unreadable field
    is a problem for the run either way, and the caller reports it with the rest.
    """
    try:
        metadata = FieldStore.load_metadata(path)
        return getattr(metadata.simulation, "patient_translation", None) is not None
    except Exception:
        return False


def write_missing_report(missing: list[str], directory: str = None,
                         filename: str = "missing_patient_translation.txt") -> str | None:
    """Write EVERY offending field path, one per line, so the list can be acted on directly
    (``xargs rm < missing_patient_translation.txt``). Returns the report path, or None if it could
    not be written (a read-only or missing directory must not abort the run)."""
    if not missing:
        return None
    directory = directory or os.getcwd()
    path = os.path.join(directory, filename)
    try:
        with open(path, "w") as f:
            f.write("\n".join(missing) + "\n")
        return path
    except OSError:
        return None


def resolve_missing_translation(dataset_path: str, candidates: list[str], missing: list[str],
                                skip_fields_without_translation: bool,
                                report_path: str = None) -> set:
    """Decide what to do with fields lacking the metadata: return the set to EXCLUDE, or raise.

    Failing is the default because a mixed dataset is usually an accident of generation, and
    silently training on a subset hides it. ``skip_fields_without_translation`` opts into the
    subset explicitly. The COMPLETE list of offenders is in ``report_path`` (and printed by the
    caller) either way — the exception text carries a short sample plus that pointer.
    """
    if not missing:
        return set()
    if skip_fields_without_translation:
        return set(missing)
    sample = "\n  ".join(os.path.basename(p) for p in missing[:5])
    more = f"\n  … and {len(missing) - 5} more" if len(missing) > 5 else ""
    where = f"\nFull list of all {len(missing)} field(s): {report_path}" if report_path else ""
    raise ValueError(
        f"use_translation=True, but {len(missing)} of {len(candidates)} fields in {dataset_path} "
        f"carry no 'patient_translation' metadata — training would die mid-epoch in a DataLoader "
        f"worker when one of them is drawn.\n  {sample}{more}{where}\n"
        f"Remove/regenerate those fields with patient translation enabled, or set "
        f"`dataset: skip_fields_without_translation: true` to train on the "
        f"{len(candidates) - len(missing)} fields that do carry it."
    )


def scan_fields_missing_translation(file_paths: Iterable[str], workers: int = None) -> list[str]:
    """Return the subset of ``file_paths`` whose fields lack the patient_translation metadata.

    Metadata-only reads, parallelised across ``workers`` processes (default: half the CPUs, which
    leaves room for the rest of the startup work).
    """
    paths = list(file_paths)
    if not paths:
        return []
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) // 2)
    if workers <= 1 or len(paths) < 32:
        return [p for p in paths if not field_has_translation(p)]
    from joblib import Parallel, delayed
    flags = Parallel(n_jobs=workers)(delayed(field_has_translation)(p) for p in paths)
    return [p for p, ok in zip(paths, flags) if not ok]
