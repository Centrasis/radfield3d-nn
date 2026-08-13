"""Preflight scan for the patient_translation metadata RadField3DTranslationDataset requires.

The metadata read is stubbed (building real .rf3 fields in a unit test is not worth it); the access
path mirrors RadField3DTranslationDataset._build_input, i.e. metadata.simulation.patient_translation.
"""
import types

import pytest

try:
    import RadFiled3D  # noqa: F401
except ImportError:  # pragma: no cover
    pytest.skip("RadFiled3D not available", allow_module_level=True)

from radfield3dnn.datasets import translation_scan as ts


def _meta(translation):
    return types.SimpleNamespace(simulation=types.SimpleNamespace(patient_translation=translation))


@pytest.fixture
def stub_store(monkeypatch):
    """Map path -> metadata; a path not in the map raises, like an unreadable field."""
    table = {}

    class _FieldStore:
        @staticmethod
        def load_metadata(path):
            if path not in table:
                raise OSError(f"unreadable: {path}")
            return table[path]

    monkeypatch.setattr(ts, "FieldStore", _FieldStore)
    return table


def test_field_with_translation_passes(stub_store):
    stub_store["ok.rf3"] = _meta(types.SimpleNamespace(x=0.1, y=-0.2, z=0.0))
    assert ts.field_has_translation("ok.rf3") is True


def test_field_without_translation_fails(stub_store):
    stub_store["bad.rf3"] = _meta(None)
    assert ts.field_has_translation("bad.rf3") is False


def test_unreadable_field_counts_as_missing(stub_store):
    # Not in the table -> load_metadata raises. An unreadable field breaks the run either way,
    # so it is reported with the rest instead of exploding the scan.
    assert ts.field_has_translation("gone.rf3") is False


def test_scan_reports_only_the_offending_files(stub_store):
    for i in range(6):
        stub_store[f"f{i}.rf3"] = _meta(types.SimpleNamespace(x=0.0, y=0.0, z=0.0) if i % 2 == 0 else None)
    missing = ts.scan_fields_missing_translation([f"f{i}.rf3" for i in range(6)], workers=1)
    assert missing == ["f1.rf3", "f3.rf3", "f5.rf3"]


def test_scan_of_empty_list_is_empty():
    assert ts.scan_fields_missing_translation([], workers=1) == []


def test_resolve_raises_with_actionable_message():
    candidates = [f"/ds/f{i}.rf3" for i in range(10)]
    missing = candidates[:7]
    with pytest.raises(ValueError) as err:
        ts.resolve_missing_translation("/ds", candidates, missing, skip_fields_without_translation=False,
                                       report_path="/logs/missing_patient_translation.txt")
    msg = str(err.value)
    assert "7 of 10" in msg                              # scale of the problem
    assert "skip_fields_without_translation" in msg      # the knob
    assert "f0.rf3" in msg and "and 2 more" in msg       # a sample, not a wall of names
    assert "/logs/missing_patient_translation.txt" in msg  # where the COMPLETE list lives


def test_report_file_lists_every_offender(tmp_path):
    missing = [f"/ds/fields/f{i}.rf3" for i in range(120)]
    path = ts.write_missing_report(missing, directory=str(tmp_path))
    assert path is not None
    written = open(path).read().splitlines()
    assert written == missing                            # ALL of them, one per line, no truncation


def test_report_file_write_failure_does_not_abort():
    # A read-only / missing directory must not take the run down; the caller still prints the list.
    assert ts.write_missing_report(["/ds/a.rf3"], directory="/nonexistent-dir-xyz") is None


def test_report_of_nothing_is_none(tmp_path):
    assert ts.write_missing_report([], directory=str(tmp_path)) is None


def test_resolve_returns_exclusions_when_skipping():
    candidates = [f"/ds/f{i}.rf3" for i in range(4)]
    missing = ["/ds/f1.rf3"]
    assert ts.resolve_missing_translation("/ds", candidates, missing,
                                          skip_fields_without_translation=True) == {"/ds/f1.rf3"}


def test_resolve_is_a_noop_when_nothing_is_missing():
    assert ts.resolve_missing_translation("/ds", ["/ds/a.rf3"], [], False) == set()
