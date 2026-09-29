"""
A HALF-WRITTEN FILE IS NOT EVIDENCE, AND IT IS NOT FATAL.

The operator's 2026-09-29 log carried this on every cycle:

    Qualification load failed: Expecting value: line 507 column 18 (char 23042)

`data/venue_qualification.json` had been cut off mid-write - which is exactly
what pressing Stop during a save leaves behind, because `_save()` opened the
file with "w" (truncating it) and then streamed JSON into it. The load then
failed, the whole record was discarded, and nothing repaired the file: the same
warning came back every cycle while every venue read as "never measured".

These tests pin all three halves of the fix:

  * writes are atomic (temp file + os.replace), so a killed process leaves the
    OLD complete file or the NEW complete file and never a half one;
  * a truncated file is read up to its last COMPLETE record, so real evidence is
    not thrown away with the tail;
  * a file that cannot be salvaged is quarantined, named, and replaced - the
    failure is reported once instead of on every cycle, and no venue is ever
    counted as qualified on unreadable evidence.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.ptai.venues.qualification import (QualificationResult,
                                           VenueQualificationEngine)


def _record(venue_id: str, **overrides) -> QualificationResult:
    fields = dict(venue_id=venue_id, win_rate=0.6, avg_edge=0.05,
                  brier_score=0.2, profit_paper=12.0, profit_live=0.0,
                  forecast_skill=0.1, is_qualified=False,
                  qualification_date=datetime.now(timezone.utc),
                  requirements={}, reasoning="test record")
    fields.update(overrides)
    return QualificationResult(**fields)


def _engine(tmp_path: Path, venues=("venue_a", "venue_b", "venue_c")):
    engine = VenueQualificationEngine(data_dir=str(tmp_path))
    for venue in venues:
        engine.qualifications[venue] = _record(venue)
    engine._save()
    return engine


def _truncate(path: Path, drop: int = 40) -> str:
    """Cut the file off inside its last record, like a stopped save does."""
    text = path.read_text()
    path.write_text(text[:-drop])
    return text


# ----------------------------------------------------------------------
# writes
# ----------------------------------------------------------------------

class TestWritesAreAtomic:
    def test_a_save_leaves_valid_json(self, tmp_path):
        engine = _engine(tmp_path)
        data = json.loads((tmp_path / "venue_qualification.json").read_text())
        assert sorted(data) == ["venue_a", "venue_b", "venue_c"]

    def test_a_save_that_dies_mid_write_leaves_the_old_file_intact(
            self, tmp_path, monkeypatch):
        engine = _engine(tmp_path)
        path = tmp_path / "venue_qualification.json"
        before = path.read_text()

        real_dump = json.dump

        def exploding_dump(obj, f, **kwargs):
            f.write('{"venue_a": {"venue_id": "ven')  # a half record...
            raise OSError("the process was stopped mid-write")

        monkeypatch.setattr(json, "dump", exploding_dump)
        engine.qualifications["venue_d"] = _record("venue_d")
        engine._save()
        monkeypatch.setattr(json, "dump", real_dump)

        # the previous complete file is still there, still parseable
        assert path.read_text() == before
        assert sorted(json.loads(path.read_text())) == ["venue_a", "venue_b",
                                                        "venue_c"]

    def test_no_temp_file_is_left_behind(self, tmp_path):
        _engine(tmp_path)
        assert list(tmp_path.glob("*.tmp")) == []


# ----------------------------------------------------------------------
# a truncated file
# ----------------------------------------------------------------------

class TestATruncatedFileKeepsWhatWasComplete:
    def test_the_complete_records_are_recovered(self, tmp_path):
        engine = _engine(tmp_path)
        _truncate(tmp_path / "venue_qualification.json")
        reloaded = VenueQualificationEngine(data_dir=str(tmp_path))
        # the tail is gone, the finished records are not
        assert "venue_a" in reloaded.qualifications
        assert "venue_b" in reloaded.qualifications

    def test_the_record_is_not_silently_empty(self, tmp_path):
        _engine(tmp_path)
        _truncate(tmp_path / "venue_qualification.json")
        reloaded = VenueQualificationEngine(data_dir=str(tmp_path))
        assert reloaded.qualifications, ("losing every record to a cut-off "
                                         "tail is the defect this fixes")

    def test_the_next_save_rewrites_a_valid_file(self, tmp_path):
        _engine(tmp_path)
        _truncate(tmp_path / "venue_qualification.json")
        reloaded = VenueQualificationEngine(data_dir=str(tmp_path))
        reloaded._save()
        text = (tmp_path / "venue_qualification.json").read_text()
        assert isinstance(json.loads(text), dict)

    def test_a_truncated_file_is_never_reported_as_valid_evidence(self, tmp_path,
                                                                  capsys):
        from loguru import logger
        lines = []
        sink = logger.add(lambda m: lines.append(m.record["message"]),
                          level="WARNING")
        try:
            _engine(tmp_path)
            _truncate(tmp_path / "venue_qualification.json")
            VenueQualificationEngine(data_dir=str(tmp_path))
        finally:
            logger.remove(sink)
        assert any("cut short" in line for line in lines), lines


# ----------------------------------------------------------------------
# a file that cannot be salvaged
# ----------------------------------------------------------------------

class TestAnUnreadableFileIsQuarantined:
    def test_garbage_is_moved_aside_and_named(self, tmp_path):
        path = tmp_path / "venue_qualification.json"
        path.write_text("this is not json at all")
        engine = VenueQualificationEngine(data_dir=str(tmp_path))
        assert engine.qualifications == {}
        assert not path.exists(), "the unreadable file must not stay in the way"
        kept = list(tmp_path.glob("venue_qualification.json.corrupt-*"))
        assert len(kept) == 1, "the evidence is kept, not deleted"
        assert kept[0].read_text() == "this is not json at all"

    def test_a_json_array_is_not_a_venue_record(self, tmp_path):
        path = tmp_path / "venue_qualification.json"
        path.write_text("[1, 2, 3]")
        engine = VenueQualificationEngine(data_dir=str(tmp_path))
        assert engine.qualifications == {}
        assert list(tmp_path.glob("*.corrupt-*"))

    def test_nothing_is_qualified_from_an_unreadable_file(self, tmp_path):
        path = tmp_path / "venue_qualification.json"
        path.write_text('{"polymarket": {"venue_id": "polym')
        engine = VenueQualificationEngine(data_dir=str(tmp_path))
        assert all(not q.is_qualified for q in engine.qualifications.values())

    def test_the_failure_is_reported_once_not_per_cycle(self, tmp_path):
        from loguru import logger
        lines = []
        sink = logger.add(lambda m: lines.append(m.record["message"]),
                          level="WARNING")
        try:
            (tmp_path / "venue_qualification.json").write_text("{oops")
            VenueQualificationEngine(data_dir=str(tmp_path))
            # a second engine on the repaired directory has nothing to report
            lines.clear()
            VenueQualificationEngine(data_dir=str(tmp_path))
        finally:
            logger.remove(sink)
        assert lines == []


# ----------------------------------------------------------------------
# one bad record must not take the good ones with it
# ----------------------------------------------------------------------

class TestOneBadRecordIsNotTheWholeFile:
    def test_a_record_from_an_older_version_is_skipped_by_name(self, tmp_path):
        path = tmp_path / "venue_qualification.json"
        path.write_text(json.dumps({
            "good": {"venue_id": "good", "win_rate": 0.6, "avg_edge": 0.05,
                     "brier_score": 0.2, "profit_paper": 1.0, "profit_live": 0.0,
                     "forecast_skill": 0.1, "is_qualified": False,
                     "qualification_date": None, "requirements": {},
                     "reasoning": "ok"},
            # missing every required field: nothing can be constructed
            "bad": {"venue_id": "bad"},
        }, indent=2))
        engine = VenueQualificationEngine(data_dir=str(tmp_path))
        assert "good" in engine.qualifications
        assert "bad" not in engine.qualifications

    def test_the_skipped_record_is_named_in_the_log(self, tmp_path):
        from loguru import logger
        lines = []
        sink = logger.add(lambda m: lines.append(m.record["message"]),
                          level="WARNING")
        try:
            (tmp_path / "venue_qualification.json").write_text(json.dumps({
                "bad": {"venue_id": "bad"}}, indent=2))
            VenueQualificationEngine(data_dir=str(tmp_path))
        finally:
            logger.remove(sink)
        assert any("bad" in line and "could not be read" in line
                   for line in lines), lines


# ----------------------------------------------------------------------
# the happy path still works
# ----------------------------------------------------------------------

class TestTheRecordStillRoundTrips:
    def test_a_saved_engine_reloads_identical_verdicts(self, tmp_path):
        engine = _engine(tmp_path)
        engine.qualifications["venue_a"] = _record("venue_a", is_qualified=True,
                                                   win_rate=0.71)
        engine._save()
        reloaded = VenueQualificationEngine(data_dir=str(tmp_path))
        assert reloaded.qualifications["venue_a"].is_qualified is True
        assert reloaded.qualifications["venue_a"].win_rate == 0.71

    def test_no_file_at_all_is_not_an_error(self, tmp_path):
        engine = VenueQualificationEngine(data_dir=str(tmp_path))
        assert engine.qualifications == {}
