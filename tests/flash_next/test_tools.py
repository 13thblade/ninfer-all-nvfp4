"""The flash_next diagnostic tools' input handling (no model or artifact needed)."""

from __future__ import annotations

import numpy as np
import pytest

from tools.flash_next import reference, strata_compare


def write_dump(path, vocab, rows, payload_rows=None):
    """A dump whose header claims [rows, vocab] over payload_rows (default: rows) rows of logits."""
    header = np.array([vocab, rows], dtype="<i4").tobytes()
    logits = np.zeros((rows if payload_rows is None else payload_rows, max(vocab, 0)), dtype="<f4")
    path.write_bytes(header + logits.tobytes())
    return path


def test_engine_rows_reads_a_consistent_dump(tmp_path):
    rows = strata_compare.engine_rows(write_dump(tmp_path / "ok.bin", 7, 3))
    assert rows.shape == (3, 7)


@pytest.mark.parametrize(
    ("vocab", "rows", "payload_rows"),
    [(7, 3, 2), (7, 3, 4), (0, 3, 0), (7, -1, 0), (7, 0, 0)],
)
def test_engine_rows_refuses_a_header_that_disagrees(tmp_path, vocab, rows, payload_rows):
    path = write_dump(tmp_path / "bad.bin", vocab, rows, payload_rows)
    with pytest.raises(ValueError, match="header says"):
        strata_compare.engine_rows(path)


def test_engine_rows_refuses_a_truncated_header(tmp_path):
    path = tmp_path / "short.bin"
    path.write_bytes(b"\x01\x00\x00")
    with pytest.raises(ValueError, match="shorter than the 8-byte header"):
        strata_compare.engine_rows(path)


def strata_row(pos, target, top=0, topk=((0, -0.1),)):
    return {"pos": pos, "target": target, "logprob": -1.0, "top": top, "topk": list(topk)}


def test_compare_accepts_matching_inputs():
    logits = np.zeros((3, 5), dtype=np.float32)
    result = strata_compare.compare([1, 2, 3], logits, [strata_row(0, 2), strata_row(1, 3)])
    assert result["positions"] == 2


@pytest.mark.parametrize(
    ("ids", "rows", "strata", "message"),
    [
        ([1, 2, 3], 3, [strata_row(2, 3)], "outside the text's predictions"),
        ([1, 2, 3], 3, [strata_row(-1, 3)], "outside the text's predictions"),
        ([1, 2, 3], 1, [strata_row(1, 3)], "has no row"),
        ([1, 2, 3], 3, [strata_row(0, 4)], "targets 4"),
        ([1, 9, 3], 3, [strata_row(0, 9)], "outside the dump's 5-token vocabulary"),
        ([1, 2, 3], 3, [strata_row(0, 2, topk=((7, -0.1),))], "outside the dump's 5-token vocabulary"),
        ([1, 2, 3], 3, [], "scored no position"),
    ],
)
def test_compare_refuses_inputs_that_disagree(ids, rows, strata, message):
    with pytest.raises(ValueError, match=message):
        strata_compare.compare(ids, np.zeros((rows, 5), dtype=np.float32), strata)


@pytest.mark.parametrize(
    ("flags", "expected"),
    [([], True), (["--bf16-boundaries"], True), (["--no-bf16-boundaries"], False)],
)
def test_reference_accepts_the_documented_boundary_flags(flags, expected):
    args = reference.build_parser().parse_args(["--artifact", "a", "--tokens", "1", "--out", "o", *flags])
    assert args.bf16_boundaries is expected
