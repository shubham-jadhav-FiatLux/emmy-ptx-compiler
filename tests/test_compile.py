"""End-to-end compile tests over problem folders.

Problem folders are looked up in $EMMY_PROBLEMS (one or more paths separated
by os.pathsep), defaulting to the hackathon repository checked out next to
this one. When ptxas is on PATH (or $PTXAS points at it) every module is also
assembled for sm_120, which catches any PTX the runner would reject. Every
kernel is compiled three ways: as usual, with ``--strict`` (no rewrite that
changes rounding), and with the fallback lowering (``--plain``).
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import emmy_ptx
from emmy_ptx import compile_file, lower

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT.parent / "upstream" / "emmy-hackathon-2026" / "problems"
SEARCH = [Path(p) for p in os.environ.get("EMMY_PROBLEMS", str(DEFAULT)).split(os.pathsep)]
KERNELS = sorted(ir for d in SEARCH if d.is_dir() for ir in d.glob("*/*.ir.json"))
PTXAS = os.environ.get("PTXAS") or shutil.which("ptxas")

pytestmark = pytest.mark.skipif(not KERNELS, reason="no problem folders found (set EMMY_PROBLEMS)")


def compile_ir(ir: Path, flags: tuple = ()) -> str:
    cmd = [sys.executable, str(ROOT / "compile.py"), *flags, str(ir)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


@pytest.mark.parametrize("flags", [(), ("--strict",), ("--plain",)], ids=["optimized", "strict", "plain"])
@pytest.mark.parametrize("ir", KERNELS, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_kernel_compiles_to_well_formed_ptx(ir, flags, tmp_path):
    ptx = compile_ir(ir, flags)
    meta = json.loads((ir.parent / "meta.json").read_text())
    entry = ir.name.removesuffix(".ir.json")
    kernel = next(k for k in meta["kernels"] if k["entry"] == entry)

    assert ptx.lstrip().startswith("//") and "\n.version " in ptx
    assert ptx.isascii()
    params = re.search(rf"\.visible \.entry {re.escape(entry)}\(([^)]*)\)", ptx).group(1)
    assert re.findall(r"\.param \.u64 p_(\w+)", params) == kernel["arg_order"]

    if PTXAS:
        out = tmp_path / "k.ptx"
        out.write_text(ptx)
        r = subprocess.run([PTXAS, "-arch=sm_120", str(out), "-o", str(tmp_path / "k.cubin")],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


def test_a_failing_rewrite_falls_back_to_the_plain_lowering(monkeypatch, capsys):
    """A rewrite that meets IR it did not foresee costs the kernel its
    optimizations, not its PTX."""

    def unforeseen(*args):
        raise RuntimeError("an IR shape no rewrite foresaw")

    monkeypatch.setattr(lower, "_issue_loads_early", unforeseen)  # every optimized lowering calls it
    ir = str(KERNELS[0])
    assert compile_file(ir) == compile_file(ir, plain=True)
    err = capsys.readouterr().err
    assert "RuntimeError: an IR shape no rewrite foresaw" in err and "without the optional rewrites" in err


def test_a_module_too_long_for_the_runner_falls_back(monkeypatch, capsys):
    monkeypatch.setattr(emmy_ptx, "PTX_LIMIT", 100)
    ir = str(KERNELS[0])
    assert compile_file(ir) == compile_file(ir, plain=True)
    assert "more than the runner accepts" in capsys.readouterr().err


@pytest.mark.parametrize("flags", [{}, {"strict": True}, {"plain": True}], ids=["optimized", "strict", "plain"])
def test_report_lists_the_decisions_and_changes_no_ptx(flags, capsys):
    """``--report`` writes the lowering's decisions and a PTX profile to stderr,
    and the PTX is the same with or without it."""
    ir = str(next((k for k in KERNELS if k.parent.name == "r01_array_sum"), KERNELS[0]))
    ptx = compile_file(ir, **flags)
    capsys.readouterr()
    assert compile_file(ir, report=True, **flags) == ptx
    err = capsys.readouterr().err
    name = Path(ir).name.removesuffix(".ir.json")
    assert err.startswith(f"{name}: ") and "\n  PTX: " in err and "instructions:" in err
    assert ("launch hints: .maxntid" in err) and ("lowered plainly" in err) == bool(flags.get("plain"))
