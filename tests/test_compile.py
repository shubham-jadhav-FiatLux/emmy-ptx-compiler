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


def test_explore_keeps_trying_while_a_smaller_budget_can_change_the_module(monkeypatch, capsys):
    """Eight chained sines in a 16-trip loop stay unrolled at budgets 1 and 1/4
    (the same module), and roll at 1/16 (a quarter of the size): a plateau is
    not the end while the budget still decides an unrolling. A module that
    stays large says so."""
    from emmy.compiler.dim import Dim
    from emmy.compiler.ir.axis import Axis
    from emmy.compiler.ir.expr import Var
    from emmy.compiler.ir.kernel.ir import Tile
    from emmy.compiler.ir.stmt import Accum, Assign, Load, Loop, Write

    from emmy_ptx.frontend import Buffer, KernelSource, Launch

    sines = [Load(name="v0", input="x", index=(Var("t"), Var("i")))]
    sines += [Assign(f"v{k + 1}", "sin", (f"v{k}",)) for k in range(8)] + [Accum("s", "v8")]
    body = (Tile(axes=(Axis("t", Dim(32)),), block_threads=32, body=(
        Loop(axis=Axis("i", Dim(16)), body=tuple(sines)), Write("out", (Var("t"),), "s"))),)
    src = KernelSource("k_sines", body, ("x", "out"), {"x": Buffer("x", (32, 16), "f32"),
                       "out": Buffer("out", (32,), "f32")}, Launch(1, 32, 0), frozenset({"out"}))
    monkeypatch.setattr(emmy_ptx, "load", lambda path: src)
    full = compile_file("k_sines")
    assert len(full) > emmy_ptx.EXPLORE_PTX_CHARS
    assert len(compile_file("k_sines", explore=True, report=True)) <= emmy_ptx.EXPLORE_PTX_CHARS
    assert "complete unrolling scaled by 0.0625" in capsys.readouterr().err
    monkeypatch.setattr(emmy_ptx, "EXPLORE_PTX_CHARS", 1000)  # beyond reach: nothing optional is left to roll
    assert len(compile_file("k_sines", explore=True, report=True)) > 1000
    assert "target not met" in capsys.readouterr().err
