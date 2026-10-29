"""Unit tests for the instruction builder: folding, ranges, value numbering."""

import numpy as np
import pytest

from emmy_ptx import elementwise
from emmy_ptx.builder import Builder, Value, eliminate_dead_code
from emmy_ptx.ptx import Imm, Instr, Kernel, Reg


def make():
    k = Kernel("t", ())
    return k, Builder(k)


def ranged(b, lo, hi):
    """A register-held int known to lie in [lo, hi] (like a thread index)."""
    return b.special("%tid.x", lo, hi)


def ops(k):
    return [i.op for i in k.body if isinstance(i, Instr)]


@pytest.mark.parametrize("a,c", [(7, 2), (-7, 2), (7, -2), (-7, -2), (0, 5), (-1, 3)])
def test_c_division_semantics(a, c):
    _, b = make()
    q, r = b.div(b.int_const(a), b.int_const(c)), b.rem(b.int_const(a), b.int_const(c))
    assert q.const == int(a / c)  # truncates toward zero, like C
    assert r.const == a - int(a / c) * c  # takes the dividend's sign
    assert q.const * c + r.const == a


def test_power_of_two_divide_on_nonnegative_range_is_a_shift():
    k, b = make()
    x = ranged(b, 0, 1023)
    q = b.div(x, b.int_const(128))
    m = b.rem(x, b.int_const(128))
    assert (q.lo, q.hi) == (0, 7)
    assert (m.lo, m.hi) == (0, 127)
    assert "shr.u32" in ops(k) and "and.b32" in ops(k)
    assert not any(op.startswith(("div", "rem")) for op in ops(k))


def test_signed_divide_is_not_turned_into_a_shift():
    k, b = make()
    x = b.sub(ranged(b, 0, 10), b.int_const(5))  # may be negative
    b.div(x, b.int_const(4))
    assert "div.s32" in ops(k)


def test_remainder_below_divisor_is_dropped():
    _, b = make()
    x = ranged(b, 0, 63)
    assert b.rem(x, b.int_const(64)) is x


def test_quotient_with_single_possible_value_folds():
    _, b = make()
    assert b.div(ranged(b, 0, 127), b.int_const(128)).const == 0


def test_comparisons_decided_by_ranges():
    _, b = make()
    x = ranged(b, 0, 255)
    assert b.icmp("<", x, b.int_const(256)).const is True
    assert b.icmp(">=", x, b.int_const(256)).const is False
    assert b.icmp("<", x, b.int_const(100)).const is None


def test_value_numbering_reuses_and_respects_scopes():
    k, b = make()
    x = ranged(b, 0, 99)
    first = b.mul(x, b.int_const(3))
    assert b.mul(b.int_const(3), x) is first  # commutative operands normalised
    with b.cse_scope():
        inner = b.add(x, b.int_const(7))
        assert b.add(x, b.int_const(7)) is inner
    assert b.add(x, b.int_const(7)) is not inner  # the branch body did not dominate this point


def test_mutable_reads_are_never_reused():
    _, b = make()
    var = b.variable("s32", b.int_const(0))
    one = b.add(var.read(), b.int_const(1))
    b.assign(var, one)
    assert b.add(var.read(), b.int_const(1)) is not one


def test_overflowing_range_widens_to_64_bits():
    _, b = make()
    big = b.mul(ranged(b, 0, 1 << 20), b.int_const(1 << 12))
    assert big.type == "s64"
    assert big.hi == (1 << 32)


def test_mask_bounds_hold_for_negative_operands():
    _, b = make()
    x = b.bits(Value("f32", Reg("%f1", "f32")))  # any 32-bit pattern
    m = b.bitop("&", x, b.int_const(0x7FFFFF))
    assert (m.lo, m.hi) == (0, 0x7FFFFF)


def test_float_constants_fold_with_ieee_single_rounding():
    _, b = make()
    s = b.fbin("+", b.f32_const(1.0), b.f32_const(2.0 ** -30))
    assert s.const == 1.0  # 1 + 2^-30 rounds back to 1 in float32


def test_float64_arithmetic_stays_in_double_precision():
    k, b = make()
    s = b.fbin("+", b.float_const(0.1, "f64"), b.float_const(0.2, "f64"))
    assert s.type == "f64" and s.const == 0.1 + 0.2  # folded in double, not via float32
    x = b.pure("ld.global.f64", "f64", (Imm(0, "u64"),))
    y = b.fbin("*", x, b.f32_const(0.5))  # an f32 operand is widened, not x narrowed
    z = b.fbin("/", x, b.float_const(2.0**-600, "f64"))  # 1/2^-600 is a normal double: a multiply
    assert (y.type, z.type) == ("f64", "f64")
    assert b.fcmp("<", x, b.f32_const(1.0)).type == "pred"
    assert b.select(b.fcmp("<", x, y), x, b.f32_const(0.0)).type == "f64"
    assert b.convert(b.int_const(3), "f64").const == 3.0
    assert ops(k) == ["ld.global.f64", "mul.rn.f64", "mul.rn.f64", "setp.lt.f64", "setp.lt.f64", "selp.f64"]


def test_a_bool_is_one_for_any_non_zero_value():
    """C's conversion to bool, folded for constants (2, 0.5 and NaN give 1,
    -0.0 gives 0) and otherwise a test against zero that keeps a NaN true."""
    k, b = make()
    for x, want in ((b.int_const(2), 1), (b.f32_const(0.5), 1), (b.f32_const(float("nan")), 1),
                    (b.f32_const(-0.0), 0), (b.int_const(256), 1)):
        assert elementwise.declared(b, x, "bool").const == want
    flag = elementwise.declared(b, b.pure("ld.global.f32", "f32", (Imm(0, "u64"),)), "bool")
    assert (flag.lo, flag.hi) == (0, 1) and ops(k) == ["ld.global.f32", "setp.neu.f32", "selp.b32"]


def test_dead_code_elimination_keeps_effects():
    k, b = make()
    x = ranged(b, 0, 9)
    b.mul(x, b.int_const(5))  # never used
    b.emit("st.global.u32", Imm(0, "u32"), x)
    eliminate_dead_code(k)
    assert ops(k) == ["mov.u32", "st.global.u32"]


@pytest.mark.parametrize("c", [3.0, 7.0, -0.1, 6.0, 777.0, 1e-38, 3e38, 1.1754942e-38])
def test_double_product_with_the_reciprocal_rounds_like_float_division(c):
    """Lowerer._quotient: in double, x * RN(1/c) rounded to float32 is the
    correctly rounded float32 x / c, for every x (checked here on 4M random
    bit patterns, both extremes of every exponent, and the special values)."""
    rng = np.random.default_rng(int(abs(c) * 1e6) % 2**32)
    bits = rng.integers(0, 2**32, 1 << 22, dtype=np.uint64).astype(np.uint32)
    edges = np.array([0, 0x80000000, 1, 0x007FFFFF, 0x00800000, 0x7F7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000],
                     dtype=np.uint32)
    exps = (np.arange(256, dtype=np.uint32) << 23)
    x = np.concatenate([bits, edges, exps, exps | 0x7FFFFF]).view(np.float32)
    c32 = np.float32(c)
    with np.errstate(all="ignore"):
        want = x / c32
        got = (x.astype(np.float64) * (1.0 / float(c32))).astype(np.float32)
    same = (got.view(np.uint32) == want.view(np.uint32)) | (np.isnan(got) & np.isnan(want))
    assert same.all(), x[~same][:5]


def test_a_store_into_a_bool_array_writes_0_or_1():
    """Each element of a bool array is one byte, 0 or 1: storing 0.5 stores 1
    (truncated, it would store 0) and 256 stores 1 (its low byte is 0)."""
    from emmy.compiler.dim import Dim
    from emmy.compiler.ir.axis import Axis
    from emmy.compiler.ir.expr import Var
    from emmy.compiler.ir.kernel.ir import Tile
    from emmy.compiler.ir.stmt import Load, Write

    from emmy_ptx import PTX_VERSION, TARGET
    from emmy_ptx.frontend import Buffer, KernelSource, Launch
    from emmy_ptx.lower import Lowerer

    body = (Tile(axes=(Axis("t", Dim(32)),), body=(Load(name="v", input="x", index=(Var("t"),)),
                                                     Write("flags", (Var("t"),), "v"))),)
    buffers = {"x": Buffer("x", (32,), "f32"), "flags": Buffer("flags", (32,), "bool")}
    src = KernelSource("k", body, ("x", "flags"), buffers, Launch(1, 32, 0), frozenset({"flags"}))
    for options in ({}, {"plain": True}):
        ptx = Lowerer(src, **options).run().render(PTX_VERSION, TARGET)
        assert "setp.neu.f32" in ptx and "st.global.u8" in ptx and "cvt.rzi" not in ptx, ptx
