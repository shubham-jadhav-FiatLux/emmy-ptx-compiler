"""Unit tests for the instruction builder: folding, ranges, value numbering."""

import pytest

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


def test_dead_code_elimination_keeps_effects():
    k, b = make()
    x = ranged(b, 0, 9)
    b.mul(x, b.int_const(5))  # never used
    b.emit("st.global.u32", Imm(0, "u32"), x)
    eliminate_dead_code(k)
    assert ops(k) == ["mov.u32", "st.global.u32"]
