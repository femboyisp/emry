"""Tests for emry.coerce — scalar coercion without hard torch/numpy deps."""

import math

import pytest

from emry.coerce import coerce_metrics, to_float


def test_python_numbers():
    assert to_float(3) == 3.0
    assert to_float(2.5) == 2.5
    assert isinstance(to_float(3), float)


def test_bool_coerces_to_one_or_zero():
    assert to_float(True) == 1.0
    assert to_float(False) == 0.0


def test_non_finite_passes_through():
    assert math.isnan(to_float(float("nan")))
    assert math.isinf(to_float(float("inf")))


class _Scalar:
    """Stand-in for a 0-dim tensor / numpy scalar: has a scalar .item()."""

    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class _NonScalar:
    """Stand-in for a multi-element tensor: .item() raises ValueError."""

    def item(self):
        raise ValueError("only one element tensors can be converted to scalars")


def test_item_bearing_scalar():
    assert to_float(_Scalar(0.25)) == 0.25
    assert to_float(_Scalar(7)) == 7.0


def test_non_scalar_item_raises_typeerror():
    with pytest.raises(TypeError, match="non-scalar"):
        to_float(_NonScalar())


class _FloatLike:
    def __float__(self):
        return 1.5


def test_float_dunder_fallback():
    assert to_float(_FloatLike()) == 1.5


@pytest.mark.parametrize("bad", [None, "loss", [1, 2], {}])
def test_unconvertible_values_raise(bad):
    with pytest.raises(TypeError):
        to_float(bad)


def test_coerce_metrics_mapping():
    out = coerce_metrics({"loss": 0.5, "step": 3, "acc": _Scalar(0.9)})
    assert out == {"loss": 0.5, "step": 3.0, "acc": 0.9}
    assert all(isinstance(v, float) for v in out.values())


def test_coerce_metrics_names_offending_key():
    with pytest.raises(TypeError, match="'bad'"):
        coerce_metrics({"good": 1.0, "bad": None})


def test_numpy_scalars_when_available():
    np = pytest.importorskip("numpy")
    assert to_float(np.float32(2.0)) == 2.0
    assert to_float(np.int64(5)) == 5.0
    assert to_float(np.array(3.0)) == 3.0  # 0-dim array
    with pytest.raises(TypeError):
        to_float(np.array([1.0, 2.0]))  # multi-element


def test_torch_tensors_when_available():
    torch = pytest.importorskip("torch")
    assert to_float(torch.tensor(0.5)) == 0.5
    with pytest.raises(TypeError):
        to_float(torch.tensor([1.0, 2.0]))
