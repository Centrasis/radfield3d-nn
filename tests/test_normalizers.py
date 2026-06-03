import torch
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import models
from normalizations.lognormalizer import LogNormalizer
from normalizations.linear import LinearNormalizer
from normalizations.special_fits import SpecialPolynomialNormalizer, LearnableLogNorm
from normalizations.asinh import AsinhNormalizer


def test_lognormalizer():
    #norm = LogNormalizer(epsilon=1e-9, input_scale=1e+5)
    norm = LogNormalizer(epsilon=1e-13, range=(0.0, 1.0), input_scale=1e+3)
    x = torch.tensor([0.0, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-5, 1e-2, 0.1, 0.3, 0.5, 0.75, 1.0], dtype=torch.float32)
    norm.validate_range(x)
    y = norm(x)
    assert y.shape == x.shape
    assert torch.all(y >= 0), f"All values should be >= 0, got {y}"
    assert torch.all(y <= 1), f"All values should be <= 1, got {y}"
    assert torch.all(torch.isfinite(y)), f"All values should be finite, got {y}"
    assert torch.max(y) == 1.0, f"Max value should be 1.0, got {torch.max(y)}"
    assert torch.min(y) == 0.0, f"Min value should be 0.0, got {torch.min(y)}"
    x_inv = norm.inverse(y)
    assert torch.all(torch.isfinite(x_inv)), f"All inverse values should be finite, got {x_inv}"
    assert torch.isclose(x_inv, x, atol=1e-13).all(), f"Inverse normalization failed, got {x_inv}, expected {x} (correct elements are {torch.isclose(x_inv, x, atol=1e-13).sum()} of {x.numel()})"
    print("LogNormalizer test passed.")

test_lognormalizer()

def test_linearnormalizer():
    norm = LinearNormalizer()
    x = torch.tensor([0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-3, 1e-2, 1e-1, 1.0, 2.0], dtype=torch.float32)
    norm.validate_range(x)
    y = norm(x)
    assert y.shape == x.shape
    assert torch.all(y >= 0), f"All values should be >= 0, got {y}"
    assert torch.all(y <= 1), f"All values should be <= 1, got {y}"
    assert torch.all(torch.isfinite(y)), f"All values should be finite, got {y}"
    assert torch.max(y) == 1.0, f"Max value should be 1.0, got {torch.max(y)}"
    assert torch.min(y) == 0.0, f"Min value should be 0.0, got {torch.min(y)}"
    x_inv = norm.inverse(y, respect_to=x)
    assert torch.all(torch.isfinite(x_inv)), f"All inverse values should be finite, got {x_inv}"
    assert torch.isclose(x_inv, x, atol=1e-9).all(), f"Inverse normalization failed, got {x_inv}, expected {x} (correct elements are {torch.isclose(x_inv, x, atol=1e-9).sum()} of {x.numel()})"
    print("LinearNormalizer test passed.")


def test_linearnormalizer_neg_pos():
    norm = LinearNormalizer((-1, 1))
    x = torch.tensor([0.0, 1e-6, 1e-5, 1e-3, 1e-2, 1e-1, 1.0, 2.0], dtype=torch.float32)
    norm.validate_range(x)
    y = norm(x)
    assert y.shape == x.shape
    assert torch.all(y >= -1), f"All values should be >= -1, got {y}"
    assert torch.all(y <= 1), f"All values should be <= 1, got {y}"
    assert torch.all(torch.isfinite(y)), f"All values should be finite, got {y}"
    assert torch.max(y) == 1.0, f"Max value should be 1.0, got {torch.max(y)}"
    assert torch.min(y) == -1.0, f"Min value should be -1.0, got {torch.min(y)}"
    x_inv = norm.inverse(y, respect_to=x)
    assert torch.all(torch.isfinite(x_inv)), f"All inverse values should be finite, got {x_inv}"
    assert torch.isclose(x_inv, x, atol=1e-7).all(), f"Inverse normalization failed, got {x_inv}, expected {x} (correct elements are {torch.isclose(x_inv, x, atol=1e-7).sum()} of {x.numel()})"

    x = torch.tensor([0.0, 1e-7, 1e-6, 1e-5, 1e-3, 1e-2, 1e-1, 1.0, 2.0], dtype=torch.float32)
    try:
        norm.validate_range(x)
        assert False, "Expected ValueError for input with too small values."
    except ValueError as e:
        assert "Input to LinearNormalizer " in str(e), f"Unexpected error message: {e}"

    print("LinearNormalizer (-1, 1) test passed.")


def test_asinhnormalizer():
    norm = AsinhNormalizer(input_scale=1e-3, range=(0.0, 1.0))
    x = torch.tensor([0.0, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-3, 1e-2, 1e-1, 1.0, 2.0], dtype=torch.float32)
    norm.validate_range(x)
    y = norm(x)
    assert y.shape == x.shape
    assert torch.all(y >= 0), f"All values should be >= 0, got {y}"
    assert torch.all(y <= 1), f"All values should be <= 1, got {y}"
    assert torch.all(torch.isfinite(y)), f"All values should be finite, got {y}"
    assert torch.isclose(torch.max(y), torch.tensor(1.0), atol=1e-13), f"Max value should be 1.0, got {torch.max(y)}"
    assert torch.isclose(torch.min(y), torch.tensor(0.0), atol=1e-13), f"Min value should be 0.0, got {torch.min(y)}"
    x_inv = norm.inverse(y, respect_to=x)
    assert torch.all(torch.isfinite(x_inv)), f"All inverse values should be finite, got {x_inv}"
    assert torch.isclose(x_inv, x, atol=1e-13).all(), f"Inverse normalization failed, got {x_inv}, expected {x} (correct elements are {torch.isclose(x_inv, x, atol=1e-13).sum()} of {x.numel()})"
    print("AsinhNormalizer test passed.")


def test_specialpolynomialnormalizer():
    norm = SpecialPolynomialNormalizer()
    x = torch.tensor([0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-3, 1e-2, 1e-1, 1.0], dtype=torch.float32)
    x = x.view(1, 1, 3, 3, 1)  # Shape: (1, 1, 3, 3, 1)
    x = x.expand(2, 1, 3, 3, 3)  # Shape: (2, 1, 3, 3, 3)
    norm.validate_range(x)
    y = norm(x)
    assert y.shape == x.shape
    assert torch.all(y >= -1), f"All values should be >= -1, got {y}"
    assert torch.all(y <= 1), f"All values should be <= 1, got {y}"
    assert torch.all(torch.isfinite(y)), f"All values should be finite, got {y}"
    assert torch.max(y).isclose(torch.tensor(1.0)), f"Max value should be 1.0, got {torch.max(y)}"
    assert torch.min(y).isclose(torch.tensor(-1.0)), f"Min value should be -1.0, got {torch.min(y)}"

    x_inv = norm.inverse(y)
    assert torch.all(torch.isfinite(x_inv)), f"All inverse values should be finite, got {x_inv}"
    assert torch.isclose(x_inv, x, atol=1e-6).all(), f"Inverse normalization failed, got {x_inv}, expected {x} (correct elements are {torch.isclose(x_inv, x, atol=1e-6).sum()} of {x.numel()})"
    print("SpecialPolynomialNormalizer test passed.")


def test_learnablelognorm():
    norm = LearnableLogNorm()
    x = torch.tensor([0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-3, 1e-2, 1e-1, 1.0], dtype=torch.float32)
    x = x.view(1, 1, 3, 3, 1)  # Shape: (1, 1, 3, 3, 1)
    x = x.expand(2, 1, 3, 3, 3)  # Shape: (2, 1, 3, 3, 3)
    norm.validate_range(x)
    y = norm(x)
    assert y.shape == x.shape
    assert torch.all(y >= 0), f"All values should be >= 0, got {y}"
    assert torch.all(y <= 1), f"All values should be <= 1, got {y}"
    assert torch.all(torch.isfinite(y)), f"All values should be finite, got {y}"
    assert torch.max(y).isclose(torch.tensor(1.0)), f"Max value should be 1.0, got {torch.max(y)}"
    assert torch.min(y).isclose(torch.tensor(0.0)), f"Min value should be 0.0, got {torch.min(y)}"

    x_inv = norm.inverse(y, respect_to=x)
    assert torch.all(torch.isfinite(x_inv)), f"All inverse values should be finite, got {x_inv}"
    assert torch.isclose(x_inv, x, atol=1e-6).all(), f"Inverse normalization failed, got {x_inv}, expected {x} (correct elements are {torch.isclose(x_inv, x, atol=1e-6).sum()} of {x.numel()})"
    print("LearnableLogNorm test passed.")


if __name__ == "__main__":
    from plotly import graph_objects as go
    test_linearnormalizer()
    test_linearnormalizer_neg_pos()
    test_asinhnormalizer()
    
    fig = go.Figure()
    #for norm, name in [(LogNormalizer((-1, 1)), "LogNormalizer"), (LinearNormalizer((-1, 1)), "LinearNormalizer"), (SpecialPolynomialNormalizer(), "SpecialPolynomialNormalizer"), (LearnableLogNorm(), "LearnableLogNorm"), (AsinhNormalizer((-1, 1), input_scale=1.0), "AsinhNormalizer")]:
    for norm, name in [(AsinhNormalizer((0, 1), input_scale=1.0), "AsinhNormalizer alpha=1.0"),
                       (AsinhNormalizer((0, 1), input_scale=1e-1), "AsinhNormalizer alpha=1e-1"),
                       (AsinhNormalizer((0, 1), input_scale=1e-5), "AsinhNormalizer alpha=1e-5"),
                       (AsinhNormalizer((0, 1), input_scale=1e-4), "AsinhNormalizer alpha=1e-4"),
                       (AsinhNormalizer((0, 1), input_scale=1e-3), "AsinhNormalizer alpha=1e-3"),
                       (AsinhNormalizer((0, 1), input_scale=1e-2), "AsinhNormalizer alpha=1e-2"),
                       (LearnableLogNorm(), "LearnableLogNorm"),
                       (LogNormalizer(range=(0, 1), input_scale=1.0), "LogNormalizer alpha=1.0"),
                       (LogNormalizer(range=(0, 1), input_scale=1e-1), "LogNormalizer alpha=1e-1"),
                       (LogNormalizer(range=(0, 1), input_scale=1e-3), "LogNormalizer alpha=1e-3"),
                       (LogNormalizer(range=(0, 1), input_scale=1e+1), "LogNormalizer alpha=1e+1"),
                       (LogNormalizer(range=(0, 1), input_scale=1e+3), "LogNormalizer alpha=1e+3"),
                       (LogNormalizer(range=(0, 1), input_scale=1e+5), "LogNormalizer alpha=1e+5"),
                       ]:
        x = torch.linspace(0, 1, steps=100000, dtype=torch.float32)
        #x = x.view(1, 1, 10, 10, 10)  # Shape: (1, 1, 10, 10, 10)
        y = norm(x)
        x = x.flatten()
        y = y.flatten()
        fig.add_trace(go.Scatter(x=x.numpy(), y=y.detach().numpy(), mode='lines', name=name))
        print(f"{name}: executed successfully.")
    fig.show()

    while True:
        pass
