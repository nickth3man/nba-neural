"""Unit tests for the DANN model components."""

import torch

from era_translator.model import EraTranslatorDANN, GradientReversalLayer


def test_grl_forward_identity_backward_reversal():
    layer = GradientReversalLayer()
    layer.lambda_ = 2.0
    x = torch.randn(4, 8, requires_grad=True)
    out = layer(x)
    assert torch.allclose(out, x)  # forward identity

    loss = (out ** 2).sum()
    loss.backward()
    # d/dx (x^2) = 2x; reversed by -lambda -> -4x
    assert torch.allclose(x.grad, -4.0 * x.detach(), atol=1e-5)


def test_dann_forward_shapes():
    model = EraTranslatorDANN(input_dim=13)
    x = torch.randn(8, 13)
    z, value, era = model(x, lambda_=1.0)
    assert z.shape == (8, model.latent_dim)
    assert value.shape == (8,)
    assert era.shape == (8, model.num_eras)


def test_dann_trains_one_step():
    torch.manual_seed(0)
    model = EraTranslatorDANN(input_dim=13)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    mse = torch.nn.MSELoss()
    ce = torch.nn.CrossEntropyLoss()
    x = torch.randn(16, 13)
    y = torch.randn(16)
    e = torch.randint(0, 9, (16,))
    z, v, era = model(x, lambda_=0.5)
    loss = mse(v, y) + 0.5 * ce(era, e)
    loss.backward()
    opt.step()
    assert loss.item() > 0
