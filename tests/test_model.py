import pytest
import torch

from pea_pgnn import ModelConfig, PriorAnchoredTemporalModel, audit_trajectory


def test_model_details_respect_admissible_domains() -> None:
    torch.manual_seed(3)
    config = ModelConfig(hidden_dims=(16, 8), dropout=0.0)
    model = PriorAnchoredTemporalModel(context_dim=3, config=config)
    context = torch.randn(12, 3)
    time = torch.linspace(0.0, 500.0, 12)
    prediction, details = model(
        context,
        time,
        torch.full((12,), 600.0),
        torch.full((12,), 80.0),
        return_details=True,
    )
    assert prediction.shape == (12,)
    assert torch.all(prediction >= 0.0)
    assert torch.allclose(details["weights"].sum(dim=1), torch.ones(12))
    assert torch.all(details["magnitude"] >= config.magnitude_bounds[0])
    assert torch.all(details["magnitude"] <= config.magnitude_bounds[1])
    assert torch.all(details["timescale"] >= config.timescale_bounds[0])
    assert torch.all(details["timescale"] <= config.timescale_bounds[1])


def test_initial_corrections_are_anchor_centered() -> None:
    model = PriorAnchoredTemporalModel(2, ModelConfig(hidden_dims=(8,), dropout=0.0))
    _, details = model(
        torch.zeros(4, 2),
        torch.ones(4),
        torch.full((4,), 500.0),
        torch.full((4,), 50.0),
        return_details=True,
    )
    assert torch.max(torch.abs(details["magnitude_relative_correction"])).item() < 1e-6
    assert torch.max(torch.abs(details["timescale_relative_correction"])).item() < 1e-6
    assert torch.max(torch.abs(details["magnitude_additive_correction"])).item() < 1e-6


def test_fixed_context_trajectory_is_monotonic() -> None:
    model = PriorAnchoredTemporalModel(2, ModelConfig(hidden_dims=(8,), dropout=0.0))
    time = torch.linspace(0.0, 2000.0, 300)
    context = torch.tensor([[0.25, -0.5]]).repeat(len(time), 1)
    with torch.no_grad():
        prediction, details = model(
            context, time, torch.tensor(650.0), torch.tensor(75.0), return_details=True
        )
    report = audit_trajectory(
        prediction.numpy(), magnitude=details["magnitude"].numpy()
    )
    assert report.passed


def test_negative_time_is_rejected() -> None:
    model = PriorAnchoredTemporalModel(2, ModelConfig(hidden_dims=(8,)))
    with pytest.raises(ValueError, match="non-negative"):
        model(torch.zeros(1, 2), torch.tensor([-1.0]), torch.tensor([500.0]), torch.tensor([50.0]))
