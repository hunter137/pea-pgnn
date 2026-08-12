from pathlib import Path

import numpy as np

from pea_pgnn import ModelConfig, PriorAnchoredRegressor, TrainingConfig


def test_estimator_fit_predict_and_checkpoint(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    context = rng.normal(size=(40, 2))
    time = rng.uniform(1.0, 300.0, size=40)
    magnitude = 500.0 + 25.0 * context[:, 0]
    timescale = 60.0 + 5.0 * context[:, 1]
    target = magnitude * np.tanh(np.sqrt(time / timescale))

    estimator = PriorAnchoredRegressor(
        ModelConfig(hidden_dims=(8,), dropout=0.0),
        TrainingConfig(
            epochs=8,
            batch_size=16,
            patience=4,
            learning_rate=1e-3,
            seed=5,
            device="cpu",
        ),
    )
    estimator.fit(context, time, target, magnitude * 0.95, timescale * 1.05)
    prediction = estimator.predict(context, time, magnitude * 0.95, timescale * 1.05)
    assert prediction.shape == (40,)
    assert np.all(np.isfinite(prediction))
    assert np.all(prediction >= 0.0)

    checkpoint = tmp_path / "model.pt"
    estimator.save(checkpoint)
    restored = PriorAnchoredRegressor.load(checkpoint, device="cpu")
    restored_prediction = restored.predict(
        context, time, magnitude * 0.95, timescale * 1.05
    )
    np.testing.assert_allclose(prediction, restored_prediction, rtol=1e-6, atol=1e-6)

