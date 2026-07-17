from __future__ import annotations

import numpy as np

from legendre_mia.attacks.core import (
    conservative_tpr_at_fp,
    legendre_coefficients,
    legendre_features,
    pseudo_validation_tpr_at_fp,
    psi_table,
    standardize_image_triplet,
)


def test_integrated_shifted_legendre_low_orders() -> None:
    points = np.asarray([0.0, 0.25, 0.5, 1.0], dtype=np.float64)
    psi = psi_table(points, 2)
    np.testing.assert_allclose(psi[0], points, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(psi[1], points * points - points, atol=1e-15)
    expected_order_2 = 2.0 * points**3 - 3.0 * points**2 + points
    np.testing.assert_allclose(psi[2], expected_order_2, atol=1e-15)


def test_k4_feature_dimension_and_zero_residual() -> None:
    p_target = np.asarray(
        [0.1, 0.2],
        dtype=np.float32,
    )

    # Each row contains responses from two identical reference models.
    p_reference = np.asarray(
        [
            [0.1, 0.1],
            [0.2, 0.2],
        ],
        dtype=np.float32,
    )

    features = legendre_features(
        p_target[:, None],
        p_reference,
        4,
    )

    assert features.shape == (2, 7)

    # The Tail is Gamma_2, Gamma_3, Gamma_4.
    np.testing.assert_allclose(
        features[:, 4:],
        np.zeros((2, 3), dtype=np.float32),
        atol=0.0,
        rtol=0.0,
    )


def test_views_are_averaged_before_legendre_projection() -> None:
    views = np.asarray(
        [[0.1, 0.9]],
        dtype=np.float64,
    )

    stabilized = views.mean(axis=1)

    correct = legendre_coefficients(
        stabilized[:, None],
        4,
    )

    # This is the old, incorrect behavior:
    # averaging Psi_k over individual views.
    incorrect = legendre_coefficients(
        views,
        4,
    )

    # Q0 is linear, so higher-order coordinates must differ.
    np.testing.assert_allclose(correct[:, 0], incorrect[:, 0])
    assert not np.allclose(correct[:, 1:], incorrect[:, 1:])


def test_pseudo_validation_selector_differs_from_conservative_rule() -> None:
    labels = np.asarray([0, 1, 0], dtype=bool)
    scores = np.asarray([3.0, 2.0, 1.0])
    assert pseudo_validation_tpr_at_fp(labels, scores, 1) == 1.0
    assert conservative_tpr_at_fp(labels, scores, 1) == 0.0


def test_image_standardization_uses_unrounded_float64_statistics() -> None:
    train = np.asarray([[0.1, 1.0], [0.2, 4.0], [0.4, 9.0]], dtype=np.float32)
    validation = train[:1]
    evaluation = train[1:]
    x_train, x_validation, x_evaluation, mean, std = standardize_image_triplet(
        train, validation, evaluation
    )
    mean64 = train.astype(np.float64).mean(axis=0, keepdims=True)
    std64 = train.astype(np.float64).std(axis=0, keepdims=True)
    expected = ((train.astype(np.float64) - mean64) / std64).astype(np.float32)
    np.testing.assert_array_equal(x_train, expected)
    assert x_validation.dtype == np.float32
    assert x_evaluation.dtype == np.float32
    assert mean.dtype == np.float32
    assert std.dtype == np.float32
