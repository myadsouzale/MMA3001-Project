
"""
Unit tests for occupancy prediction functions.

Tests feature encoding, accuracy calculations,
and historical occupancy prediction behaviour
using small synthetic datasets.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Allow imports from the project's src folder.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sensitivity_analysis import (
    encode,
    calculate_metrics,
    fit_baselines,
    predict_space_mean,
    predict_weekly,
)


def test_perfect_predictions():
    """Perfect predictions should have zero error."""

    actual = np.array([0, 2, 4, 6])
    predicted = np.array([0, 2, 4, 6])

    mae, rmse = calculate_metrics(actual, predicted)

    assert mae == pytest.approx(0)
    assert rmse == pytest.approx(0)


def test_known_prediction_errors():
    """Check MAE and RMSE against hand calculations."""

    actual = np.array([2, 4])
    predicted = np.array([3, 6])

    mae, rmse = calculate_metrics(actual, predicted)

    assert mae == pytest.approx(1.5)
    assert rmse == pytest.approx(np.sqrt(2.5))


def test_negative_predictions_are_clipped():
    """Predicted occupancy cannot be negative."""

    actual = np.array([0, 2])
    predicted = np.array([-5, 2])

    mae, rmse = calculate_metrics(actual, predicted)

    assert mae == pytest.approx(0)
    assert rmse == pytest.approx(0)


def test_feature_encoding():
    """Check one-hot encoding of building spaces."""

    data = pd.DataFrame({
        "space_number": [0, 1],
        "day_number": [0, 1],
        "hour_of_day": [9, 10],
        "month": [5, 5],
    })

    encoded = encode(data)

    assert "space_number_0" in encoded.columns
    assert "space_number_1" in encoded.columns

    assert encoded.loc[0, "space_number_0"] == 1
    assert encoded.loc[1, "space_number_1"] == 1


def test_feature_alignment():
    """Validation features must match training columns."""

    training = pd.DataFrame({
        "space_number": [0, 1],
        "day_number": [0, 1],
        "hour_of_day": [9, 10],
        "month": [5, 5],
    })

    validation = pd.DataFrame({
        "space_number": [1],
        "day_number": [2],
        "hour_of_day": [11],
        "month": [5],
    })

    train_features = encode(training)

    val_features = encode(
        validation,
        train_features.columns
    )

    assert list(val_features.columns) == list(
        train_features.columns
    )

    assert val_features.iloc[0]["space_number_0"] == 0
    assert val_features.iloc[0]["space_number_1"] == 1


@pytest.fixture
def training_data():
    """Create a small known occupancy dataset."""

    return pd.DataFrame({
        "floorspaceid": ["A", "A", "A", "B"],
        "day_number": [0, 0, 1, 0],
        "hour_of_day": [9, 9, 10, 9],
        "headcount_time_weighted": [2, 4, 6, 8],
    })


def test_space_mean_prediction(training_data):
    """Check the historical mean for each space."""

    fitted = fit_baselines(training_data)

    observations = pd.DataFrame({
        "floorspaceid": ["A", "B"],
    })

    predictions = predict_space_mean(
        observations,
        fitted
    )

    assert predictions[0] == pytest.approx(4)
    assert predictions[1] == pytest.approx(8)


def test_weekly_pattern_prediction(training_data):
    """Check prediction for a known weekday and hour."""

    fitted = fit_baselines(training_data)

    observations = pd.DataFrame({
        "floorspaceid": ["A"],
        "day_number": [0],
        "hour_of_day": [9],
    })

    predictions = predict_weekly(
        observations,
        fitted
    )

    # Mean of training headcounts 2 and 4.
    assert predictions[0] == pytest.approx(3)


def test_weekly_fallback_to_space_hour(training_data):
    """Use space-hour average if weekday is missing."""

    fitted = fit_baselines(training_data)

    observations = pd.DataFrame({
        "floorspaceid": ["A"],
        "day_number": [5],
        "hour_of_day": [9],
    })

    predictions = predict_weekly(
        observations,
        fitted
    )

    assert predictions[0] == pytest.approx(3)


def test_weekly_fallback_to_space_mean(training_data):
    """Use space mean if weekday and hour are missing."""

    fitted = fit_baselines(training_data)

    observations = pd.DataFrame({
        "floorspaceid": ["A"],
        "day_number": [5],
        "hour_of_day": [20],
    })

    predictions = predict_weekly(
        observations,
        fitted
    )

    assert predictions[0] == pytest.approx(4)


def test_unknown_space_uses_global_mean(training_data):
    """Unknown spaces should use the global mean."""

    fitted = fit_baselines(training_data)

    observations = pd.DataFrame({
        "floorspaceid": ["Unknown"],
        "day_number": [5],
        "hour_of_day": [20],
    })

    predictions = predict_weekly(
        observations,
        fitted
    )

    # Global mean = (2 + 4 + 6 + 8) / 4.
    assert predictions[0] == pytest.approx(5)

