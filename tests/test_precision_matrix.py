import pytest
import torch
import numpy as np
from evaluators.precision_matrix_computer import PrecisionMatrixComputer
from models.UMNN import MonotonicNN

@pytest.fixture
def mock_data():
    test_data = torch.rand((50, 10))
    trained_models = [MonotonicNN(10, [16, 32], 50, 'cpu') for _ in range(10)]
    return test_data, trained_models

def test_compute_precision_matrix(mock_data):
    test_data, trained_models = mock_data
    evaluator = PrecisionMatrixComputer(trained_models, test_data, 10)

    precision_matrix = evaluator.compute()

    assert precision_matrix.shape == (10, 10), "Precision matrix shape mismatch"
    assert np.allclose(np.diag(precision_matrix), 1), "Diagonal elements should be 1"

def test_visualize_precision_matrix(mock_data):
    _, trained_models = mock_data
    evaluator = PrecisionMatrixComputer(trained_models, torch.rand((50, 10)), 10)

    # Mock precision matrix
    mock_matrix = np.random.rand(10, 10)
    try:
        evaluator.visualize(mock_matrix, title="Test Matrix")
    except Exception as e:
        pytest.fail(f"Visualization failed: {e}")
