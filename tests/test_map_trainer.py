import pytest
import torch
from trainers.map_trainer import MapTrainer
from models.UMNN import MonotonicNN

@pytest.fixture
def mock_data():
    training_data = torch.rand((100, 10))
    validation_data = torch.rand((50, 10))
    base_map = [MonotonicNN(10, [16, 32], 50, 'cpu') for _ in range(10)]
    return training_data, validation_data, base_map

def test_train_model(mock_data):
    training_data, validation_data, base_map = mock_data
    trainer = MapTrainer(base_map, 10, [16, 32], 50, 0.01, 5)

    # Test model training for feature 0
    model, reg, _ = trainer.train_model(
        training_data, 0, list(range(1, 10)), [0.1, 1.0], validation_data
    )

    assert model is not None, "Trained model should not be None"
    assert isinstance(reg, float), "Regularization value should be a float"

def test_evaluate_map(mock_data):
    training_data, _, base_map = mock_data
    trainer = MapTrainer(base_map, 10, [16, 32], 50, 0.01, 5)

    # Mock model evaluation
    non_target_indices = list(range(1, 10))
    output, jacobian = trainer.evaluate_map(training_data, non_target_indices, 0, base_map[0])

    assert output.shape == (100, 1), "Output shape mismatch"
    assert jacobian.shape == (100, 1), "Jacobian shape mismatch"
