import torch
import numpy as np
from models.UMNN import MonotonicNN
from map_trainer import MapTrainer
from precision_matrix_computer import PrecisionMatrixComputer

# Configuration
NUM_FEATURES = 10
HIDDEN_LAYERS = [16, 32, 8]
NUM_STEPS = 50
LEARNING_RATE = 0.01
MAX_EPOCHS = 100
REGULARIZATIONS = [1, 0.1, 0.01, 0.001, 0]
TRAIN_SIZE = 5000
VALIDATION_SIZE = 1000
TEST_SIZE = 1000

# Generate synthetic data
def generate_data(num_samples, num_features):
    return torch.rand((num_samples, num_features))

training_data = generate_data(TRAIN_SIZE, NUM_FEATURES)
validation_data = generate_data(VALIDATION_SIZE, NUM_FEATURES)
test_data = generate_data(TEST_SIZE, NUM_FEATURES)

# Initialize fixed map
base_map = [
    MonotonicNN(input_dim=NUM_FEATURES - 1, hidden_layers=HIDDEN_LAYERS, nb_steps=NUM_STEPS, device="cpu")
    for _ in range(NUM_FEATURES)
]

# Train the models
trainer = MapTrainer(
    base_map=base_map,
    num_features=NUM_FEATURES,
    hidden_layers=HIDDEN_LAYERS,
    num_steps=NUM_STEPS,
    learning_rate=LEARNING_RATE,
    max_epochs=MAX_EPOCHS
)

trained_models = []
optimal_regularizations = []

for target_index in range(NUM_FEATURES):
    other_indices = [i for i in range(NUM_FEATURES) if i != target_index]
    best_model, best_reg, _ = trainer.train_model(
        training_data, target_index, other_indices, REGULARIZATIONS, validation_data
    )
    trained_models.append(best_model)
    optimal_regularizations.append(best_reg)

# Compute precision matrix
precision_computer = PrecisionMatrixComputer(
    trained_models=trained_models,
    test_data=test_data,
    num_features=NUM_FEATURES
)

precision_matrix = precision_computer.compute()

# Visualize precision matrix need to make it symmetric and do etc. 
precision_computer.visualize(precision_matrix, title="Precision Matrix for UMNN Models")

# Save precision matrix for future use
np.save("precision_matrix.npy", precision_matrix)

print("Experiment complete. Precision matrix saved as 'precision_matrix.npy'.")
