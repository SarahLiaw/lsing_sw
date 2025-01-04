#%%
import os
import torch
import numpy as np

from src.trainers.map_trainer import MapTrainer
from src.evaluators.precision_matrix_computer import PrecisionMatrixComputer
from models.UMNN import MonotonicNN

import matplotlib.pyplot as plt
import yaml


# Load experiment configuration
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    # Load configuration
    config = load_config("config.yaml")
    os.makedirs("results", exist_ok=True)

    # Generate synthetic data for the Butterfly distribution
    training_data = torch.rand((config["train_size"], config["num_features"]))
    validation_data = torch.rand((config["validation_size"], config["num_features"]))
    test_data = torch.rand((config["test_size"], config["num_features"]))

    # Initialize fixed map for each feature
    base_map = [
        MonotonicNN(config["num_features"] - 1, config["hidden_layers"], config["num_steps"], "cpu")
        for _ in range(config["num_features"])
    ]

    # Train models
    trainer = MapTrainer(
        base_map,
        config["num_features"],
        config["hidden_layers"],
        config["num_steps"],
        config["learning_rate"],
        config["max_epochs"],
    )

    trained_models = []
    for target_index in range(config["num_features"]):
        other_indices = [i for i in range(config["num_features"]) if i != target_index]
        best_model, _, _ = trainer.train_model(
            training_data, target_index, other_indices, config["regularizations"], validation_data
        )
        trained_models.append(best_model)

    # Compute precision matrix
    evaluator = PrecisionMatrixComputer(trained_models, test_data, config["num_features"])
    precision_matrix = evaluator.compute()

    # Save precision matrix
    matrix_path = os.path.join("results", "butterfly_matrix.npy")
    np.save(matrix_path, precision_matrix)

    # Visualize precision matrix
    plot_path = os.path.join("results", "butterfly_plots.png")
    evaluator.visualize(precision_matrix, title="Butterfly Experiment Precision Matrix")
    plt.savefig(plot_path)

    # Log results
    with open(os.path.join("results", "log.txt"), "w") as log_file:
        log_file.write(f"Experiment completed.\nPrecision matrix saved to {matrix_path}.\n")

if __name__ == "__main__":
    main()
