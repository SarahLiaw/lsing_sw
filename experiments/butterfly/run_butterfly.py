import os
import torch
import numpy as np

from src.trainers.map_trainer import MapTrainer
from src.evaluators.precision_matrix_computer import PrecisionMatrixComputer
from models.UMNN import MonotonicNN
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import yaml


def load_data(file_path):
    """
    Load data from a file.

    Args:
        file_path (str): Path to the dataset file in CSV format.

    Returns:
        torch.Tensor: Loaded dataset as a PyTorch tensor.
    """
    print(f"Loading dataset from {file_path}")
    data = np.loadtxt(file_path, delimiter=",")
    return torch.tensor(data, dtype=torch.float32)


def load_config(config_path):
    """
    Load experiment configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Configuration parameters as a dictionary.
    """
    print(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def main():
    """
    Main function to execute the Butterfly experiment.

    - Loads configuration and datasets.
    - Trains models using MonotonicNN for each feature.
    - Computes and visualizes the precision matrix.
    - Saves results (matrix and plots) to the output directory.
    """
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    config = load_config(config_path)
    os.makedirs("results", exist_ok=True)

    # Load datasets
    training_data = load_data(config["training_file"])
    validation_data = load_data(config["validation_file"])
    test_data = load_data(config["testing_file"])

    # Initialize fixed map for each feature
    base_map = [
        MonotonicNN(config["num_features"], config["hidden_layers"], config["num_steps"], "cpu")
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
    plot_path = os.path.join("results", "butterfly_plots.png")
    evaluator.visualize(precision_matrix, title="Butterfly Experiment Precision Matrix")
    plt.savefig(plot_path)

    transpose_matrix = precision_matrix.T 
    symmetric_matrix = (transpose_matrix + precision_matrix) / 2

    symmetric_precision_matrix = symmetric_matrix / np.max(symmetric_matrix)
    np.fill_diagonal(symmetric_precision_matrix, 1)

    # Save precision matrix as a .txt file
    matrix_path = os.path.join("results", "butterfly_matrix.txt")
    np.savetxt(matrix_path, symmetric_precision_matrix, fmt="%.6f", delimiter=",")
    print(f"Precision matrix saved to {matrix_path}")

    # Visualize precision matrix
    plot_path = os.path.join("results", "butterfly_plots_normalized.png")
    evaluator.visualize(symmetric_precision_matrix, title="Butterfly Experiment Precision Matrix")
    plt.savefig(plot_path)

    # Log results
    with open(os.path.join("results", "log.txt"), "w") as log_file:
        log_file.write(f"Experiment completed.\nPrecision matrix saved to {matrix_path}.\n")


if __name__ == "__main__":
    main()
