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
    Main function to execute the experiment.

    - Loads configuration and datasets.
    - Trains models using MonotonicNN for each feature.
    - Computes and visualizes the precision matrix.
    - Optionally saves trained models.
    - Saves results (matrix and plots) to a uniquely indexed directory.
    """
    
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    config = load_config(config_path)

    base_results_path = config["results_path"]
    index = 0
    while True:
        results_dir = os.path.join(base_results_path, f"{config['experiment_name']}_{index}")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            break
        index += 1

    print(f"Results will be saved in {results_dir}")

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

    # Save trained models, if configured
    if config["save_models"]:
        models_path = os.path.join(results_dir, "trained_models.pth")
        torch.save(trained_models, models_path)
        print(f"All trained models saved to {models_path}")

    # Compute precision matrix
    evaluator = PrecisionMatrixComputer(trained_models, test_data, config["num_features"])
    precision_matrix = evaluator.compute()
    symmetric_precision_matrix = evaluator.make_symmetric(precision_matrix)

    matrix_path = os.path.join(results_dir, "precision_matrix.txt")
    np.savetxt(matrix_path, symmetric_precision_matrix, fmt="%.6f", delimiter=",")
    print(f"Precision matrix saved to {matrix_path}")

    # Visualize precision matrix
    plot_path = os.path.join(results_dir, "precision_matrix_normalized.png")
    evaluator.visualize(symmetric_precision_matrix, title="Experiment Precision Matrix")
    plt.savefig(plot_path)

    # Log results and config details
    log_path = os.path.join(results_dir, "log.txt")
    with open(log_path, "w") as log_file:
        log_file.write("Experiment Log\n")
        log_file.write("=" * 50 + "\n")
        log_file.write(f"Results Directory: {results_dir}\n")
        log_file.write(f"Precision Matrix File: {matrix_path}\n")
        log_file.write(f"Normalized Precision Matrix Plot: {plot_path}\n")
        if config["save_models"]:
            log_file.write(f"Trained Models File: {models_path}\n")
        log_file.write("\nConfiguration Details:\n")
        log_file.write("=" * 50 + "\n")
        for key, value in config.items():
            log_file.write(f"{key}: {value}\n")
        log_file.write("=" * 50 + "\n")
    print(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()
