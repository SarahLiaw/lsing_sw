import os
import numpy as np
import torch
import argparse
from sklearn.datasets import make_sparse_spd_matrix
import matplotlib.pyplot as plt


def generate_gaussian(num_samples, num_features, random_state=None):
    """
    Generates a Gaussian dataset with a specified precision matrix.

    Args:
        num_samples (int): Number of samples to generate.
        num_features (int): Number of features in the dataset.
        random_state (int or np.random.RandomState): Random state for reproducibility.

    Returns:
        tuple: Generated dataset (torch.Tensor), precision matrix (np.ndarray), and covariance matrix (np.ndarray).
    """
    if random_state is None:
        random_state = np.random.RandomState(1)

    precision_matrix = make_sparse_spd_matrix(
        num_features, alpha=0.95, smallest_coef=0.3, largest_coef=0.8, random_state=random_state
    )
    precision_matrix = np.abs(precision_matrix)
    covariance_matrix = np.linalg.inv(precision_matrix)

    X = random_state.multivariate_normal(np.zeros(num_features), covariance_matrix, size=num_samples)
    X -= X.mean(axis=0)
    X /= X.std(axis=0)

    return torch.tensor(X, dtype=torch.float32), precision_matrix, covariance_matrix


def save_dataset(dataset, output_dir, dataset_type, num_samples, num_features):
    """
    Saves the generated dataset to a text file.

    Args:
        dataset (torch.Tensor): Generated dataset.
        output_dir (str): Directory to save the dataset.
        dataset_type (str): Type of dataset (e.g., 'training', 'validation', 'testing').
        num_samples (int): Number of samples in the dataset.
        num_features (int): Number of features in the dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"gauss/g_{dataset_type}_{num_samples}_{num_features}d.txt")
    np.savetxt(output_path, dataset.numpy(), fmt="%.8f", delimiter=",")
    print(f"{dataset_type.capitalize()} dataset saved to {output_path}")


def visualize_precision_matrix(precision_matrix, output_dir):
    """
    Visualizes and saves the precision matrix as a heatmap.

    Args:
        precision_matrix (np.ndarray): Precision matrix to visualize.
        output_dir (str): Directory to save the plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.xticks(np.arange(0, len(precision_matrix), 1), np.arange(1, len(precision_matrix) + 1))
    plt.yticks(np.arange(0, len(precision_matrix), 1), np.arange(1, len(precision_matrix) + 1))
    plt.imshow(precision_matrix, cmap='gray', interpolation='nearest')

    for i in range(len(precision_matrix)):
        for j in range(len(precision_matrix)):
            plt.text(j, i, f'{precision_matrix[i, j]:.2f}', ha='center', va='center', color='red')

    plt.colorbar()
    plt.show()

def main():
    
    parser = argparse.ArgumentParser(description="Generate Gaussian datasets.")
    parser.add_argument("--dataset_type", type=str, required=True,
                        choices=["trn", "val", "tst"],
                        help="Type of dataset to generate.")
    parser.add_argument("--num_samples", type=int, required=True,
                        help="Number of samples to generate.")
    parser.add_argument("--num_features", type=int, required=True,
                        help="Number of features in the dataset.")
    parser.add_argument("--output_dir", type=str, default="../../data",
                        help="Directory to save the dataset.")
    parser.add_argument("--visualize", action="store_true",
                        help="Whether to visualize the precision matrix.")
    args = parser.parse_args()

    dataset, precision_matrix, _ = generate_gaussian(args.num_samples, args.num_features)
    save_dataset(dataset, args.output_dir, args.dataset_type, args.num_samples, args.num_features)

    if args.visualize:
        visualize_precision_matrix(precision_matrix, args.output_dir)


if __name__ == "__main__":
    main()
