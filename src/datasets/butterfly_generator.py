import os
import numpy as np
import torch
import argparse


def generate_butterfly(num_samples, num_pairs):
    """
    Generates Butterfly data with paired samples.

    Args:
        num_samples (int): Number of samples to generate.
        num_pairs (int): Number of (X, Y) pairs in each sample.

    Returns:
        torch.Tensor: Generated Butterfly dataset.
    """
    samples = []
    for _ in range(num_samples):
        sample = []
        for _ in range(num_pairs):
            X = np.random.normal(0, 1)
            W = np.random.normal(0, 1)
            Y = X * W 
            sample.append(X)
            sample.append(Y)
        samples.append(sample)
    samples = torch.tensor(samples, dtype=torch.float32)
    return samples


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
    full_output_dir = os.path.join(output_dir, "gauss")
    os.makedirs(full_output_dir, exist_ok=True)

    output_path = os.path.join(full_output_dir, f"g_{dataset_type}_{num_samples}_{num_features}d.txt")
    np.savetxt(output_path, dataset.numpy(), fmt="%.8f", delimiter=",")
    print(f"{dataset_type.capitalize()} dataset saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Butterfly datasets.")
    parser.add_argument("--dataset_type", type=str, required=True,
                        choices=["trn", "val", "tst"],
                        help="Type of dataset to generate.")
    parser.add_argument("--num_samples", type=int, required=True,
                        help="Number of samples to generate.")
    parser.add_argument("--num_pairs", type=int, required=True,
                        help="Number of (X, Y) pairs in each sample.")
    parser.add_argument("--output_dir", type=str, default="../../data",
                        help="Directory to save the dataset.")
    args = parser.parse_args()

    # Generate dataset
    dataset = generate_butterfly(args.num_samples, args.num_pairs)
    save_dataset(dataset, args.output_dir, args.dataset_type, args.num_samples, args.num_pairs)


if __name__ == "__main__":
    main()
