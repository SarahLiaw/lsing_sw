import numpy as np
import torch
import matplotlib.pyplot as plt


class PrecisionMatrixComputer:
    """
    Computes precision matrices based on trained UMNN models.
    """
    def __init__(self, trained_models, test_data, num_features):
        """
        Initialize the PrecisionMatrixComputer.

        Args:
            trained_models (list): List of trained UMNN models.
            test_data (torch.Tensor): Test data tensor.
            num_features (int): Number of features in the dataset.
        """
        self.trained_models = trained_models
        self.test_data = test_data
        self.num_features = num_features

    def compute(self):
        """
        Computes the precision matrix.

        Returns:
            np.ndarray: Computed precision matrix.
        """
        precision_matrix = []
        for feature_idx, model in enumerate(self.trained_models):
            model.eval()
            row = []
            non_target_indices = [i for i in range(self.test_data.shape[1]) if i != feature_idx]

            data = self.test_data.detach().requires_grad_(True)
            non_target_data = data[:, non_target_indices]
            target_data = data[:, [feature_idx]]

            output = model(target_data, non_target_data)
            for i in range(self.num_features):
                if i != feature_idx:
                    # Compute derivatives
                    first_derivative = torch.autograd.grad(
                        output, data, torch.ones_like(output), create_graph=True
                    )[0]
                    first_derivative = torch.log(torch.abs(first_derivative))
                    second_derivative = torch.autograd.grad(
                        first_derivative[:, [feature_idx]],
                        data,
                        torch.ones_like(first_derivative[:, [feature_idx]]),
                        create_graph=True
                    )[0]
                    third_derivative = torch.autograd.grad(
                        second_derivative[:, [feature_idx]],
                        data,
                        torch.ones_like(second_derivative[:, [feature_idx]]),
                        create_graph=True
                    )[0]

                    # Compute terms
                    second_term = torch.abs(third_derivative[:, [i]]).mean().item()
                    first_half = -0.5 * (output**2)
                    first_half_derivative = torch.autograd.grad(
                        first_half, data, torch.ones_like(first_half), create_graph=True
                    )[0]
                    second_half_derivative = torch.autograd.grad(
                        first_half_derivative[:, [feature_idx]],
                        data,
                        torch.ones_like(first_half_derivative[:, [feature_idx]]),
                        create_graph=True
                    )[0]
                    first_term = torch.abs(second_half_derivative[:, [i]]).mean().item()

                    row.append(first_term + second_term)
                else:
                    row.append(1)  # Diagonal elements are set to 1
            precision_matrix.append(row)
        return np.array(precision_matrix)


    def make_symmetric(self, precision_matrix):
        """
        Converts the precision matrix into a symmetric, normalized matrix.

        Args:
            precision_matrix (np.ndarray): The computed precision matrix.

        Returns:
            np.ndarray: Symmetric, normalized precision matrix.
        """
        transpose_matrix = precision_matrix.T
        symmetric_matrix = (transpose_matrix + precision_matrix) / 2
        normalized_matrix = symmetric_matrix / np.max(symmetric_matrix)
        np.fill_diagonal(normalized_matrix, 1)
        return normalized_matrix


    @staticmethod
    def visualize(matrix, title="Precision Matrix Visualization"):
        """
        Displays a precision matrix as a heatmap.

        Args:
            matrix (np.ndarray): Precision matrix to visualize.
            title (str): Title for the plot.
        """
        plt.figure(figsize=(8, 6))
        plt.xticks(np.arange(0, len(matrix), 1), np.arange(1, len(matrix) + 1))
        plt.yticks(np.arange(0, len(matrix), 1), np.arange(1, len(matrix) + 1))
        plt.imshow(matrix, cmap='gray', interpolation='nearest')

        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                plt.text(j, i, f'{matrix[i, j]:.3f}', ha='center', va='center', color='white')

        plt.colorbar()