import os
import copy
import numpy as np
import torch
import torch.optim as optim
from models.UMNN import MonotonicNN
from tqdm import tqdm
import matplotlib.pyplot as plt

class MapTrainer:
    """
    Handles training of UMNN-based models for learning mappings.
    """
    def __init__(self, base_map, num_features, hidden_layers, num_steps, learning_rate, max_epochs):
        self.base_map = base_map
        self.num_features = num_features
        self.hidden_layers = hidden_layers
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def train_model(self, training_data, target_index, other_indices, regularizations, validation_data):
        """
        Trains a single model for the specified feature index and regularizations.

        Args:
            training_data (torch.Tensor): Training data.
            target_index (int): Index of the target feature being mapped.
            other_indices (list[int]): Indices of the non-target features.
            regularizations (list[float]): Regularization coefficients to evaluate.
            validation_data (torch.Tensor): Validation dataset.

        Returns:
            tuple: Best model, optimal regularization value, and all trained models.
        """
        best_validation_loss = float('inf')
        optimal_regularization = 0
        optimal_model = None
        models_per_regularization = {}

        for reg_value in tqdm(regularizations, desc='Training with Regularizations', leave=False):
            model = copy.deepcopy(self.base_map)[target_index]
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            early_stop_counter = 0
            best_epoch = 0
            best_loss_for_reg = float('inf')

            for epoch in range(self.max_epochs):
                input_data = training_data.detach().requires_grad_(True)
                non_target_data = input_data[:, other_indices]
                target_data = input_data[:, [target_index]]

                mapped_output = model(target_data, non_target_data)
                jacobian = torch.autograd.grad(mapped_output, target_data, torch.ones_like(mapped_output), create_graph=True)[0]
                loss = (0.5 * mapped_output**2 - torch.log(jacobian)).mean()
                regularization_term = torch.sqrt((jacobian**2).mean())
                loss += reg_value * regularization_term

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Validation
                val_output, val_jacobian = self.evaluate_map(validation_data, other_indices, target_index, model)
                val_loss = self.calculate_loss(val_output, val_jacobian)

                if val_loss[1] < best_loss_for_reg:
                    best_loss_for_reg = val_loss[1]
                    if val_loss[1] < best_validation_loss:
                        best_validation_loss = val_loss[1]
                        optimal_regularization = reg_value
                        optimal_model = model
                    best_epoch = epoch
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

                if early_stop_counter >= 10:
                    print(f'Early stopping at Epoch {epoch} for best epoch {best_epoch}.')
                    break

            models_per_regularization.setdefault(reg_value, []).append(model)

        return optimal_model, optimal_regularization, models_per_regularization

    @staticmethod
    def evaluate_map(data, non_target_indices, target_index, model):
        """
        Evaluates the model on a dataset.

        Args:
            data (torch.Tensor): Input dataset.
            non_target_indices (list[int]): Indices of the non-target features.
            target_index (int): Index of the target feature.
            model (torch.nn.Module): Trained model to evaluate.

        Returns:
            tuple: Model output and Jacobian matrix.
        """
        data = data.detach().requires_grad_(True)
        non_target_data = data[:, non_target_indices]
        target_data = data[:, [target_index]]
        output = model(target_data, non_target_data)
        jacobian = torch.autograd.grad(output, target_data, torch.ones_like(output), create_graph=True)[0]
        return output, jacobian

    @staticmethod
    def calculate_loss(mapped_output, jacobian):
        """
        Computes loss values for the given output and Jacobian.

        Args:
            mapped_output (torch.Tensor): Model output.
            jacobian (torch.Tensor): Jacobian matrix.

        Returns:
            tuple: Mean loss and regularization loss.
        """
        mean_loss = (0.5 * mapped_output**2 - torch.log(jacobian)).mean()
        regularization_loss = torch.sqrt((jacobian**2).mean())
        return mean_loss, regularization_loss

class PrecisionMatrixComputer:
    """
    Computes precision matrices based on trained UMNN models.
    """
    def __init__(self, trained_models, test_data, num_features):
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
                    first_derivative = torch.autograd.grad(output, data, torch.ones_like(output), create_graph=True)[0]
                    first_derivative = torch.log(torch.abs(first_derivative))
                    second_derivative = torch.autograd.grad(first_derivative[:, [feature_idx]], data, torch.ones_like(first_derivative[:, [feature_idx]]), create_graph=True)[0]
                    third_derivative = torch.autograd.grad(second_derivative[:, [feature_idx]], data, torch.ones_like(second_derivative[:, [feature_idx]]), create_graph=True)[0]

                    second_term = torch.abs(third_derivative[:, [i]]).mean().item()
                    first_half = -0.5 * (output**2)
                    first_half_derivative = torch.autograd.grad(first_half, data, torch.ones_like(first_half), create_graph=True)[0]
                    second_half_derivative = torch.autograd.grad(first_half_derivative[:, [feature_idx]], data, torch.ones_like(first_half_derivative[:, [feature_idx]]), create_graph=True)[0]

                    first_term = torch.abs(second_half_derivative[:, [i]]).mean().item()
                    row.append(first_term + second_term)
                else:
                    row.append(1)
            precision_matrix.append(row)
        return np.array(precision_matrix)

    @staticmethod
    def visualize(matrix, title="Precision Matrix Visualization"):
        """
        Displays a precision matrix as a heatmap.

        Args:
            matrix (np.ndarray): Precision matrix to visualize.
            title (str): Title for the plot.
        """
        plt.figure(figsize=(8, 6))
        plt.imshow(matrix, cmap='gray', interpolation='nearest')
        plt.colorbar()
        plt.title(title)
        plt.show()

# Example usage in a separate script
# Create a new script in a different folder to run experiments
# from map_trainer import MapTrainer
# from precision_matrix_computer import PrecisionMatrixComputer
