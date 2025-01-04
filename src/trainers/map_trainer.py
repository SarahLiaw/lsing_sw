import copy
import torch
import torch.optim as optim
from tqdm import tqdm


class MapTrainer:
    """
    Handles training of UMNN-based models for learning mappings.
    """
    def __init__(self, base_map, num_features, hidden_layers, num_steps, learning_rate, max_epochs):
        """
        Initialize the MapTrainer.

        Args:
            base_map (list): List of untrained UMNN models (one per feature).
            num_features (int): Number of features in the dataset.
            hidden_layers (list[int]): Structure of hidden layers for the UMNN models.
            num_steps (int): Number of steps for integral approximation.
            learning_rate (float): Learning rate for training.
            max_epochs (int): Maximum number of training epochs.
        """
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

        for reg_value in tqdm(regularizations, desc=f"Training feature {target_index}", leave=False):
            model = copy.deepcopy(self.base_map)[target_index]
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            early_stop_counter = 0
            best_epoch_loss = float('inf')

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

                if val_loss[1] < best_epoch_loss:
                    best_epoch_loss = val_loss[1]
                    if val_loss[1] < best_validation_loss:
                        best_validation_loss = val_loss[1]
                        optimal_regularization = reg_value
                        optimal_model = model
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

                if early_stop_counter >= 10:
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
