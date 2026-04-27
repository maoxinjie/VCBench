"""
GenePert Network Components

This module contains the neural network components for the GenePert model,
adapted from the original GenePert implementation for integration with VCBench.
"""

import torch
import torch.nn as nn


class GenePertMLP(nn.Module):
    """
    Multi-Layer Perceptron for GenePert model.

    This is a simple 2-layer MLP with ReLU activation that maps from
    gene embeddings to gene expression changes.

    Args:
        input_dim: Dimension of input features (gene embedding dimension)
        output_dim: Dimension of output (number of genes)
        hidden_size: Size of hidden layer (default: 128)
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_size: int = 128):
        super(GenePertMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim) with ReLU applied
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)  # Non-negative output (gene expression is non-negative)
        return x


