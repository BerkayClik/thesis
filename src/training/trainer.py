"""
Training module.

Provides training loop with early stopping, validation, and checkpointing.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable
import os


class Trainer:
    """
    Trainer class for model training and validation.

    Args:
        model: PyTorch model to train.
        optimizer: Optimizer instance.
        loss_fn: Loss function.
        device: Device to train on.
        checkpoint_dir: Directory for saving checkpoints.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        device: torch.device,
        checkpoint_dir: str = "checkpoints"
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        self.model.to(device)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader.

        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.loss_fn(pred.squeeze(-1), y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def validate(self, dataloader: DataLoader) -> float:
        """
        Run validation.

        Args:
            dataloader: Validation data loader.

        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)

                pred = self.model(x)
                loss = self.loss_fn(pred.squeeze(-1), y)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        patience: int = 10,
        verbose: bool = True
    ) -> Dict:
        """
        Full training loop with early stopping.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            num_epochs: Maximum number of epochs.
            patience: Early stopping patience.
            verbose: Whether to print progress.

        Returns:
            Dictionary with training history.
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'best_epoch': 0
        }

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Train Loss: {train_loss:.6f} - "
                      f"Val Loss: {val_loss:.6f}")

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                history['best_epoch'] = epoch
                patience_counter = 0
                self.save_checkpoint('best_model.pt')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

        return history

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
