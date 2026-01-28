"""
Training module.

Provides training loop with early stopping, validation, and checkpointing.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, Optional, Callable, List, Any, Union
import os
import warnings
from tqdm import tqdm


def compute_gradient_norms(model: nn.Module) -> Dict[str, float]:
    """
    Compute gradient norms for each parameter group.

    Args:
        model: PyTorch model after backward pass.

    Returns:
        Dict mapping parameter names to gradient norms.
    """
    grad_norms = {}
    total_norm = 0.0

    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            grad_norms[name] = param_norm
            total_norm += param_norm ** 2

    grad_norms['total'] = total_norm ** 0.5
    return grad_norms


def compute_weight_stats(model: nn.Module) -> Dict[str, Dict[str, float]]:
    """
    Compute weight statistics for each parameter.

    Args:
        model: PyTorch model.

    Returns:
        Dict mapping parameter names to stats (mean, std, min, max, norm).
    """
    stats = {}
    for name, param in model.named_parameters():
        data = param.data
        stats[name] = {
            'mean': data.mean().item(),
            'std': data.std(unbiased=False).item() if data.numel() > 1 else 0.0,
            'min': data.min().item(),
            'max': data.max().item(),
            'norm': data.norm(2).item()
        }
    return stats


class Trainer:
    """
    Trainer class for model training and validation.

    Args:
        model: PyTorch model to train.
        optimizer: Optimizer instance.
        loss_fn: Loss function.
        device: Device to train on.
        checkpoint_dir: Base directory for saving checkpoints.
        model_name: Optional model name for checkpoint subdirectory organization.
        seed: Optional seed for checkpoint subdirectory organization.
        max_grad_norm: Maximum gradient norm for clipping. None disables clipping.
        scheduler_config: Optional dict with scheduler settings.
            Example: {'type': 'reduce_on_plateau', 'factor': 0.5, 'patience': 5}
        debug: Enable debug mode with gradient tracking.
        grad_explosion_threshold: Gradient norm threshold for explosion warning.

    Checkpoint paths are organized as: checkpoint_dir/model_name/seed_N/
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        device: torch.device,
        checkpoint_dir: str = "checkpoints",
        model_name: Optional[str] = None,
        seed: Optional[int] = None,
        max_grad_norm: Optional[float] = 1.0,
        scheduler_config: Optional[Dict[str, Any]] = None,
        debug: bool = False,
        grad_explosion_threshold: float = 100.0
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

        # Build checkpoint path with optional model_name and seed subdirectories
        if model_name:
            checkpoint_dir = os.path.join(checkpoint_dir, model_name)
        if seed is not None:
            checkpoint_dir = os.path.join(checkpoint_dir, f"seed_{seed}")

        self.checkpoint_dir = checkpoint_dir
        self.max_grad_norm = max_grad_norm
        self.debug = debug
        self.grad_explosion_threshold = grad_explosion_threshold
        self.grad_explosion_count = 0

        self.model.to(device)

        # Initialize learning rate scheduler
        self.scheduler = None
        if scheduler_config:
            self.scheduler = self._create_scheduler(scheduler_config)

        # Store initial weight stats if debug mode
        if self.debug:
            self.initial_weight_stats = compute_weight_stats(model)
            self._log_weight_stats("Initial weight statistics:", self.initial_weight_stats)

    def _create_scheduler(self, config: Dict[str, Any]):
        """Create learning rate scheduler from config."""
        scheduler_type = config.get('type', 'reduce_on_plateau')
        if scheduler_type == 'reduce_on_plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=config.get('factor', 0.5),
                patience=config.get('patience', 5)
            )
        else:
            warnings.warn(f"Unknown scheduler type: {scheduler_type}, skipping scheduler")
            return None

    def _log_weight_stats(self, title: str, stats: Dict):
        """Log weight statistics in a readable format."""
        print(f"\n{title}")
        print("-" * 60)
        for name, s in stats.items():
            print(f"  {name}:")
            print(f"    mean={s['mean']:.6f}, std={s['std']:.6f}, "
                  f"min={s['min']:.6f}, max={s['max']:.6f}, norm={s['norm']:.4f}")

    def train_epoch(self, dataloader: DataLoader, track_gradients: bool = False) -> Dict:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader.
            track_gradients: Whether to track gradient norms.

        Returns:
            Dict with 'loss' and optionally 'grad_norms'.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        all_grad_norms = []
        nan_detected = False

        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.loss_fn(pred.squeeze(-1), y)

            # Check for NaN/Inf in loss
            if torch.isnan(loss) or torch.isinf(loss):
                nan_detected = True
                warnings.warn(f"NaN/Inf detected in loss at batch {num_batches}. Skipping batch.")
                continue

            loss.backward()

            # Track gradients before clipping
            if track_gradients or self.debug:
                grad_norms = compute_gradient_norms(self.model)
                total_grad_norm = grad_norms['total']
                all_grad_norms.append(total_grad_norm)

                # Early warning for gradient explosion
                if total_grad_norm > self.grad_explosion_threshold:
                    self.grad_explosion_count += 1
                    if self.grad_explosion_count <= 5:  # Limit warning spam
                        warnings.warn(
                            f"Gradient explosion detected! Norm={total_grad_norm:.2f} "
                            f"(threshold={self.grad_explosion_threshold}). "
                            f"Count: {self.grad_explosion_count}"
                        )

            # Gradient clipping
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        result = {
            'loss': total_loss / max(num_batches, 1),
            'nan_detected': nan_detected
        }

        if track_gradients or self.debug:
            result['grad_norm_mean'] = sum(all_grad_norms) / len(all_grad_norms) if all_grad_norms else 0.0
            result['grad_norm_max'] = max(all_grad_norms) if all_grad_norms else 0.0
            result['grad_norm_min'] = min(all_grad_norms) if all_grad_norms else 0.0

        return result

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
        verbose: Union[bool, int] = 1
    ) -> Dict:
        """
        Full training loop with early stopping.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            num_epochs: Maximum number of epochs.
            patience: Early stopping patience.
            verbose: Verbosity level.
                0: Silent (no output)
                1: tqdm progress bar + improvement messages (default)
                2: Detailed per-epoch output (legacy behavior)
                True/False: Mapped to 1/0 for backward compatibility.

        Returns:
            Dictionary with training history.
        """
        # Handle backward compatibility for bool verbose
        if isinstance(verbose, bool):
            verbose = 1 if verbose else 0

        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'best_epoch': 0,
            'nan_epochs': [],
            'grad_explosion_count': 0
        }

        # Reset gradient explosion counter at start of training
        self.grad_explosion_count = 0

        # Add debug tracking if enabled
        if self.debug:
            history['grad_norms'] = []
            history['weight_stats'] = []

        best_val_loss = float('inf')
        patience_counter = 0

        # Create tqdm progress bar for verbose >= 1
        epoch_iterator = tqdm(
            range(num_epochs),
            desc="Training",
            disable=(verbose == 0)
        )

        for epoch in epoch_iterator:
            train_result = self.train_epoch(train_loader, track_gradients=self.debug)
            train_loss = train_result['loss']
            val_loss = self.validate(val_loader)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

            # Update tqdm postfix with current losses
            if verbose >= 1:
                epoch_iterator.set_postfix({
                    'train': f'{train_loss:.4f}',
                    'val': f'{val_loss:.4f}'
                })

            # Track NaN detection
            if train_result.get('nan_detected', False):
                history['nan_epochs'].append(epoch)

            # Track debug info
            if self.debug:
                history['grad_norms'].append({
                    'mean': train_result.get('grad_norm_mean', 0.0),
                    'max': train_result.get('grad_norm_max', 0.0),
                    'min': train_result.get('grad_norm_min', 0.0)
                })

            # Step the scheduler based on validation loss
            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            # Verbose level 2: detailed per-epoch output (legacy behavior)
            if verbose >= 2:
                msg = (f"Epoch {epoch+1}/{num_epochs} - "
                       f"Train Loss: {train_loss:.6f} - "
                       f"Val Loss: {val_loss:.6f}")
                if self.debug:
                    msg += f" - Grad Norm: {train_result.get('grad_norm_mean', 0.0):.4f}"
                    msg += f" - LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                tqdm.write(msg)

            # Early stopping check
            if val_loss < best_val_loss:
                improvement = best_val_loss - val_loss
                # Verbose level 1: only print on improvement
                if verbose == 1 and best_val_loss != float('inf'):
                    tqdm.write(
                        f"  [Epoch {epoch+1}] New best! "
                        f"Val Loss: {val_loss:.6f} (improved by {improvement:.6f})"
                    )
                best_val_loss = val_loss
                history['best_epoch'] = epoch
                patience_counter = 0
                self.save_checkpoint('best_model.pt')
            else:
                patience_counter += 1
                # Warn when patience is half exhausted
                if verbose == 1 and patience_counter == patience // 2 and patience_counter > 0:
                    tqdm.write(
                        f"  [Epoch {epoch+1}] Warning: No improvement for "
                        f"{patience_counter} epochs (patience: {patience_counter}/{patience})"
                    )

            if patience_counter >= patience:
                if verbose >= 1:
                    tqdm.write(f"  [Epoch {epoch+1}] Early stopping triggered")
                break

        # Record final gradient explosion count
        history['grad_explosion_count'] = self.grad_explosion_count

        # Print final summary for verbose >= 1
        if verbose >= 1:
            tqdm.write(
                f"  Final: Best epoch {history['best_epoch']+1}, "
                f"Val Loss: {best_val_loss:.6f}"
            )

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
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
