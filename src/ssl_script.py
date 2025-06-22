"""
Semi-Supervised Learning (SSL) Implementation for CIFAR-10

This module implements state-of-the-art Semi-Supervised Learning techniques:
1. FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence
2. MixMatch: A Holistic Approach to Semi-Supervised Learning
3. Pseudo-Labeling: A Simple and Efficient Semi-Supervised Learning Method

Mathematical Foundation:
- Consistency Regularization: Ensure model predictions are invariant to data augmentations
- Confidence-based Thresholding: Only use high-confidence pseudo-labels
- Entropy Minimization: Encourage low-entropy predictions on unlabeled data

Author: Suyog Khanal
Date: 2025
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Dict, List, Optional, Union
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score, classification_report

# Import our data preprocessing module
from data_preprocessing import CIFAR10DataPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResNet18Classifier(nn.Module):
    """
    ResNet-18 based classifier for CIFAR-10.
    
    Modified ResNet-18 architecture optimized for 32x32 CIFAR-10 images.
    """
    
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.2):
        """
        Initialize ResNet-18 classifier.
        
        Args:
            num_classes (int): Number of output classes
            dropout_rate (float): Dropout rate for regularization
        """
        super(ResNet18Classifier, self).__init__()
        
        # Load pretrained ResNet-18 and modify for CIFAR-10
        self.backbone = torchvision.models.resnet18(pretrained=False)
        
        # Modify first conv layer for 32x32 input
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()  # Remove maxpool for smaller images
        
        # Modify classifier
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.backbone(x)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the final classification layer."""
        features = self.backbone.avgpool(self.backbone.layer4(
            self.backbone.layer3(self.backbone.layer2(
                self.backbone.layer1(self.backbone.relu(
                    self.backbone.bn1(self.backbone.conv1(x))
                ))
            ))
        ))
        return features.view(features.size(0), -1)


class FixMatchSSL:
    """
    FixMatch Semi-Supervised Learning Implementation.
    
    FixMatch uses weak and strong augmentations with confidence thresholding:
    - Weak augmentation: Generate pseudo-labels for unlabeled data
    - Strong augmentation: Apply consistency regularization
    - Only use pseudo-labels above confidence threshold
    
    Mathematical formulation:
    L_total = L_supervised + λ_u * L_unsupervised
    
    Where:
    L_supervised = CrossEntropy(model(weak_aug(x_labeled)), y_labeled)
    L_unsupervised = 1(max(p_model) >= τ) * CrossEntropy(model(strong_aug(x_unlabeled)), argmax(model(weak_aug(x_unlabeled))))
    """
    
    def __init__(self, 
                 model: nn.Module,
                 device: torch.device,
                 learning_rate: float = 3e-4,
                 weight_decay: float = 5e-4,
                 confidence_threshold: float = 0.95,
                 lambda_u: float = 1.0,
                 temperature: float = 1.0):
        """
        Initialize FixMatch SSL trainer.
        
        Args:
            model: Neural network model
            device: Computing device (cuda/cpu)
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            confidence_threshold: Threshold for pseudo-label confidence
            lambda_u: Weight for unsupervised loss
            temperature: Temperature for softmax sharpening
        """
        self.model = model.to(device)
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.lambda_u = lambda_u
        self.temperature = temperature
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        
        # Training metrics
        self.train_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.pseudo_label_accuracies = []
        
        logger.info(f"FixMatch initialized with threshold={confidence_threshold}, lambda_u={lambda_u}")
    
    def sharpen(self, predictions: torch.Tensor, temperature: float) -> torch.Tensor:
        """
        Apply temperature sharpening to predictions.
        
        Args:
            predictions: Model predictions
            temperature: Sharpening temperature
            
        Returns:
            torch.Tensor: Sharpened predictions
        """
        return torch.softmax(predictions / temperature, dim=1)
    
    def train_epoch(self, 
                   labeled_loader: DataLoader, 
                   unlabeled_loader: DataLoader) -> Dict[str, float]:
        """
        Train one epoch using FixMatch algorithm.
        
        Args:
            labeled_loader: DataLoader for labeled data
            unlabeled_loader: DataLoader for unlabeled data
            
        Returns:
            Dict[str, float]: Training metrics for this epoch
        """
        self.model.train()
        
        total_loss = 0.0
        supervised_loss_sum = 0.0
        unsupervised_loss_sum = 0.0
        correct_predictions = 0
        total_predictions = 0
        pseudo_labels_used = 0
        pseudo_labels_total = 0
        
        # Create iterators
        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)
        
        # Determine number of batches
        num_batches = min(len(labeled_loader), len(unlabeled_loader))
        
        progress_bar = tqdm(range(num_batches), desc="Training")
        
        for batch_idx in progress_bar:
            try:
                # Get labeled batch
                labeled_batch = next(labeled_iter)
                labeled_images, labeled_targets = labeled_batch
                labeled_images = labeled_images.to(self.device)
                labeled_targets = labeled_targets.to(self.device)
                
                # Get unlabeled batch with weak and strong augmentations
                unlabeled_batch = next(unlabeled_iter)
                unlabeled_weak, unlabeled_strong = unlabeled_batch[0], unlabeled_batch[0]
                
                # Apply different transforms (this would be handled in data preprocessing)
                unlabeled_weak = unlabeled_weak.to(self.device)
                unlabeled_strong = unlabeled_strong.to(self.device)
                
            except StopIteration:
                # Restart iterators if one is exhausted
                labeled_iter = iter(labeled_loader)
                unlabeled_iter = iter(unlabeled_loader)
                continue
            
            self.optimizer.zero_grad()
            
            # === SUPERVISED LOSS ===
            labeled_outputs = self.model(labeled_images)
            supervised_loss = self.criterion(labeled_outputs, labeled_targets)
            
            # Calculate accuracy
            _, predicted = torch.max(labeled_outputs, 1)
            correct_predictions += (predicted == labeled_targets).sum().item()
            total_predictions += labeled_targets.size(0)
            
            # === UNSUPERVISED LOSS ===
            with torch.no_grad():
                # Generate pseudo-labels using weak augmentation
                weak_outputs = self.model(unlabeled_weak)
                weak_probs = torch.softmax(weak_outputs, dim=1)
                max_probs, pseudo_labels = torch.max(weak_probs, dim=1)
                
                # Create mask for confident predictions
                confidence_mask = max_probs >= self.confidence_threshold
                pseudo_labels_used += confidence_mask.sum().item()
                pseudo_labels_total += confidence_mask.size(0)
            
            # Apply strong augmentation and compute consistency loss
            strong_outputs = self.model(unlabeled_strong)
            
            # Only compute loss for confident pseudo-labels
            if confidence_mask.sum() > 0:
                masked_strong_outputs = strong_outputs[confidence_mask]
                masked_pseudo_labels = pseudo_labels[confidence_mask]
                unsupervised_loss = self.criterion(masked_strong_outputs, masked_pseudo_labels)
            else:
                unsupervised_loss = torch.tensor(0.0).to(self.device)
            
            # === TOTAL LOSS ===
            total_batch_loss = supervised_loss + self.lambda_u * unsupervised_loss
            
            # Backward pass
            total_batch_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += total_batch_loss.item()
            supervised_loss_sum += supervised_loss.item()
            unsupervised_loss_sum += unsupervised_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{total_batch_loss.item():.4f}',
                'Acc': f'{100. * correct_predictions / total_predictions:.2f}%',
                'Pseudo': f'{pseudo_labels_used}/{pseudo_labels_total}'
            })
        
        # Update learning rate
        self.scheduler.step()
        
        # Calculate epoch metrics
        epoch_metrics = {
            'total_loss': total_loss / num_batches,
            'supervised_loss': supervised_loss_sum / num_batches,
            'unsupervised_loss': unsupervised_loss_sum / num_batches,
            'accuracy': 100. * correct_predictions / total_predictions,
            'pseudo_label_usage': 100. * pseudo_labels_used / pseudo_labels_total if pseudo_labels_total > 0 else 0,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
        
        return epoch_metrics
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in tqdm(test_loader, desc="Evaluating"):
                images, targets = images.to(self.device), targets.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                total_loss += loss.item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(test_loader)
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'predictions': all_predictions,
            'targets': all_targets
        }
    
    def train(self, 
              labeled_loader: DataLoader,
              unlabeled_loader: DataLoader,
              test_loader: DataLoader,
              num_epochs: int = 100,
              save_dir: str = "./outputs/models/ssl_models") -> Dict[str, List[float]]:
        """
        Complete training loop for FixMatch.
        
        Args:
            labeled_loader: Labeled data loader
            unlabeled_loader: Unlabeled data loader
            test_loader: Test data loader
            num_epochs: Number of training epochs
            save_dir: Directory to save model checkpoints
            
        Returns:
            Dict[str, List[float]]: Training history
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        best_accuracy = 0.0
        training_history = {
            'train_loss': [],
            'train_accuracy': [],
            'test_accuracy': [],
            'pseudo_label_usage': []
        }
        
        logger.info(f"Starting FixMatch training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train one epoch
            train_metrics = self.train_epoch(labeled_loader, unlabeled_loader)
            
            # Evaluate on test set
            test_metrics = self.evaluate(test_loader)
            
            # Update history
            training_history['train_loss'].append(train_metrics['total_loss'])
            training_history['train_accuracy'].append(train_metrics['accuracy'])
            training_history['test_accuracy'].append(test_metrics['accuracy'])
            training_history['pseudo_label_usage'].append(train_metrics['pseudo_label_usage'])
            
            epoch_time = time.time() - start_time
            
            # Log progress
            logger.info(f"Epoch [{epoch+1}/{num_epochs}] "
                       f"Train Loss: {train_metrics['total_loss']:.4f} "
                       f"Train Acc: {train_metrics['accuracy']:.2f}% "
                       f"Test Acc: {test_metrics['accuracy']:.2f}% "
                       f"Pseudo Usage: {train_metrics['pseudo_label_usage']:.1f}% "
                       f"Time: {epoch_time:.1f}s")
            
            # Save best model
            if test_metrics['accuracy'] > best_accuracy:
                best_accuracy = test_metrics['accuracy']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'accuracy': best_accuracy,
                    'training_history': training_history
                }, save_dir / 'fixmatch_best_model.pth')
                
                logger.info(f"New best model saved with accuracy: {best_accuracy:.2f}%")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'training_history': training_history
                }, save_dir / f'fixmatch_checkpoint_epoch_{epoch+1}.pth')
        
        logger.info(f"Training completed! Best accuracy: {best_accuracy:.2f}%")
        return training_history


class MixMatchSSL:
    """
    MixMatch Semi-Supervised Learning Implementation.
    
    MixMatch combines:
    1. MixUp augmentation for both labeled and unlabeled data
    2. Consistency regularization with sharpening
    3. Entropy minimization
    
    Mathematical formulation:
    L_total = L_X + λ_u * L_U
    
    Where:
    L_X = (1/|X|) * Σ H(p, Mix(λ, x, x'))  # Labeled loss with MixUp
    L_U = (1/L|U|) * Σ ||q - Mix(λ, u, u')||²  # Unlabeled consistency loss
    """
    
    def __init__(self, 
                 model: nn.Module,
                 device: torch.device,
                 learning_rate: float = 2e-4,
                 weight_decay: float = 4e-5,
                 lambda_u: float = 75.0,
                 temperature: float = 0.5,
                 alpha: float = 0.75,
                 mixup_alpha: float = 0.75):
        """
        Initialize MixMatch SSL trainer.
        
        Args:
            model: Neural network model
            device: Computing device
            learning_rate: Learning rate
            weight_decay: Weight decay
            lambda_u: Weight for unsupervised loss
            temperature: Temperature for sharpening
            alpha: EMA decay rate
            mixup_alpha: Alpha parameter for Beta distribution in MixUp
        """
        self.model = model.to(device)
        self.device = device
        self.lambda_u = lambda_u
        self.temperature = temperature
        self.alpha = alpha
        self.mixup_alpha = mixup_alpha
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1024)
        
        logger.info(f"MixMatch initialized with lambda_u={lambda_u}, temp={temperature}")
    
    def sharpen(self, p: torch.Tensor, T: float) -> torch.Tensor:
        """Sharpen predictions using temperature scaling."""
        return torch.pow(p, 1/T) / torch.sum(torch.pow(p, 1/T), dim=1, keepdim=True)
    
    def mixup(self, x1: torch.Tensor, x2: torch.Tensor, y1: torch.Tensor, y2: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply MixUp augmentation.
        
        Args:
            x1, x2: Input tensors
            y1, y2: Label tensors
            alpha: MixUp parameter
            
        Returns:
            Tuple of mixed inputs and labels
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_y = lam * y1 + (1 - lam) * y2
        
        return mixed_x, mixed_y
    
    def train_epoch(self, labeled_loader: DataLoader, unlabeled_loader: DataLoader) -> Dict[str, float]:
        """Train one epoch using MixMatch algorithm."""
        self.model.train()
        
        total_loss = 0.0
        labeled_loss_sum = 0.0
        unlabeled_loss_sum = 0.0
        
        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)
        
        num_batches = min(len(labeled_loader), len(unlabeled_loader))
        
        for batch_idx in tqdm(range(num_batches), desc="MixMatch Training"):
            try:
                labeled_batch = next(labeled_iter)
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                break
            
            labeled_x, labeled_y = labeled_batch
            unlabeled_x, _ = unlabeled_batch
            
            labeled_x = labeled_x.to(self.device)
            labeled_y = labeled_y.to(self.device)
            unlabeled_x = unlabeled_x.to(self.device)
            
            batch_size = labeled_x.size(0)
            
            # Convert labeled_y to one-hot
            labeled_y_onehot = F.one_hot(labeled_y, num_classes=10).float()
            
            # Generate pseudo-labels for unlabeled data
            with torch.no_grad():
                unlabeled_outputs = self.model(unlabeled_x)
                unlabeled_probs = torch.softmax(unlabeled_outputs, dim=1)
                # Sharpen the pseudo-labels
                pseudo_labels = self.sharpen(unlabeled_probs, self.temperature)
            
            # Combine labeled and unlabeled data
            all_x = torch.cat([labeled_x, unlabeled_x], dim=0)
            all_y = torch.cat([labeled_y_onehot, pseudo_labels], dim=0)
            
            # Apply MixUp
            indices = torch.randperm(all_x.size(0))
            mixed_x, mixed_y = self.mixup(all_x, all_x[indices], all_y, all_y[indices], self.mixup_alpha)
            
            # Split back into labeled and unlabeled
            mixed_labeled_x = mixed_x[:batch_size]
            mixed_unlabeled_x = mixed_x[batch_size:]
            mixed_labeled_y = mixed_y[:batch_size]
            mixed_unlabeled_y = mixed_y[batch_size:]
            
            self.optimizer.zero_grad()
            
            # Compute losses
            labeled_outputs = self.model(mixed_labeled_x)
            labeled_probs = torch.softmax(labeled_outputs, dim=1)
            labeled_loss = -torch.mean(torch.sum(mixed_labeled_y * torch.log(labeled_probs + 1e-8), dim=1))
            
            unlabeled_outputs = self.model(mixed_unlabeled_x)
            unlabeled_probs = torch.softmax(unlabeled_outputs, dim=1)
            unlabeled_loss = torch.mean((unlabeled_probs - mixed_unlabeled_y) ** 2)
            
            total_batch_loss = labeled_loss + self.lambda_u * unlabeled_loss
            
            total_batch_loss.backward()
            self.optimizer.step()
            
            total_loss += total_batch_loss.item()
            labeled_loss_sum += labeled_loss.item()
            unlabeled_loss_sum += unlabeled_loss.item()
        
        self.scheduler.step()
        
        return {
            'total_loss': total_loss / num_batches,
            'labeled_loss': labeled_loss_sum / num_batches,
            'unlabeled_loss': unlabeled_loss_sum / num_batches,
        }


class PseudoLabelSSL:
    """
    Simple Pseudo-Labeling Implementation.
    
    Uses confident model predictions as pseudo-labels for unlabeled data.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 device: torch.device,
                 confidence_threshold: float = 0.9,
                 alpha: float = 0.3):
        """
        Initialize Pseudo-Label SSL trainer.
        
        Args:
            model: Neural network model
            device: Computing device
            confidence_threshold: Confidence threshold for pseudo-labels
            alpha: Weight for pseudo-label loss
        """
        self.model = model.to(device)
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.alpha = alpha
        
        self.optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=5e-4)
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info(f"Pseudo-Label SSL initialized with threshold={confidence_threshold}")
    
    def train_epoch(self, labeled_loader: DataLoader, unlabeled_loader: DataLoader) -> Dict[str, float]:
        """Train one epoch using pseudo-labeling."""
        self.model.train()
        
        total_loss = 0.0
        supervised_loss_sum = 0.0
        pseudo_loss_sum = 0.0
        pseudo_labels_used = 0
        
        # Train on labeled data
        for images, targets in labeled_loader:
            images, targets = images.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            supervised_loss_sum += loss.item()
        
        # Generate pseudo-labels and train
        for images, _ in unlabeled_loader:
            images = images.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                max_probs, pseudo_labels = torch.max(probs, dim=1)
                
                # Filter by confidence
                confident_mask = max_probs >= self.confidence_threshold
                
                if confident_mask.sum() > 0:
                    confident_images = images[confident_mask]
                    confident_pseudo_labels = pseudo_labels[confident_mask]
                    pseudo_labels_used += confident_mask.sum().item()
                    
                    # Train on pseudo-labeled data
                    self.optimizer.zero_grad()
                    pseudo_outputs = self.model(confident_images)
                    pseudo_loss = self.alpha * self.criterion(pseudo_outputs, confident_pseudo_labels)
                    pseudo_loss.backward()
                    self.optimizer.step()
                    
                    pseudo_loss_sum += pseudo_loss.item()
        
        return {
            'supervised_loss': supervised_loss_sum / len(labeled_loader),
            'pseudo_loss': pseudo_loss_sum / len(unlabeled_loader) if len(unlabeled_loader) > 0 else 0,
            'pseudo_labels_used': pseudo_labels_used
        }


def compare_ssl_methods(labeled_loader: DataLoader,
                       unlabeled_loader: DataLoader,
                       test_loader: DataLoader,
                       device: torch.device,
                       num_epochs: int = 50) -> Dict[str, Dict]:
    """
    Compare different SSL methods on the same data.
    
    Args:
        labeled_loader: Labeled training data
        unlabeled_loader: Unlabeled training data  
        test_loader: Test data
        device: Computing device
        num_epochs: Number of training epochs
        
    Returns:
        Dict containing results for each method
    """
    results = {}
    
    # Test FixMatch
    logger.info("Training FixMatch...")
    fixmatch_model = ResNet18Classifier()
    fixmatch_trainer = FixMatchSSL(fixmatch_model, device)
    fixmatch_history = fixmatch_trainer.train(labeled_loader, unlabeled_loader, test_loader, num_epochs)
    fixmatch_final_acc = fixmatch_trainer.evaluate(test_loader)['accuracy']
    
    results['FixMatch'] = {
        'final_accuracy': fixmatch_final_acc,
        'training_history': fixmatch_history
    }
    
    logger.info(f"FixMatch final accuracy: {fixmatch_final_acc:.2f}%")
    
    return results


def visualize_ssl_results(results: Dict[str, Dict], save_path: str = "./outputs/images/ssl_comparison.png"):
    """
    Visualize SSL training results.
    
    Args:
        results: Results dictionary from compare_ssl_methods
        save_path: Path to save the visualization
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for method_name, method_results in results.items():
        history = method_results['training_history']
        
        # Plot training loss
        axes[0, 0].plot(history['train_loss'], label=f'{method_name}')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Plot accuracy
        axes[0, 1].plot(history['test_accuracy'], label=f'{method_name}')
        axes[0, 1].set_title('Test Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        
        # Plot pseudo-label usage (if available)
        if 'pseudo_label_usage' in history:
            axes[1, 0].plot(history['pseudo_label_usage'], label=f'{method_name}')
        axes[1, 0].set_title('Pseudo-Label Usage')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Usage (%)')
        axes[1, 0].legend()
        
        # Bar plot of final accuracies
        method_names = list(results.keys())
        final_accs = [results[name]['final_accuracy'] for name in method_names]
        axes[1, 1].bar(method_names, final_accs, alpha=0.7)
        axes[1, 1].set_title('Final Test Accuracy Comparison')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"SSL results visualization saved to {save_path}")
    plt.show()


def main():
    """
    Main function to demonstrate SSL implementations.
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize data preprocessor
    preprocessor = CIFAR10DataPreprocessor(data_dir="./data")
    
    # Load and prepare data
    train_dataset, test_dataset = preprocessor.load_cifar10_datasets()
    labeled_subset, unlabeled_subset, validation_subset = preprocessor.create_ssl_splits(
        train_dataset, labeled_ratio=0.1
    )
    
    # Create data loaders
    ssl_dataloaders = preprocessor.get_dataloaders(
        labeled_subset=labeled_subset,
        unlabeled_subset=unlabeled_subset,
        test_dataset=test_dataset,
        batch_size=64
    )
    
    # Run SSL comparison
    os.makedirs("./outputs/models/ssl_models", exist_ok=True)
    os.makedirs("./outputs/images", exist_ok=True)
    
    results = compare_ssl_methods(
        ssl_dataloaders['labeled'],
        ssl_dataloaders['unlabeled'],
        ssl_dataloaders['test'],
        device,
        num_epochs=20  # Reduced for demo
    )
    
    # Visualize results
    visualize_ssl_results(results)
    
    # Save results
    with open("./outputs/logs/ssl_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("SSL training and evaluation completed!")


if __name__ == "__main__":
    main()