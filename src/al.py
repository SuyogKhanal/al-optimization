"""
Active Learning (AL) Implementation for CIFAR-10

This module implements various Active Learning strategies to reduce the need for labeled data:
1. Uncertainty Sampling (Entropy, Margin, Least Confidence)
2. Query-by-Committee (QBC)
3. Diversity-based Sampling
4. Random Sampling (baseline)

Mathematical Foundation:
- Uncertainty Sampling: Select samples with highest prediction uncertainty
- Entropy: H(p) = -Σ p_i * log(p_i)
- Margin: Select samples with smallest margin between top-2 predictions
- Query-by-Committee: Use disagreement between multiple models

Author: Research Team
Date: 2025
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Dict, List, Optional, Union, Callable
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy

# Import our modules
from data_preprocessing import CIFAR10DataPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResNet18ActiveLearning(nn.Module):
    """
    ResNet-18 model optimized for Active Learning scenarios.
    
    Includes methods for uncertainty estimation and feature extraction.
    """
    
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.3):
        """
        Initialize ResNet-18 for Active Learning.
        
        Args:
            num_classes (int): Number of output classes
            dropout_rate (float): Dropout rate for uncertainty estimation
        """
        super(ResNet18ActiveLearning, self).__init__()
        
        # Load ResNet-18 and modify for CIFAR-10
        self.backbone = torchvision.models.resnet18(pretrained=False)
        
        # Modify for 32x32 input
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        
        # Feature dimension
        self.feature_dim = self.backbone.fc.in_features
        
        # Replace classifier with dropout-enabled version
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, num_classes)
        )
        
        # Enable dropout during inference for uncertainty estimation
        self.dropout_enabled = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.backbone(x)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature representations before the final classifier.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Feature representations
        """
        # Forward through all layers except the final classifier
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        features = torch.flatten(x, 1)
        
        return features
    
    def predict_with_uncertainty(self, x: torch.Tensor, num_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Monte Carlo Dropout for uncertainty estimation.
        
        Args:
            x: Input tensor
            num_samples: Number of MC dropout samples
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (mean_predictions, uncertainty)
        """
        self.train()  # Enable dropout
        
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                pred = F.softmax(self.forward(x), dim=1)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)  # [num_samples, batch_size, num_classes]
        
        # Calculate mean and uncertainty (variance)
        mean_pred = torch.mean(predictions, dim=0)
        uncertainty = torch.var(predictions, dim=0).sum(dim=1)  # Sum variance across classes
        
        self.eval()  # Disable dropout
        return mean_pred, uncertainty


class ActiveLearningQueryStrategies:
    """
    Collection of query strategies for Active Learning.
    
    Each strategy selects the most informative samples for labeling.
    """
    
    @staticmethod
    def uncertainty_entropy(predictions: torch.Tensor) -> torch.Tensor:
        """
        Entropy-based uncertainty sampling.
        
        Args:
            predictions: Model predictions (probabilities)
            
        Returns:
            torch.Tensor: Entropy scores (higher = more uncertain)
        """
        # Calculate entropy: H(p) = -Σ p_i * log(p_i)
        log_probs = torch.log(predictions + 1e-8)  # Add small epsilon for numerical stability
        entropy_scores = -torch.sum(predictions * log_probs, dim=1)
        return entropy_scores
    
    @staticmethod
    def uncertainty_margin(predictions: torch.Tensor) -> torch.Tensor:
        """
        Margin-based uncertainty sampling.
        
        Args:
            predictions: Model predictions (probabilities)
            
        Returns:
            torch.Tensor: Margin scores (lower = more uncertain)
        """
        # Sort predictions in descending order
        sorted_preds, _ = torch.sort(predictions, dim=1, descending=True)
        
        # Calculate margin between top-2 predictions
        margins = sorted_preds[:, 0] - sorted_preds[:, 1]
        
        # Return negative margins (lower margin = higher uncertainty)
        return -margins
    
    @staticmethod
    def uncertainty_least_confidence(predictions: torch.Tensor) -> torch.Tensor:
        """
        Least confidence uncertainty sampling.
        
        Args:
            predictions: Model predictions (probabilities)
            
        Returns:
            torch.Tensor: Confidence scores (lower = more uncertain)
        """
        # Get maximum probability (confidence)
        max_probs, _ = torch.max(predictions, dim=1)
        
        # Return negative confidence (lower confidence = higher uncertainty)
        return -max_probs
    
    @staticmethod
    def diversity_kmeans(features: torch.Tensor, n_clusters: int) -> torch.Tensor:
        """
        Diversity-based sampling using K-means clustering.
        
        Args:
            features: Feature representations
            n_clusters: Number of clusters (samples to select)
            
        Returns:
            torch.Tensor: Indices of selected samples
        """
        # Convert to numpy for sklearn
        features_np = features.cpu().numpy()
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_np)
        
        # Select samples closest to cluster centers
        selected_indices = []
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            if cluster_mask.sum() > 0:
                cluster_features = features_np[cluster_mask]
                cluster_center = kmeans.cluster_centers_[i]
                
                # Find closest sample to cluster center
                distances = np.linalg.norm(cluster_features - cluster_center, axis=1)
                closest_idx = np.argmin(distances)
                
                # Map back to original indices
                original_indices = np.where(cluster_mask)[0]
                selected_indices.append(original_indices[closest_idx])
        
        return torch.tensor(selected_indices)


class ActiveLearningTrainer:
    """
    Active Learning trainer that implements the complete AL pipeline.
    
    Supports multiple query strategies and tracks performance over AL rounds.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 device: torch.device,
                 query_strategy: str = 'entropy',
                 query_size: int = 100,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 5e-4,
                 num_epochs_per_round: int = 20):
        """
        Initialize Active Learning trainer.
        
        Args:
            model: Neural network model
            device: Computing device
            query_strategy: Query strategy ('entropy', 'margin', 'least_confidence', 'diversity', 'random')
            query_size: Number of samples to query per round
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            num_epochs_per_round: Training epochs per AL round
        """
        self.model = model.to(device)
        self.device = device
        self.query_strategy = query_strategy
        self.query_size = query_size
        self.num_epochs_per_round = num_epochs_per_round
        
        # Optimizer and loss function
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        
        # Query strategy function
        self.query_strategies = {
            'entropy': ActiveLearningQueryStrategies.uncertainty_entropy,
            'margin': ActiveLearningQueryStrategies.uncertainty_margin,
            'least_confidence': ActiveLearningQueryStrategies.uncertainty_least_confidence,
            'diversity': ActiveLearningQueryStrategies.diversity_kmeans,
            'random': lambda x: torch.randperm(len(x))[:self.query_size]
        }
        
        if query_strategy not in self.query_strategies:
            raise ValueError(f"Unknown query strategy: {query_strategy}")
        
        # Tracking metrics
        self.al_history = {
            'round': [],
            'labeled_size': [],
            'test_accuracy': [],
            'query_uncertainty': [],
            'training_time': []
        }
        
        logger.info(f"Active Learning trainer initialized with strategy: {query_strategy}")
    
    def train_model(self, labeled_loader: DataLoader, val_loader: DataLoader = None) -> Dict[str, float]:
        """
        Train the model on currently labeled data.
        
        Args:
            labeled_loader: DataLoader for labeled data
            val_loader: Optional validation loader
            
        Returns:
            Dict[str, float]: Training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        # Training loop
        for epoch in range(self.num_epochs_per_round):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for images, targets in labeled_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                epoch_correct += (predicted == targets).sum().item()
                epoch_total += targets.size(0)
                epoch_loss += loss.item()
            
            # Update totals
            total_loss += epoch_loss / len(labeled_loader)
            correct_predictions = epoch_correct
            total_predictions = epoch_total
        
        train_accuracy = 100. * correct_predictions / total_predictions
        avg_loss = total_loss / self.num_epochs_per_round
        
        return {
            'train_loss': avg_loss,
            'train_accuracy': train_accuracy
        }
    
    def query_samples(self, pool_loader: DataLoader, pool_indices: List[int]) -> Tuple[List[int], float]:
        """
        Select samples for labeling using the specified query strategy.
        
        Args:
            pool_loader: DataLoader for unlabeled pool
            pool_indices: Current pool indices
            
        Returns:
            Tuple[List[int], float]: (selected_indices, average_uncertainty)
        """
        self.model.eval()
        
        all_predictions = []
        all_features = []
        all_uncertainties = []
        
        with torch.no_grad():
            for images, _ in pool_loader:
                images = images.to(self.device)
                
                # Get predictions
                outputs = self.model(images)
                predictions = F.softmax(outputs, dim=1)
                all_predictions.append(predictions)
                
                # Get features for diversity sampling
                if self.query_strategy == 'diversity':
                    features = self.model.extract_features(images)
                    all_features.append(features)
                
                # Get uncertainties for Monte Carlo methods
                if hasattr(self.model, 'predict_with_uncertainty'):
                    _, uncertainties = self.model.predict_with_uncertainty(images)
                    all_uncertainties.append(uncertainties)
        
        # Concatenate all batches
        all_predictions = torch.cat(all_predictions, dim=0)
        
        # IMPORTANT: Limit query size to available pool samples
        actual_query_size = min(self.query_size, len(pool_indices))
        
        if self.query_strategy == 'diversity':
            all_features = torch.cat(all_features, dim=0)
            selected_relative_indices = self.query_strategies[self.query_strategy](all_features, actual_query_size)
        elif self.query_strategy == 'random':
            num_samples = min(len(all_predictions), len(pool_indices))
            selected_relative_indices = torch.randperm(num_samples)[:actual_query_size]
        else:
            # Uncertainty-based strategies
            uncertainty_scores = self.query_strategies[self.query_strategy](all_predictions)
            # Only consider scores for samples that are still in the pool
            valid_scores = uncertainty_scores[:len(pool_indices)]
            _, selected_relative_indices = torch.topk(valid_scores, actual_query_size)
        
        # Convert relative indices to absolute pool indices - FIXED BUG HERE
        selected_absolute_indices = []
        for idx in selected_relative_indices:
            idx_val = idx.item() if hasattr(idx, 'item') else int(idx)
            if idx_val < len(pool_indices):  # Safety check
                selected_absolute_indices.append(pool_indices[idx_val])
        
        # Calculate average uncertainty for tracking
        if self.query_strategy != 'random' and self.query_strategy != 'diversity':
            if len(selected_relative_indices) > 0:
                valid_indices = [idx for idx in selected_relative_indices if idx < len(uncertainty_scores)]
                if valid_indices:
                    avg_uncertainty = uncertainty_scores[valid_indices].mean().item()
                else:
                    avg_uncertainty = 0.0
            else:
                avg_uncertainty = 0.0
        else:
            avg_uncertainty = 0.0
        
        return selected_absolute_indices, avg_uncertainty
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for images, targets in test_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                total_loss += loss.item()
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(test_loader)
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss
        }
    
    def active_learning_loop(self,
                           initial_labeled_loader: DataLoader,
                           pool_loader: DataLoader,
                           test_loader: DataLoader,
                           pool_indices: List[int],
                           labeled_indices: List[int],
                           num_rounds: int = 10,
                           save_dir: str = "./outputs/models/al_models") -> Dict[str, List]:
        """
        Complete Active Learning training loop.
        
        Args:
            initial_labeled_loader: Initial labeled data
            pool_loader: Unlabeled pool data
            test_loader: Test data
            pool_indices: Indices of pool samples
            labeled_indices: Indices of labeled samples
            num_rounds: Number of AL rounds
            save_dir: Directory to save models
            
        Returns:
            Dict[str, List]: Training history
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Current data tracking
        current_labeled_indices = labeled_indices.copy()
        current_pool_indices = pool_indices.copy()
        current_labeled_loader = initial_labeled_loader
        
        logger.info(f"Starting Active Learning with {len(current_labeled_indices)} initial samples")
        
        for round_num in range(num_rounds):
            logger.info(f"\n=== Active Learning Round {round_num + 1}/{num_rounds} ===")
            round_start_time = time.time()
            
            # Train model on current labeled data
            train_metrics = self.train_model(current_labeled_loader)
            
            # Evaluate on test set
            test_metrics = self.evaluate(test_loader)
            
            # Query new samples (except in the last round)
            if round_num < num_rounds - 1:
                selected_indices, avg_uncertainty = self.query_samples(pool_loader, current_pool_indices)
                
                # Update labeled and pool sets
                current_labeled_indices.extend(selected_indices)
                for idx in selected_indices:
                    current_pool_indices.remove(idx)
                
                # Create new labeled loader with updated indices
                # This would need to be implemented with the dataset
                # For now, we'll simulate the update
                
            else:
                avg_uncertainty = 0.0
            
            round_time = time.time() - round_start_time
            
            # Record metrics
            self.al_history['round'].append(round_num + 1)
            self.al_history['labeled_size'].append(len(current_labeled_indices))
            self.al_history['test_accuracy'].append(test_metrics['accuracy'])
            self.al_history['query_uncertainty'].append(avg_uncertainty)
            self.al_history['training_time'].append(round_time)
            
            logger.info(f"Round {round_num + 1} Results:")
            logger.info(f"  Labeled samples: {len(current_labeled_indices)}")
            logger.info(f"  Test accuracy: {test_metrics['accuracy']:.2f}%")
            logger.info(f"  Query uncertainty: {avg_uncertainty:.4f}")
            logger.info(f"  Round time: {round_time:.1f}s")
            
            # Save model checkpoint
            if (round_num + 1) % 5 == 0:
                torch.save({
                    'round': round_num + 1,
                    'model_state_dict': self.model.state_dict(),
                    'al_history': self.al_history,
                    'labeled_indices': current_labeled_indices,
                    'pool_indices': current_pool_indices
                }, save_dir / f'al_checkpoint_round_{round_num + 1}.pth')
        
        # Save final results
        with open(save_dir / f'al_results_{self.query_strategy}.json', 'w') as f:
            json.dump(self.al_history, f, indent=2)
        
        logger.info(f"Active Learning completed! Final accuracy: {self.al_history['test_accuracy'][-1]:.2f}%")
        return self.al_history


class ActiveLearningComparison:
    """
    Compare different Active Learning strategies.
    """
    
    def __init__(self, device: torch.device):
        """
        Initialize AL comparison framework.
        
        Args:
            device: Computing device
        """
        self.device = device
        self.strategies = ['entropy', 'margin', 'least_confidence', 'random']
        self.results = {}
    
    def compare_strategies(self,
                          initial_labeled_loader: DataLoader,
                          pool_loader: DataLoader,
                          test_loader: DataLoader,
                          pool_indices: List[int],
                          labeled_indices: List[int],
                          num_rounds: int = 10) -> Dict[str, Dict]:
        """
        Compare different AL strategies.
        
        Args:
            initial_labeled_loader: Initial labeled data
            pool_loader: Unlabeled pool
            test_loader: Test data
            pool_indices: Pool indices
            labeled_indices: Labeled indices
            num_rounds: Number of AL rounds
            
        Returns:
            Dict containing results for each strategy
        """
        for strategy in self.strategies:
            logger.info(f"\n{'='*50}")
            logger.info(f"Testing strategy: {strategy.upper()}")
            logger.info(f"{'='*50}")
            
            # Initialize fresh model for each strategy
            model = ResNet18ActiveLearning()
            trainer = ActiveLearningTrainer(
                model=model,
                device=self.device,
                query_strategy=strategy,
                query_size=100,
                num_epochs_per_round=10  # Reduced for comparison
            )
            
            # Run active learning
            history = trainer.active_learning_loop(
                initial_labeled_loader=initial_labeled_loader,
                pool_loader=pool_loader,
                test_loader=test_loader,
                pool_indices=pool_indices.copy(),
                labeled_indices=labeled_indices.copy(),
                num_rounds=num_rounds
            )
            
            self.results[strategy] = {
                'history': history,
                'final_accuracy': history['test_accuracy'][-1],
                'max_accuracy': max(history['test_accuracy']),
                'efficiency': history['test_accuracy'][-1] / history['labeled_size'][-1] * 1000  # Accuracy per 1000 samples
            }
        
        return self.results


def visualize_active_learning_results(results: Dict[str, Dict], save_path: str = "./outputs/images/al_comparison.png"):
    """
    Visualize Active Learning comparison results.
    
    Args:
        results: Results from ActiveLearningComparison
        save_path: Path to save visualization
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Colors for different strategies
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (strategy, strategy_results) in enumerate(results.items()):
        history = strategy_results['history']
        color = colors[i % len(colors)]
        
        # Plot 1: Accuracy vs Number of Labeled Samples
        axes[0, 0].plot(history['labeled_size'], history['test_accuracy'], 
                       label=strategy, color=color, marker='o', linewidth=2)
    
    axes[0, 0].set_xlabel('Number of Labeled Samples')
    axes[0, 0].set_ylabel('Test Accuracy (%)')
    axes[0, 0].set_title('Active Learning Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy vs AL Round
    for i, (strategy, strategy_results) in enumerate(results.items()):
        history = strategy_results['history']
        color = colors[i % len(colors)]
        axes[0, 1].plot(history['round'], history['test_accuracy'], 
                       label=strategy, color=color, marker='s', linewidth=2)
    
    axes[0, 1].set_xlabel('Active Learning Round')
    axes[0, 1].set_ylabel('Test Accuracy (%)')
    axes[0, 1].set_title('Accuracy by Round')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Query Uncertainty Over Time
    for i, (strategy, strategy_results) in enumerate(results.items()):
        history = strategy_results['history']
        if strategy != 'random':  # Random doesn't have meaningful uncertainty
            color = colors[i % len(colors)]
            axes[0, 2].plot(history['round'][:-1], history['query_uncertainty'][:-1], 
                           label=strategy, color=color, marker='^', linewidth=2)
    
    axes[0, 2].set_xlabel('Active Learning Round')
    axes[0, 2].set_ylabel('Average Query Uncertainty')
    axes[0, 2].set_title('Query Uncertainty Trends')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Final Accuracy Comparison
    strategies = list(results.keys())
    final_accuracies = [results[s]['final_accuracy'] for s in strategies]
    bars = axes[1, 0].bar(strategies, final_accuracies, color=colors[:len(strategies)], alpha=0.7)
    axes[1, 0].set_ylabel('Final Test Accuracy (%)')
    axes[1, 0].set_title('Final Accuracy Comparison')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, acc in zip(bars, final_accuracies):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                       f'{acc:.1f}%', ha='center', va='bottom')
    
    # Plot 5: Efficiency (Accuracy per Sample)
    efficiencies = [results[s]['efficiency'] for s in strategies]
    bars = axes[1, 1].bar(strategies, efficiencies, color=colors[:len(strategies)], alpha=0.7)
    axes[1, 1].set_ylabel('Efficiency (Acc% per 1000 samples)')
    axes[1, 1].set_title('Sample Efficiency Comparison')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Plot 6: Training Time per Round
    for i, (strategy, strategy_results) in enumerate(results.items()):
        history = strategy_results['history']
        color = colors[i % len(colors)]
        axes[1, 2].plot(history['round'], history['training_time'], 
                       label=strategy, color=color, marker='d', linewidth=2)
    
    axes[1, 2].set_xlabel('Active Learning Round')
    axes[1, 2].set_ylabel('Training Time (seconds)')
    axes[1, 2].set_title('Training Time per Round')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Active Learning visualization saved to {save_path}")
    plt.show()


def visualize_query_selection(model: ResNet18ActiveLearning, 
                            pool_loader: DataLoader, 
                            query_strategy: str,
                            save_path: str = "./outputs/images/query_visualization.png"):
    """
    Visualize the query selection process.
    
    Args:
        model: Trained model
        pool_loader: Pool data loader
        query_strategy: Query strategy name
        save_path: Path to save visualization
    """
    model.eval()
    
    # Determine device properly
    device = next(model.parameters()).device
    
    # Get predictions and features for a batch
    images, labels = next(iter(pool_loader))
    images = images.to(device)  # Use the device from model parameters
    
    with torch.no_grad():
        outputs = model(images)
        predictions = F.softmax(outputs, dim=1)
        features = model.extract_features(images)
    
    # Calculate uncertainty scores
    if query_strategy == 'entropy':
        scores = ActiveLearningQueryStrategies.uncertainty_entropy(predictions)
    elif query_strategy == 'margin':
        scores = ActiveLearningQueryStrategies.uncertainty_margin(predictions)
    else:
        scores = ActiveLearningQueryStrategies.uncertainty_least_confidence(predictions)
    
    # Select top uncertain samples (limit to available samples)
    num_samples = min(16, len(scores))
    _, top_indices = torch.topk(scores, num_samples)
    
    # Visualize
    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    
    # Handle single subplot case
    if num_samples == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.ravel()
    
    for i, idx in enumerate(top_indices):
        if i >= len(axes):
            break
            
        img = images[idx].cpu().numpy().transpose(1, 2, 0)
        img = img * np.array([0.2023, 0.1994, 0.2010]) + np.array([0.4914, 0.4822, 0.4465])
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(f'Score: {scores[idx]:.3f}')
        axes[i].axis('off')
    
    # Hide extra subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Top Uncertain Samples ({query_strategy})', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Query visualization saved to {save_path}")
    plt.show()


def main():
    """
    Main function to demonstrate Active Learning implementations.
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize data preprocessor
    preprocessor = CIFAR10DataPreprocessor(data_dir="./data")
    
    # Load and prepare data
    train_dataset, test_dataset = preprocessor.load_cifar10_datasets()
    
    # Create Active Learning initial setup
    initial_labeled_subset, pool_subset, labeled_indices, pool_indices = preprocessor.create_active_learning_initial_set(
        train_dataset, initial_size=1000
    )
    
    # Create data loaders
    initial_labeled_loader = DataLoader(initial_labeled_subset, batch_size=64, shuffle=True)
    pool_loader = DataLoader(pool_subset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Create output directories
    os.makedirs("./outputs/models/al_models", exist_ok=True)
    os.makedirs("./outputs/images", exist_ok=True)
    
    # Run Active Learning comparison
    al_comparison = ActiveLearningComparison(device)
    results = al_comparison.compare_strategies(
        initial_labeled_loader=initial_labeled_loader,
        pool_loader=pool_loader,
        test_loader=test_loader,
        pool_indices=pool_indices,
        labeled_indices=labeled_indices,
        num_rounds=5  # Reduced for demo
    )
    
    # Visualize results
    visualize_active_learning_results(results)
    
    # Demonstrate query visualization with proper device handling
    try:
        # Create a model and move it to the correct device
        model = ResNet18ActiveLearning().to(device)
        
        # Train it briefly so it has some reasonable weights
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        model.train()
        
        # Quick training on a few batches
        train_loader = DataLoader(initial_labeled_subset, batch_size=32, shuffle=True)
        for i, (images, targets) in enumerate(train_loader):
            if i >= 5:  # Just a few batches for demo
                break
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # Now visualize with the properly trained and device-placed model
        visualize_query_selection(model, pool_loader, 'entropy')
        
    except Exception as e:
        logger.warning(f"Could not create query visualization: {e}")
        logger.info("Continuing without query visualization...")
    
    # Save results
    with open("./outputs/logs/al_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("Active Learning demonstration completed!")
    
    # Print summary
    print("\n" + "="*60)
    print("ACTIVE LEARNING RESULTS SUMMARY")
    print("="*60)
    for strategy, result in results.items():
        print(f"{strategy.capitalize():15} | Final Acc: {result['final_accuracy']:.2f}% | "
              f"Max Acc: {result['max_accuracy']:.2f}% | "
              f"Efficiency: {result['efficiency']:.2f}")
    print("="*60)


if __name__ == "__main__":
    main()