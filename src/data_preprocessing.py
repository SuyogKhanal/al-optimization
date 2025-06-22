"""
CIFAR-10 Data Preprocessing Module for Efficient Image Classification Research

This module provides comprehensive data preprocessing utilities for comparing
Semi-Supervised Learning, Active Learning, and GAN-based Data Augmentation
techniques on the CIFAR-10 dataset.

Author: Research Team
Date: 2025
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict, Optional
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CIFAR10DataPreprocessor:
    """
    Comprehensive CIFAR-10 data preprocessing class for research experiments.
    
    This class handles:
    1. Dataset downloading and loading
    2. Data normalization and augmentation
    3. Creating labeled/unlabeled splits for SSL
    4. Preparing data for Active Learning scenarios
    5. Data preparation for GAN training
    """
    
    def __init__(self, data_dir: str = "./data", seed: int = 42):
        """
        Initialize the data preprocessor.
        
        Args:
            data_dir (str): Directory to store/load CIFAR-10 data
            seed (int): Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        
        # Set random seeds for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # CIFAR-10 class names
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        
        # Dataset statistics for normalization
        self.cifar10_mean = (0.4914, 0.4822, 0.4465)
        self.cifar10_std = (0.2023, 0.1994, 0.2010)
        
        logger.info(f"CIFAR10DataPreprocessor initialized with data_dir: {self.data_dir}")
    
    def get_base_transforms(self, train: bool = True) -> transforms.Compose:
        """
        Get base data transformations for CIFAR-10.
        
        Args:
            train (bool): Whether to apply training augmentations
            
        Returns:
            transforms.Compose: Composition of transformations
        """
        if train:
            # Training transforms with data augmentation
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(self.cifar10_mean, self.cifar10_std)
            ])
        else:
            # Test transforms - no augmentation
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.cifar10_mean, self.cifar10_std)
            ])
        
        return transform
    
    def get_ssl_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        """
        Get transformations for Semi-Supervised Learning (weak and strong augmentations).
        
        Returns:
            Tuple[transforms.Compose, transforms.Compose]: (weak_transform, strong_transform)
        """
        # Weak augmentation (similar to base training transforms)
        weak_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(self.cifar10_mean, self.cifar10_std)
        ])
        
        # Strong augmentation for SSL consistency training
        strong_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(self.cifar10_mean, self.cifar10_std)
        ])
        
        return weak_transform, strong_transform
    
    def load_cifar10_datasets(self) -> Tuple[torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR10]:
        """
        Load CIFAR-10 training and test datasets.
        
        Returns:
            Tuple[CIFAR10, CIFAR10]: (train_dataset, test_dataset)
        """
        logger.info("Loading CIFAR-10 datasets...")
        
        # Load training data with base transforms
        train_transform = self.get_base_transforms(train=True)
        train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, 
            train=True, 
            download=True, 
            transform=train_transform
        )
        
        # Load test data
        test_transform = self.get_base_transforms(train=False)
        test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, 
            train=False, 
            download=True, 
            transform=test_transform
        )
        
        logger.info(f"Loaded {len(train_dataset)} training samples and {len(test_dataset)} test samples")
        return train_dataset, test_dataset
    
    def create_ssl_splits(self, 
                         train_dataset: torchvision.datasets.CIFAR10, 
                         labeled_ratio: float = 0.1,
                         validation_ratio: float = 0.1) -> Tuple[Subset, Subset, Subset]:
        """
        Create labeled/unlabeled splits for Semi-Supervised Learning.
        
        Args:
            train_dataset: Full CIFAR-10 training dataset
            labeled_ratio: Ratio of data to use as labeled (default: 10%)
            validation_ratio: Ratio of labeled data to use for validation
            
        Returns:
            Tuple[Subset, Subset, Subset]: (labeled_subset, unlabeled_subset, validation_subset)
        """
        logger.info(f"Creating SSL splits with {labeled_ratio*100}% labeled data...")
        
        # Get all training indices
        total_samples = len(train_dataset)
        all_indices = list(range(total_samples))
        
        # Stratified split to maintain class balance
        train_labels = [train_dataset[i][1] for i in range(total_samples)]
        
        # Split into labeled and unlabeled
        labeled_indices, unlabeled_indices = train_test_split(
            all_indices, 
            test_size=1-labeled_ratio,
            stratify=train_labels,
            random_state=self.seed
        )
        
        # Further split labeled data into train/validation
        labeled_labels = [train_labels[i] for i in labeled_indices]
        train_labeled_indices, val_labeled_indices = train_test_split(
            labeled_indices,
            test_size=validation_ratio,
            stratify=labeled_labels,
            random_state=self.seed
        )
        
        # Create subsets
        labeled_subset = Subset(train_dataset, train_labeled_indices)
        unlabeled_subset = Subset(train_dataset, unlabeled_indices)
        validation_subset = Subset(train_dataset, val_labeled_indices)
        
        logger.info(f"SSL splits created: "
                   f"Labeled: {len(labeled_subset)}, "
                   f"Unlabeled: {len(unlabeled_subset)}, "
                   f"Validation: {len(validation_subset)}")
        
        return labeled_subset, unlabeled_subset, validation_subset
    
    def create_active_learning_initial_set(self, 
                                         train_dataset: torchvision.datasets.CIFAR10,
                                         initial_size: int = 1000) -> Tuple[Subset, Subset, List[int], List[int]]:
        """
        Create initial labeled set for Active Learning experiments.
        
        Args:
            train_dataset: Full CIFAR-10 training dataset
            initial_size: Size of initial labeled set
            
        Returns:
            Tuple[Subset, Subset, List[int], List[int]]: 
                (initial_labeled_subset, pool_subset, labeled_indices, pool_indices)
        """
        logger.info(f"Creating Active Learning initial set with {initial_size} samples...")
        
        # Get all training indices and labels
        total_samples = len(train_dataset)
        all_indices = list(range(total_samples))
        train_labels = [train_dataset[i][1] for i in range(total_samples)]
        
        # Stratified sampling for initial labeled set
        initial_indices, pool_indices = train_test_split(
            all_indices,
            train_size=initial_size,
            stratify=train_labels,
            random_state=self.seed
        )
        
        # Create subsets
        initial_labeled_subset = Subset(train_dataset, initial_indices)
        pool_subset = Subset(train_dataset, pool_indices)
        
        logger.info(f"Active Learning sets created: "
                   f"Initial labeled: {len(initial_labeled_subset)}, "
                   f"Pool: {len(pool_subset)}")
        
        return initial_labeled_subset, pool_subset, initial_indices, pool_indices
    
    def prepare_gan_training_data(self, train_dataset: torchvision.datasets.CIFAR10) -> DataLoader:
        """
        Prepare data for GAN training with specific transformations.
        
        Args:
            train_dataset: CIFAR-10 training dataset
            
        Returns:
            DataLoader: DataLoader optimized for GAN training
        """
        logger.info("Preparing data for GAN training...")
        
        # GAN-specific transforms (minimal preprocessing)
        gan_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])
        
        # Create dataset with GAN transforms
        gan_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, 
            train=True, 
            download=False, 
            transform=gan_transform
        )
        
        # Create DataLoader
        gan_dataloader = DataLoader(
            gan_dataset,
            batch_size=128,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        logger.info(f"GAN training dataloader created with {len(gan_dataset)} samples")
        return gan_dataloader
    
    def visualize_data_splits(self, 
                            labeled_subset: Subset, 
                            unlabeled_subset: Subset,
                            save_path: Optional[str] = None) -> None:
        """
        Visualize the distribution of classes in labeled/unlabeled splits.
        
        Args:
            labeled_subset: Labeled data subset
            unlabeled_subset: Unlabeled data subset
            save_path: Path to save the visualization
        """
        # Count classes in labeled subset
        labeled_classes = [labeled_subset.dataset[idx][1] for idx in labeled_subset.indices]
        labeled_counts = np.bincount(labeled_classes, minlength=10)
        
        # Count classes in unlabeled subset
        unlabeled_classes = [unlabeled_subset.dataset[idx][1] for idx in unlabeled_subset.indices]
        unlabeled_counts = np.bincount(unlabeled_classes, minlength=10)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Labeled distribution
        ax1.bar(range(10), labeled_counts, color='skyblue', alpha=0.7)
        ax1.set_title('Labeled Data Distribution')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Number of Samples')
        ax1.set_xticks(range(10))
        ax1.set_xticklabels(self.class_names, rotation=45)
        
        # Unlabeled distribution
        ax2.bar(range(10), unlabeled_counts, color='lightcoral', alpha=0.7)
        ax2.set_title('Unlabeled Data Distribution')
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Number of Samples')
        ax2.set_xticks(range(10))
        ax2.set_xticklabels(self.class_names, rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Data distribution visualization saved to {save_path}")
        
        plt.show()
    
    def visualize_sample_images(self, dataset: Dataset, num_samples: int = 16, save_path: Optional[str] = None) -> None:
        """
        Visualize sample images from the dataset.
        
        Args:
            dataset: Dataset to visualize
            num_samples: Number of samples to display
            save_path: Path to save the visualization
        """
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.ravel()
        
        # Get random samples
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        
        for i, idx in enumerate(indices):
            image, label = dataset[idx]
            
            # Denormalize image for visualization
            if isinstance(image, torch.Tensor):
                image = image.numpy().transpose(1, 2, 0)
                image = image * np.array(self.cifar10_std) + np.array(self.cifar10_mean)
                image = np.clip(image, 0, 1)
            
            axes[i].imshow(image)
            axes[i].set_title(f'{self.class_names[label]}')
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Sample images visualization saved to {save_path}")
        
        plt.show()
    
    def save_splits_metadata(self, 
                           labeled_indices: List[int], 
                           unlabeled_indices: List[int],
                           save_dir: str) -> None:
        """
        Save data split metadata for reproducibility.
        
        Args:
            labeled_indices: Indices of labeled samples
            unlabeled_indices: Indices of unlabeled samples
            save_dir: Directory to save metadata
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            'labeled_indices': labeled_indices,
            'unlabeled_indices': unlabeled_indices,
            'seed': self.seed,
            'total_samples': len(labeled_indices) + len(unlabeled_indices)
        }
        
        metadata_path = save_dir / 'data_splits_metadata.pkl'
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Data splits metadata saved to {metadata_path}")
    
    def get_dataloaders(self, 
                       labeled_subset: Subset,
                       unlabeled_subset: Optional[Subset] = None,
                       test_dataset: Optional[Dataset] = None,
                       batch_size: int = 64,
                       num_workers: int = 2) -> Dict[str, DataLoader]:
        """
        Create DataLoaders for different dataset splits.
        
        Args:
            labeled_subset: Labeled training data
            unlabeled_subset: Unlabeled training data (optional)
            test_dataset: Test dataset (optional)
            batch_size: Batch size for DataLoaders
            num_workers: Number of worker processes
            
        Returns:
            Dict[str, DataLoader]: Dictionary of DataLoaders
        """
        dataloaders = {}
        
        # Labeled data loader
        dataloaders['labeled'] = DataLoader(
            labeled_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        # Unlabeled data loader (if provided)
        if unlabeled_subset is not None:
            dataloaders['unlabeled'] = DataLoader(
                unlabeled_subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True
            )
        
        # Test data loader (if provided)
        if test_dataset is not None:
            dataloaders['test'] = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
        
        logger.info(f"Created {len(dataloaders)} DataLoaders with batch_size={batch_size}")
        return dataloaders


def main():
    """
    Example usage of the CIFAR10DataPreprocessor class.
    This demonstrates how to use the preprocessor for different learning scenarios.
    """
    # Initialize preprocessor
    preprocessor = CIFAR10DataPreprocessor(data_dir="./data")
    
    # Load datasets
    train_dataset, test_dataset = preprocessor.load_cifar10_datasets()
    
    # Create SSL splits
    labeled_subset, unlabeled_subset, validation_subset = preprocessor.create_ssl_splits(
        train_dataset, labeled_ratio=0.1
    )
    
    # Create Active Learning initial set
    al_labeled, al_pool, al_labeled_indices, al_pool_indices = preprocessor.create_active_learning_initial_set(
        train_dataset, initial_size=1000
    )
    
    # Prepare GAN training data
    gan_dataloader = preprocessor.prepare_gan_training_data(train_dataset)
    
    # Get DataLoaders for SSL
    ssl_dataloaders = preprocessor.get_dataloaders(
        labeled_subset=labeled_subset,
        unlabeled_subset=unlabeled_subset,
        test_dataset=test_dataset
    )
    
    # Visualize data splits
    os.makedirs("./outputs/images", exist_ok=True)
    preprocessor.visualize_data_splits(
        labeled_subset, 
        unlabeled_subset, 
        save_path="./outputs/images/ssl_data_distribution.png"
    )
    
    # Visualize sample images
    preprocessor.visualize_sample_images(
        train_dataset, 
        save_path="./outputs/images/sample_cifar10_images.png"
    )
    
    # Save metadata
    os.makedirs("./outputs/logs", exist_ok=True)
    preprocessor.save_splits_metadata(
        labeled_subset.indices,
        unlabeled_subset.indices,
        "./outputs/logs"
    )
    
    print("Data preprocessing completed successfully!")
    print(f"SSL - Labeled: {len(labeled_subset)}, Unlabeled: {len(unlabeled_subset)}")
    print(f"AL - Initial: {len(al_labeled)}, Pool: {len(al_pool)}")
    print(f"GAN - Training batches: {len(gan_dataloader)}")


if __name__ == "__main__":
    main()