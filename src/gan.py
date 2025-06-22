"""
GAN-based Data Augmentation Implementation for CIFAR-10

This module implements Generative Adversarial Networks for data augmentation:
1. DCGAN (Deep Convolutional GAN)
2. WGAN-GP (Wasserstein GAN with Gradient Penalty)
3. Conditional GAN for class-specific generation
4. Quality assessment using FID and Inception Score

Mathematical Foundation:
- GAN Objective: min_G max_D V(D,G) = E[log D(x)] + E[log(1-D(G(z)))]
- WGAN-GP: min_G max_D E[D(x)] - E[D(G(z))] + λ*E[(||∇D(x̂)||₂ - 1)²]
- Conditional GAN: Generate class-specific samples using label information

Author: Research Team
Date: 2025
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from typing import Tuple, Dict, List, Optional, Union
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score, classification_report
from scipy.linalg import sqrtm
from scipy.stats import entropy
import seaborn as sns

# Import our modules
from data_preprocessing import CIFAR10DataPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DCGANGenerator(nn.Module):
    """
    Deep Convolutional GAN Generator for CIFAR-10.
    
    Architecture based on DCGAN paper with modifications for 32x32 images.
    Generates high-quality CIFAR-10 images from random noise.
    """
    
    def __init__(self, 
                 latent_dim: int = 100, 
                 num_classes: int = 10, 
                 channels: int = 3,
                 feature_maps: int = 64,
                 conditional: bool = False):
        """
        Initialize DCGAN Generator.
        
        Args:
            latent_dim: Dimension of latent noise vector
            num_classes: Number of classes for conditional generation
            channels: Number of output channels (3 for RGB)
            feature_maps: Base number of feature maps
            conditional: Whether to use conditional generation
        """
        super(DCGANGenerator, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.conditional = conditional
        self.feature_maps = feature_maps
        
        # Input dimension (latent + class embedding if conditional)
        input_dim = latent_dim + num_classes if conditional else latent_dim
        
        # Initial dense layer and reshape
        self.initial = nn.Sequential(
            nn.Linear(input_dim, feature_maps * 8 * 4 * 4),
            nn.BatchNorm1d(feature_maps * 8 * 4 * 4),
            nn.ReLU(True)
        )
        
        # Transposed convolution layers
        self.conv_layers = nn.Sequential(
            # State size: (feature_maps*8) x 4 x 4
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            # State size: (feature_maps*4) x 8 x 8
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            # State size: (feature_maps*2) x 16 x 16
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            # State size: (feature_maps) x 32 x 32
            nn.ConvTranspose2d(feature_maps, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # Final state size: (channels) x 32 x 32
        )
        
        # Weight initialization
        self.apply(self._weights_init)
    
    def _weights_init(self, m):
        """Initialize network weights."""
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    def forward(self, noise: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through generator.
        
        Args:
            noise: Random noise tensor [batch_size, latent_dim]
            labels: Class labels for conditional generation [batch_size]
            
        Returns:
            torch.Tensor: Generated images [batch_size, channels, 32, 32]
        """
        batch_size = noise.size(0)
        
        if self.conditional and labels is not None:
            # One-hot encode labels
            labels_onehot = F.one_hot(labels, self.num_classes).float()
            # Concatenate noise and labels
            gen_input = torch.cat([noise, labels_onehot], dim=1)
        else:
            gen_input = noise
        
        # Initial dense layer
        x = self.initial(gen_input)
        
        # Reshape to feature map format
        x = x.view(batch_size, self.feature_maps * 8, 4, 4)
        
        # Transposed convolutions
        x = self.conv_layers(x)
        
        return x


class DCGANDiscriminator(nn.Module):
    """
    Deep Convolutional GAN Discriminator for CIFAR-10.
    
    Discriminates between real and generated images with optional conditional input.
    """
    
    def __init__(self, 
                 channels: int = 3,
                 num_classes: int = 10,
                 feature_maps: int = 64,
                 conditional: bool = False):
        """
        Initialize DCGAN Discriminator.
        
        Args:
            channels: Number of input channels
            num_classes: Number of classes for conditional discrimination
            feature_maps: Base number of feature maps
            conditional: Whether to use conditional discrimination
        """
        super(DCGANDiscriminator, self).__init__()
        
        self.conditional = conditional
        self.num_classes = num_classes
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # Input size: (channels) x 32 x 32
            nn.Conv2d(channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (feature_maps) x 16 x 16
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (feature_maps*2) x 8 x 8
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (feature_maps*4) x 4 x 4
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (feature_maps*8) x 2 x 2
            nn.AdaptiveAvgPool2d(1)  # Force to 1x1 to avoid size mismatches
        )
        
        # Calculate the correct flattened size
        flattened_size = feature_maps * 8  # After adaptive avg pooling: (feature_maps*8) x 1 x 1
        
        # Output layers
        if conditional:
            # Conditional discriminator: predict real/fake + class
            self.fc_validity = nn.Linear(flattened_size, 1)
            self.fc_class = nn.Linear(flattened_size, num_classes)
        else:
            # Standard discriminator: predict real/fake only
            self.fc = nn.Sequential(
                nn.Linear(flattened_size, 1),
                nn.Sigmoid()
            )
        
        # Weight initialization
        self.apply(self._weights_init)
    
    def _weights_init(self, m):
        """Initialize network weights."""
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    def forward(self, images: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through discriminator.
        
        Args:
            images: Input images [batch_size, channels, 32, 32]
            
        Returns:
            torch.Tensor or Tuple: Validity scores (and class predictions if conditional)
        """
        # Convolutional layers
        x = self.conv_layers(images)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        if self.conditional:
            # Conditional discriminator
            validity = torch.sigmoid(self.fc_validity(x))
            class_pred = self.fc_class(x)
            return validity, class_pred
        else:
            # Standard discriminator
            validity = self.fc(x)
            return validity


class WGANGPDiscriminator(nn.Module):
    """
    WGAN-GP Discriminator (Critic) for improved training stability.
    
    Uses Wasserstein loss with gradient penalty for better convergence.
    """
    
    def __init__(self, channels: int = 3, feature_maps: int = 64):
        """
        Initialize WGAN-GP Discriminator.
        
        Args:
            channels: Number of input channels
            feature_maps: Base number of feature maps
        """
        super(WGANGPDiscriminator, self).__init__()
        
        self.model = nn.Sequential(
            # Input: (channels) x 32 x 32
            nn.Conv2d(channels, feature_maps, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (feature_maps) x 16 x 16
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1),
            nn.InstanceNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (feature_maps*2) x 8 x 8
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1),
            nn.InstanceNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (feature_maps*4) x 4 x 4
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1),
            nn.InstanceNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (feature_maps*8) x 2 x 2
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_maps * 8, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through critic."""
        return self.model(x)


class GANTrainer:
    """
    Comprehensive GAN trainer supporting multiple architectures and training strategies.
    
    Supports DCGAN, WGAN-GP, and conditional variants with advanced training techniques.
    """
    
    def __init__(self,
                 generator: nn.Module,
                 discriminator: nn.Module,
                 device: torch.device,
                 latent_dim: int = 100,
                 gan_type: str = 'dcgan',
                 learning_rate_g: float = 2e-4,
                 learning_rate_d: float = 2e-4,
                 beta1: float = 0.5,
                 beta2: float = 0.999,
                 gp_lambda: float = 10.0):
        """
        Initialize GAN trainer.
        
        Args:
            generator: Generator network
            discriminator: Discriminator/Critic network
            device: Computing device
            latent_dim: Latent noise dimension
            gan_type: Type of GAN ('dcgan', 'wgan-gp')
            learning_rate_g: Generator learning rate
            learning_rate_d: Discriminator learning rate
            beta1: Adam beta1 parameter
            beta2: Adam beta2 parameter
            gp_lambda: Gradient penalty coefficient for WGAN-GP
        """
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        self.latent_dim = latent_dim
        self.gan_type = gan_type
        self.gp_lambda = gp_lambda
        
        # Optimizers
        if gan_type == 'wgan-gp':
            # WGAN-GP uses RMSprop or Adam without momentum
            self.optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate_g, betas=(0, 0.9))
            self.optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate_d, betas=(0, 0.9))
        else:
            # Standard DCGAN optimizers
            self.optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate_g, betas=(beta1, beta2))
            self.optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate_d, betas=(beta1, beta2))
        
        # Loss functions
        self.criterion = nn.BCELoss()
        self.class_criterion = nn.CrossEntropyLoss()
        
        # Training metrics
        self.training_history = {
            'generator_loss': [],
            'discriminator_loss': [],
            'd_real_acc': [],
            'd_fake_acc': [],
            'gradient_penalty': []
        }
        
        logger.info(f"GAN trainer initialized: {gan_type.upper()}")
    
    def gradient_penalty(self, real_samples: torch.Tensor, fake_samples: torch.Tensor) -> torch.Tensor:
        """
        Calculate gradient penalty for WGAN-GP.
        
        Args:
            real_samples: Real image samples
            fake_samples: Generated image samples
            
        Returns:
            torch.Tensor: Gradient penalty loss
        """
        batch_size = real_samples.size(0)
        
        # Random interpolation factor
        epsilon = torch.rand(batch_size, 1, 1, 1).to(self.device)
        
        # Interpolated samples
        interpolated = epsilon * real_samples + (1 - epsilon) * fake_samples
        interpolated.requires_grad_(True)
        
        # Discriminator output for interpolated samples
        d_interpolated = self.discriminator(interpolated)
        
        # Gradients with respect to interpolated samples
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        
        return gradient_penalty
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train GAN for one epoch.
        
        Args:
            dataloader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Dict[str, float]: Training metrics for this epoch
        """
        self.generator.train()
        self.discriminator.train()
        
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_d_real_acc = 0.0
        epoch_d_fake_acc = 0.0
        epoch_gp = 0.0
        
        # Real and fake labels
        real_label = 1.0
        fake_label = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for i, (real_images, real_labels) in enumerate(progress_bar):
            batch_size = real_images.size(0)
            real_images = real_images.to(self.device)
            real_labels = real_labels.to(self.device)
            
            # === TRAIN DISCRIMINATOR ===
            self.optimizer_d.zero_grad()
            
            if self.gan_type == 'wgan-gp':
                # WGAN-GP discriminator training
                # Real images
                d_real = self.discriminator(real_images).view(-1)
                d_real_loss = -torch.mean(d_real)
                
                # Fake images
                noise = torch.randn(batch_size, self.latent_dim, device=self.device)
                if hasattr(self.generator, 'conditional') and self.generator.conditional:
                    fake_images = self.generator(noise, real_labels)
                else:
                    fake_images = self.generator(noise)
                
                d_fake = self.discriminator(fake_images.detach()).view(-1)
                d_fake_loss = torch.mean(d_fake)
                
                # Gradient penalty
                gp = self.gradient_penalty(real_images, fake_images)
                
                # Total discriminator loss
                d_loss = d_real_loss + d_fake_loss + self.gp_lambda * gp
                epoch_gp += gp.item()
                
                # Accuracy metrics (for WGAN, we use sign of outputs)
                d_real_acc = (d_real > 0).float().mean().item()
                d_fake_acc = (d_fake < 0).float().mean().item()
                
            else:
                # Standard DCGAN discriminator training
                # Real images
                if hasattr(self.discriminator, 'conditional') and self.discriminator.conditional:
                    d_real_output, d_real_class = self.discriminator(real_images)
                    real_labels_tensor = torch.full((batch_size,), real_label, device=self.device, dtype=torch.float)
                    d_real_loss = self.criterion(d_real_output.view(-1), real_labels_tensor)
                    d_real_class_loss = self.class_criterion(d_real_class, real_labels)
                    d_real_total = d_real_loss + d_real_class_loss
                else:
                    d_real_output = self.discriminator(real_images).view(-1)
                    real_labels_tensor = torch.full((batch_size,), real_label, device=self.device, dtype=torch.float)
                    d_real_total = self.criterion(d_real_output, real_labels_tensor)
                
                # Fake images
                noise = torch.randn(batch_size, self.latent_dim, device=self.device)
                if hasattr(self.generator, 'conditional') and self.generator.conditional:
                    fake_images = self.generator(noise, real_labels)
                else:
                    fake_images = self.generator(noise)
                
                if hasattr(self.discriminator, 'conditional') and self.discriminator.conditional:
                    d_fake_output, _ = self.discriminator(fake_images.detach())
                    fake_labels_tensor = torch.full((batch_size,), fake_label, device=self.device, dtype=torch.float)
                    d_fake_loss = self.criterion(d_fake_output.view(-1), fake_labels_tensor)
                else:
                    d_fake_output = self.discriminator(fake_images.detach()).view(-1)
                    fake_labels_tensor = torch.full((batch_size,), fake_label, device=self.device, dtype=torch.float)
                    d_fake_loss = self.criterion(d_fake_output, fake_labels_tensor)
                
                # Total discriminator loss
                d_loss = d_real_total + d_fake_loss
                
                # Accuracy metrics
                d_real_acc = (d_real_output > 0.5).float().mean().item()
                d_fake_acc = (d_fake_output < 0.5).float().mean().item()
            
            d_loss.backward()
            self.optimizer_d.step()
            
            # === TRAIN GENERATOR ===
            if self.gan_type == 'wgan-gp':
                # Train generator every 5 discriminator updates for WGAN-GP
                if i % 5 == 0:
                    self.optimizer_g.zero_grad()
                    
                    # Generate fake images
                    noise = torch.randn(batch_size, self.latent_dim, device=self.device)
                    if hasattr(self.generator, 'conditional') and self.generator.conditional:
                        fake_images = self.generator(noise, real_labels)
                    else:
                        fake_images = self.generator(noise)
                    
                    # Generator loss (maximize discriminator output for fake images)
                    d_fake = self.discriminator(fake_images).view(-1)
                    g_loss = -torch.mean(d_fake)
                    
                    g_loss.backward()
                    self.optimizer_g.step()
            else:
                # Standard DCGAN generator training
                self.optimizer_g.zero_grad()
                
                # Generate fake images
                noise = torch.randn(batch_size, self.latent_dim, device=self.device)
                if hasattr(self.generator, 'conditional') and self.generator.conditional:
                    fake_images = self.generator(noise, real_labels)
                else:
                    fake_images = self.generator(noise)
                
                if hasattr(self.discriminator, 'conditional') and self.discriminator.conditional:
                    d_fake_output, d_fake_class = self.discriminator(fake_images)
                    real_labels_tensor = torch.full((batch_size,), real_label, device=self.device, dtype=torch.float)
                    g_adv_loss = self.criterion(d_fake_output.view(-1), real_labels_tensor)
                    g_class_loss = self.class_criterion(d_fake_class, real_labels)
                    g_loss = g_adv_loss + g_class_loss
                else:
                    d_fake_output = self.discriminator(fake_images).view(-1)
                    real_labels_tensor = torch.full((batch_size,), real_label, device=self.device, dtype=torch.float)
                    g_loss = self.criterion(d_fake_output, real_labels_tensor)
                
                g_loss.backward()
                self.optimizer_g.step()
            
            # Update metrics
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_d_real_acc += d_real_acc
            epoch_d_fake_acc += d_fake_acc
            
            # Update progress bar
            progress_bar.set_postfix({
                'G_Loss': f'{g_loss.item():.4f}',
                'D_Loss': f'{d_loss.item():.4f}',
                'D_Real_Acc': f'{d_real_acc:.3f}',
                'D_Fake_Acc': f'{d_fake_acc:.3f}'
            })
        
        # Calculate epoch averages
        num_batches = len(dataloader)
        epoch_metrics = {
            'generator_loss': epoch_g_loss / num_batches,
            'discriminator_loss': epoch_d_loss / num_batches,
            'd_real_acc': epoch_d_real_acc / num_batches,
            'd_fake_acc': epoch_d_fake_acc / num_batches,
            'gradient_penalty': epoch_gp / num_batches if self.gan_type == 'wgan-gp' else 0.0
        }
        
        return epoch_metrics
    
    def generate_samples(self, 
                        num_samples: int, 
                        labels: Optional[torch.Tensor] = None,
                        save_path: Optional[str] = None) -> torch.Tensor:
        """
        Generate synthetic samples using trained generator.
        
        Args:
            num_samples: Number of samples to generate
            labels: Class labels for conditional generation
            save_path: Path to save generated images
            
        Returns:
            torch.Tensor: Generated images
        """
        self.generator.eval()
        
        with torch.no_grad():
            # Generate random noise
            noise = torch.randn(num_samples, self.latent_dim, device=self.device)
            
            # Handle conditional generation
            if hasattr(self.generator, 'conditional') and self.generator.conditional:
                if labels is None:
                    # Generate random labels for each class
                    labels = torch.randint(0, 10, (num_samples,), device=self.device)
                else:
                    labels = labels.to(self.device)
                fake_images = self.generator(noise, labels)
            else:
                fake_images = self.generator(noise)
            
            # Denormalize images (from [-1, 1] to [0, 1])
            fake_images = (fake_images + 1) / 2
            fake_images = torch.clamp(fake_images, 0, 1)
            
            if save_path:
                # Create directory if it doesn't exist
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                vutils.save_image(fake_images, save_path, nrow=8, normalize=False)
                logger.info(f"Generated samples saved to {save_path}")
        
        return fake_images
    
    def train(self, 
              dataloader: DataLoader,
              num_epochs: int = 100,
              save_dir: str = "./outputs/models/gan_models",
              save_interval: int = 10) -> Dict[str, List[float]]:
        """
        Complete GAN training loop.
        
        Args:
            dataloader: Training data loader
            num_epochs: Number of training epochs
            save_dir: Directory to save model checkpoints
            save_interval: Interval for saving checkpoints and samples
            
        Returns:
            Dict[str, List[float]]: Training history
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting GAN training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train one epoch
            epoch_metrics = self.train_epoch(dataloader, epoch + 1)
            
            # Update training history
            for key, value in epoch_metrics.items():
                self.training_history[key].append(value)
            
            epoch_time = time.time() - start_time
            
            # Log progress
            logger.info(f"Epoch [{epoch+1}/{num_epochs}] "
                       f"G_Loss: {epoch_metrics['generator_loss']:.4f} "
                       f"D_Loss: {epoch_metrics['discriminator_loss']:.4f} "
                       f"D_Real_Acc: {epoch_metrics['d_real_acc']:.3f} "
                       f"D_Fake_Acc: {epoch_metrics['d_fake_acc']:.3f} "
                       f"Time: {epoch_time:.1f}s")
            
            # Save checkpoints and generate samples
            if (epoch + 1) % save_interval == 0:
                # Save model checkpoint
                torch.save({
                    'epoch': epoch + 1,
                    'generator_state_dict': self.generator.state_dict(),
                    'discriminator_state_dict': self.discriminator.state_dict(),
                    'optimizer_g_state_dict': self.optimizer_g.state_dict(),
                    'optimizer_d_state_dict': self.optimizer_d.state_dict(),
                    'training_history': self.training_history
                }, save_dir / f'gan_checkpoint_epoch_{epoch+1}.pth')
                
                # Generate and save sample images
                sample_path = save_dir.parent.parent / "images" / f"gan_samples_epoch_{epoch+1}.png"
                sample_path.parent.mkdir(parents=True, exist_ok=True)
                self.generate_samples(64, save_path=str(sample_path))
        
        logger.info("GAN training completed!")
        return self.training_history


class GANDataAugmentation:
    """
    Data augmentation using pre-trained GAN models.
    
    Integrates synthetic data generation with classification training.
    """
    
    def __init__(self, 
                 generator: nn.Module, 
                 device: torch.device,
                 num_classes: int = 10):
        """
        Initialize GAN-based data augmentation.
        
        Args:
            generator: Pre-trained generator model
            device: Computing device
            num_classes: Number of classes
        """
        self.generator = generator.to(device)
        self.generator.eval()
        self.device = device
        self.num_classes = num_classes
        
        logger.info("GAN data augmentation initialized")
    
    def augment_dataset(self, 
                       original_loader: DataLoader,
                       augmentation_ratio: float = 1.0,
                       class_balanced: bool = True) -> TensorDataset:
        """
        Augment dataset with GAN-generated samples.
        
        Args:
            original_loader: Original training data
            augmentation_ratio: Ratio of synthetic to real samples
            class_balanced: Whether to balance classes in augmentation
            
        Returns:
            TensorDataset: Augmented dataset
        """
        # Collect original data
        original_images = []
        original_labels = []
        
        for images, labels in original_loader:
            original_images.append(images)
            original_labels.append(labels)
        
        original_images = torch.cat(original_images, dim=0)
        original_labels = torch.cat(original_labels, dim=0)
        
        logger.info(f"Original images shape: {original_images.shape}")
        
        # Calculate number of synthetic samples
        total_original = len(original_images)
        num_synthetic = int(total_original * augmentation_ratio)
        
        if class_balanced:
            # Generate equal number of samples per class
            samples_per_class = num_synthetic // self.num_classes
            synthetic_images = []
            synthetic_labels = []
            
            for class_id in range(self.num_classes):
                class_labels = torch.full((samples_per_class,), class_id, dtype=torch.long)
                
                with torch.no_grad():
                    noise = torch.randn(samples_per_class, self.generator.latent_dim, device=self.device)
                    
                    if hasattr(self.generator, 'conditional') and self.generator.conditional:
                        class_images = self.generator(noise, class_labels.to(self.device))
                    else:
                        class_images = self.generator(noise)
                    
                    # Move to CPU and ensure same device as original
                    class_images = class_images.cpu()
                    
                    # Ensure synthetic images have same shape as original
                    if class_images.shape[1:] != original_images.shape[1:]:
                        logger.warning(f"Shape mismatch: Original {original_images.shape[1:]} vs Synthetic {class_images.shape[1:]}")
                        # Resize if needed (this shouldn't happen with proper architecture)
                        class_images = F.interpolate(class_images, size=original_images.shape[2:], mode='bilinear', align_corners=False)
                    
                    # Normalize to match original data distribution
                    # Original data is already normalized, synthetic comes from [-1,1]
                    class_images = (class_images + 1) / 2  # [-1, 1] to [0, 1]
                    
                    # Match the normalization of original CIFAR-10 data
                    # Convert from [0,1] to the same normalization as original
                    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
                    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
                    class_images = (class_images - mean) / std
                
                synthetic_images.append(class_images)
                synthetic_labels.append(class_labels)
            
            synthetic_images = torch.cat(synthetic_images, dim=0)
            synthetic_labels = torch.cat(synthetic_labels, dim=0)
        
        else:
            # Random generation
            with torch.no_grad():
                noise = torch.randn(num_synthetic, self.generator.latent_dim, device=self.device)
                
                if hasattr(self.generator, 'conditional') and self.generator.conditional:
                    random_labels = torch.randint(0, self.num_classes, (num_synthetic,), device=self.device)
                    synthetic_images = self.generator(noise, random_labels)
                else:
                    synthetic_images = self.generator(noise)
                
                synthetic_images = synthetic_images.cpu()
                synthetic_labels = torch.randint(0, self.num_classes, (num_synthetic,))
                
                # Ensure shape consistency
                if synthetic_images.shape[1:] != original_images.shape[1:]:
                    synthetic_images = F.interpolate(synthetic_images, size=original_images.shape[2:], mode='bilinear', align_corners=False)
                
                # Normalize to match original data
                synthetic_images = (synthetic_images + 1) / 2  # [-1, 1] to [0, 1]
                mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
                std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
                synthetic_images = (synthetic_images - mean) / std
        
        logger.info(f"Synthetic images shape: {synthetic_images.shape}")
        logger.info(f"About to concatenate - Original: {original_images.shape}, Synthetic: {synthetic_images.shape}")
        
        # Ensure both tensors are on the same device and have exactly the same shape (except batch dimension)
        assert original_images.shape[1:] == synthetic_images.shape[1:], f"Shape mismatch: {original_images.shape[1:]} vs {synthetic_images.shape[1:]}"
        
        # Combine original and synthetic data
        augmented_images = torch.cat([original_images, synthetic_images], dim=0)
        augmented_labels = torch.cat([original_labels, synthetic_labels], dim=0)
        
        logger.info(f"Dataset augmented: {total_original} -> {len(augmented_images)} samples")
        
        return TensorDataset(augmented_images, augmented_labels)


def calculate_fid_score(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    """
    Calculate Fréchet Inception Distance (FID) score.
    
    Args:
        real_features: Features from real images
        fake_features: Features from generated images
        
    Returns:
        float: FID score (lower is better)
    """
    # Calculate mean and covariance
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    # Calculate squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    
    # Calculate sqrt of product between covariances
    covmean = sqrtm(sigma1.dot(sigma2))
    
    # Check for imaginary numbers
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # Calculate FID
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    
    return fid


def calculate_inception_score(predictions: np.ndarray, splits: int = 10) -> Tuple[float, float]:
    """
    Calculate Inception Score (IS).
    
    Args:
        predictions: Model predictions on generated images
        splits: Number of splits for calculation
        
    Returns:
        Tuple[float, float]: (mean_score, std_score)
    """
    # Split predictions
    split_scores = []
    
    for k in range(splits):
        part = predictions[k * (len(predictions) // splits): (k + 1) * (len(predictions) // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        
        split_scores.append(np.exp(np.mean(scores)))
    
    return np.mean(split_scores), np.std(split_scores)


def visualize_gan_training(training_history: Dict[str, List[float]], 
                          save_path: str = "./outputs/images/gan_training_curves.png"):
    """
    Visualize GAN training progress.
    
    Args:
        training_history: Training metrics history
        save_path: Path to save visualization
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(training_history['generator_loss']) + 1)
    
    # Generator and Discriminator Loss
    axes[0, 0].plot(epochs, training_history['generator_loss'], label='Generator', color='blue')
    axes[0, 0].plot(epochs, training_history['discriminator_loss'], label='Discriminator', color='red')
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Discriminator Accuracies
    axes[0, 1].plot(epochs, training_history['d_real_acc'], label='Real Accuracy', color='green')
    axes[0, 1].plot(epochs, training_history['d_fake_acc'], label='Fake Accuracy', color='orange')
    axes[0, 1].axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Random')
    axes[0, 1].set_title('Discriminator Accuracies')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Loss Difference (Measure of Balance)
    loss_diff = np.array(training_history['generator_loss']) - np.array(training_history['discriminator_loss'])
    axes[1, 0].plot(epochs, loss_diff, color='purple')
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].set_title('Generator - Discriminator Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss Difference')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Gradient Penalty (if WGAN-GP)
    if 'gradient_penalty' in training_history and any(gp > 0 for gp in training_history['gradient_penalty']):
        axes[1, 1].plot(epochs, training_history['gradient_penalty'], color='brown')
        axes[1, 1].set_title('Gradient Penalty (WGAN-GP)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Gradient Penalty')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # Show combined accuracy
        combined_acc = (np.array(training_history['d_real_acc']) + np.array(training_history['d_fake_acc'])) / 2
        axes[1, 1].plot(epochs, combined_acc, color='teal')
        axes[1, 1].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Combined Discriminator Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Average Accuracy')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"GAN training visualization saved to {save_path}")
    plt.show()


def main():
    """
    Main function to demonstrate GAN implementations.
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize data preprocessor
    preprocessor = CIFAR10DataPreprocessor(data_dir="./data")
    
    # Prepare GAN training data
    gan_dataloader = preprocessor.prepare_gan_training_data(
        preprocessor.load_cifar10_datasets()[0]
    )
    
    # Create output directories
    os.makedirs("./outputs/models/gan_models", exist_ok=True)
    os.makedirs("./outputs/images/gan_samples", exist_ok=True)
    
    # Initialize models
    generator = DCGANGenerator(latent_dim=100, conditional=True)
    discriminator = DCGANDiscriminator(conditional=True)
    
    # Initialize trainer
    gan_trainer = GANTrainer(
        generator=generator,
        discriminator=discriminator,
        device=device,
        gan_type='dcgan'
    )
    
    # Train GAN
    training_history = gan_trainer.train(
        dataloader=gan_dataloader,
        num_epochs=20,
        save_interval=10
    )
    
    # Visualize training progress
    visualize_gan_training(training_history)
    
    # Generate sample images
    sample_images = gan_trainer.generate_samples(
        num_samples=64,
        save_path="./outputs/images/final_gan_samples.png"
    )
    
    # Demonstrate data augmentation
    original_loader = DataLoader(
        preprocessor.load_cifar10_datasets()[0],
        batch_size=64,
        shuffle=True
    )
    
    gan_augmentation = GANDataAugmentation(generator, device)
    augmented_dataset = gan_augmentation.augment_dataset(
        original_loader,
        augmentation_ratio=0.5,
        class_balanced=True
    )
    
    logger.info(f"Original dataset size: {len(preprocessor.load_cifar10_datasets()[0])}")
    logger.info(f"Augmented dataset size: {len(augmented_dataset)}")
    
    # Save training history
    with open("./outputs/logs/gan_training_history.json", "w") as f:
        json.dump(training_history, f, indent=2)
    
    logger.info("GAN training and evaluation completed!")


if __name__ == "__main__":
    main()