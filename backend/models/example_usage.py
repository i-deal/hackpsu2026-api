#!/usr/bin/env python3
"""
Example usage of DCSASS Data Loader
Shows how to use the data loader for training and inference
"""
import torch
import torch.nn as nn
from dcsass_data_loader import DCSASSDataLoader
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_training_loop():
    """Example of how to use the data loader for training"""
    logger.info("🚀 Example Training Loop with DCSASS Data Loader")
    
    # Create data loader
    data_loader = DCSASSDataLoader(
        data_root="data/DCSASS Dataset/",
        batch_size=8,
        max_videos_per_crime=3,  # Use 3 video folders per crime type
        clip_duration=2.5,  # 2.5 second clips
        target_size=(224, 224),
        random_seed=42
    )
    
    # Get dataset info
    info = data_loader.get_dataset_info()
    logger.info(f"📊 Dataset Info:")
    logger.info(f"   Total samples: {info['total_samples']}")
    logger.info(f"   Crime types: {info['num_crime_types']}")
    logger.info(f"   Batch size: {info['batch_size']}")
    
    # Show crime type distribution
    logger.info(f"📈 Crime Type Distribution:")
    for crime_type, count in info['crime_type_counts'].items():
        logger.info(f"   {crime_type}: {count} samples")
    
    # Example training loop
    logger.info("🔄 Example Training Loop:")
    for epoch in range(2):  # Just 2 epochs for example
        logger.info(f"   Epoch {epoch + 1}/2")
        
        # Training phase
        for batch_idx, (videos, labels) in enumerate(data_loader.train_loader):
            # videos shape: [batch_size, frames, channels, height, width]
            # labels shape: [batch_size]
            
            logger.info(f"     Batch {batch_idx + 1}: {videos.shape[0]} videos, {videos.shape[1]} frames each")
            
            # Here you would:
            # 1. Move data to GPU if available
            # 2. Forward pass through your model
            # 3. Calculate loss
            # 4. Backward pass
            # 5. Update weights
            
            if batch_idx >= 2:  # Just show first 3 batches
                break
        
        # Validation phase
        logger.info(f"   Validation:")
        for batch_idx, (videos, labels) in enumerate(data_loader.val_loader):
            logger.info(f"     Val Batch {batch_idx + 1}: {videos.shape[0]} videos")
            if batch_idx >= 1:  # Just show first 2 batches
                break
    
    logger.info("✅ Training loop example completed!")


def example_inference():
    """Example of how to use the data loader for inference"""
    logger.info("🔍 Example Inference with DCSASS Data Loader")
    
    # Create data loader for inference
    data_loader = DCSASSDataLoader(
        data_root="data/DCSASS Dataset/",
        batch_size=1,  # Single video for inference
        max_videos_per_crime=1,  # Just 1 video per crime type
        clip_duration=2.5,
        target_size=(224, 224)
    )
    
    # Example inference loop
    logger.info("🔄 Example Inference Loop:")
    for batch_idx, (videos, labels) in enumerate(data_loader.test_loader):
        # videos shape: [1, frames, channels, height, width]
        # labels shape: [1]
        
        logger.info(f"   Processing video {batch_idx + 1}")
        logger.info(f"   Video shape: {videos.shape}")
        logger.info(f"   True label: {labels.item()}")
        
        # Here you would:
        # 1. Load your trained model
        # 2. Run inference
        # 3. Get predictions
        # 4. Calculate accuracy
        
        if batch_idx >= 4:  # Just show first 5 videos
            break
    
    logger.info("✅ Inference example completed!")


def example_custom_parameters():
    """Example with custom parameters"""
    logger.info("⚙️ Example with Custom Parameters")
    
    # Create data loader with specific crime types
    data_loader = DCSASSDataLoader(
        data_root="data/DCSASS Dataset/",
        batch_size=4,
        max_videos_per_crime=2,
        clip_duration=3.0,  # 3 second clips
        target_size=(256, 256),  # Larger resolution
        random_seed=123
    )
    
    # Save dataset info
    data_loader.save_dataset_info("custom_dataset_info.json")
    logger.info("📁 Dataset info saved to custom_dataset_info.json")
    
    # Test loading
    for batch_idx, (videos, labels) in enumerate(data_loader.train_loader):
        logger.info(f"   Custom batch {batch_idx + 1}: {videos.shape}")
        if batch_idx >= 1:
            break
    
    logger.info("✅ Custom parameters example completed!")


def main():
    """Run all examples"""
    logger.info("🎬 DCSASS Data Loader Examples")
    logger.info("=" * 50)
    
    try:
        # Example 1: Training loop
        example_training_loop()
        print()
        
        # Example 2: Inference
        example_inference()
        print()
        
        # Example 3: Custom parameters
        example_custom_parameters()
        print()
        
        logger.info("🎉 All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Example failed: {e}")


if __name__ == "__main__":
    main()
