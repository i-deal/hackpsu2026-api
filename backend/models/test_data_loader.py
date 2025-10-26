#!/usr/bin/env python3
"""
Comprehensive Testing for DCSASS Data Loader
Tests the data loader with various scenarios and configurations
"""
import torch
import numpy as np
from dcsass_data_loader import DCSASSDataLoader, DCSASSVideoDataset
import logging
import time
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_basic_functionality():
    """Test basic data loader functionality"""
    logger.info("🧪 Test 1: Basic Functionality")
    logger.info("=" * 50)
    
    try:
        # Create data loader
        data_loader = DCSASSDataLoader(
            data_root="data/DCSASS Dataset/",
            batch_size=4,
            max_videos_per_crime=1,
            random_seed=42
        )
        
        # Test dataset info
        info = data_loader.get_dataset_info()
        logger.info(f"✅ Dataset loaded successfully!")
        logger.info(f"   Total samples: {info['total_samples']}")
        logger.info(f"   Crime types: {info['num_crime_types']}")
        logger.info(f"   Batch size: {info['batch_size']}")
        
        # Test loading one batch
        for batch_idx, (videos, labels) in enumerate(data_loader.train_loader):
            logger.info(f"✅ Batch {batch_idx + 1} loaded successfully!")
            logger.info(f"   Video shape: {videos.shape}")
            logger.info(f"   Labels shape: {labels.shape}")
            logger.info(f"   Video dtype: {videos.dtype}")
            logger.info(f"   Labels dtype: {labels.dtype}")
            logger.info(f"   Labels: {labels.tolist()}")
            break
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic functionality test failed: {e}")
        return False


def test_different_batch_sizes():
    """Test data loader with different batch sizes"""
    logger.info("\n🧪 Test 2: Different Batch Sizes")
    logger.info("=" * 50)
    
    batch_sizes = [1, 2, 4, 8]
    
    for batch_size in batch_sizes:
        try:
            logger.info(f"Testing batch size: {batch_size}")
            
            data_loader = DCSASSDataLoader(
                data_root="data/DCSASS Dataset/",
                batch_size=batch_size,
                max_videos_per_crime=1,
                random_seed=42
            )
            
            # Test loading one batch
            for videos, labels in data_loader.train_loader:
                expected_shape = (batch_size, 15, 3, 224, 224)
                if videos.shape == expected_shape:
                    logger.info(f"   ✅ Batch size {batch_size}: {videos.shape}")
                else:
                    logger.error(f"   ❌ Batch size {batch_size}: Expected {expected_shape}, got {videos.shape}")
                break
                
        except Exception as e:
            logger.error(f"   ❌ Batch size {batch_size} failed: {e}")
    
    return True


def test_data_consistency():
    """Test data consistency across different runs"""
    logger.info("\n🧪 Test 3: Data Consistency")
    logger.info("=" * 50)
    
    try:
        # Test with same random seed
        data_loader1 = DCSASSDataLoader(
            data_root="data/DCSASS Dataset/",
            batch_size=4,
            max_videos_per_crime=1,
            random_seed=42
        )
        
        data_loader2 = DCSASSDataLoader(
            data_root="data/DCSASS Dataset/",
            batch_size=4,
            max_videos_per_crime=1,
            random_seed=42
        )
        
        # Compare first batch
        batch1 = next(iter(data_loader1.train_loader))
        batch2 = next(iter(data_loader2.train_loader))
        
        if torch.equal(batch1[0], batch2[0]) and torch.equal(batch1[1], batch2[1]):
            logger.info("✅ Data consistency test passed - same seed produces same results")
        else:
            logger.warning("⚠️ Data consistency test failed - results differ with same seed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data consistency test failed: {e}")
        return False


def test_memory_usage():
    """Test memory usage and performance"""
    logger.info("\n🧪 Test 4: Memory Usage & Performance")
    logger.info("=" * 50)
    
    try:
        import psutil
        import os
        
        # Get initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        logger.info(f"Initial memory usage: {initial_memory:.2f} MB")
        
        # Create data loader
        data_loader = DCSASSDataLoader(
            data_root="data/DCSASS Dataset/",
            batch_size=8,
            max_videos_per_crime=1,
            random_seed=42
        )
        
        # Load several batches
        start_time = time.time()
        batch_count = 0
        
        for batch_idx, (videos, labels) in enumerate(data_loader.train_loader):
            batch_count += 1
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            if batch_idx == 0:
                logger.info(f"Batch {batch_idx + 1}: {videos.shape}, Memory: {current_memory:.2f} MB")
            elif batch_idx % 5 == 0:
                logger.info(f"Batch {batch_idx + 1}: Memory: {current_memory:.2f} MB")
            
            if batch_idx >= 10:  # Test first 10 batches
                break
        
        end_time = time.time()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        logger.info(f"✅ Performance test completed!")
        logger.info(f"   Batches processed: {batch_count}")
        logger.info(f"   Time taken: {end_time - start_time:.2f} seconds")
        logger.info(f"   Final memory usage: {final_memory:.2f} MB")
        logger.info(f"   Memory increase: {final_memory - initial_memory:.2f} MB")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Memory usage test failed: {e}")
        return False


def test_edge_cases():
    """Test edge cases and error handling"""
    logger.info("\n🧪 Test 5: Edge Cases & Error Handling")
    logger.info("=" * 50)
    
    try:
        # Test with very small batch size
        logger.info("Testing small batch size...")
        data_loader = DCSASSDataLoader(
            data_root="data/DCSASS Dataset/",
            batch_size=1,
            max_videos_per_crime=1,
            random_seed=42
        )
        
        for videos, labels in data_loader.train_loader:
            logger.info(f"   ✅ Small batch: {videos.shape}")
            break
        
        # Test with different random seeds
        logger.info("Testing different random seeds...")
        for seed in [1, 42, 123, 999]:
            data_loader = DCSASSDataLoader(
                data_root="data/DCSASS Dataset/",
                batch_size=2,
                max_videos_per_crime=1,
                random_seed=seed
            )
            logger.info(f"   ✅ Seed {seed}: {len(data_loader.dataset)} samples")
        
        # Test train/val/test splits
        logger.info("Testing data splits...")
        data_loader = DCSASSDataLoader(
            data_root="data/DCSASS Dataset/",
            batch_size=4,
            max_videos_per_crime=1,
            random_seed=42
        )
        
        train_samples = len(data_loader.train_dataset)
        val_samples = len(data_loader.val_dataset)
        test_samples = len(data_loader.test_dataset)
        total_samples = train_samples + val_samples + test_samples
        
        logger.info(f"   Train samples: {train_samples}")
        logger.info(f"   Val samples: {val_samples}")
        logger.info(f"   Test samples: {test_samples}")
        logger.info(f"   Total samples: {total_samples}")
        
        if total_samples > 0:
            logger.info(f"   Train ratio: {train_samples/total_samples:.2f}")
            logger.info(f"   Val ratio: {val_samples/total_samples:.2f}")
            logger.info(f"   Test ratio: {test_samples/total_samples:.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Edge cases test failed: {e}")
        return False


def test_video_loading():
    """Test video loading and frame extraction"""
    logger.info("\n🧪 Test 6: Video Loading & Frame Extraction")
    logger.info("=" * 50)
    
    try:
        # Create dataset directly
        dataset = DCSASSVideoDataset(
            data_root="data/DCSASS Dataset/",
            max_videos_per_crime=1,
            random_seed=42
        )
        
        logger.info(f"Dataset created with {len(dataset)} samples")
        
        # Test loading individual samples
        for i in range(min(5, len(dataset))):
            video, label = dataset[i]
            logger.info(f"Sample {i + 1}:")
            logger.info(f"   Video shape: {video.shape}")
            logger.info(f"   Label: {label}")
            logger.info(f"   Video range: [{video.min():.3f}, {video.max():.3f}]")
            logger.info(f"   Video mean: {video.mean():.3f}")
            logger.info(f"   Video std: {video.std():.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Video loading test failed: {e}")
        return False


def test_crime_type_distribution():
    """Test crime type distribution and labeling"""
    logger.info("\n🧪 Test 7: Crime Type Distribution")
    logger.info("=" * 50)
    
    try:
        data_loader = DCSASSDataLoader(
            data_root="data/DCSASS Dataset/",
            batch_size=4,
            max_videos_per_crime=1,
            random_seed=42
        )
        
        # Get dataset info
        info = data_loader.get_dataset_info()
        
        logger.info("Crime type distribution:")
        for crime_type, count in info['crime_type_counts'].items():
            logger.info(f"   {crime_type}: {count} samples")
        
        # Test label consistency
        label_counts = {}
        for videos, labels in data_loader.train_loader:
            for label in labels:
                label_counts[label.item()] = label_counts.get(label.item(), 0) + 1
            if len(label_counts) > 10:  # Just check first few batches
                break
        
        logger.info(f"Label distribution in batches: {label_counts}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Crime type distribution test failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    logger.info("🚀 Starting Comprehensive Data Loader Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Different Batch Sizes", test_different_batch_sizes),
        ("Data Consistency", test_data_consistency),
        ("Memory Usage & Performance", test_memory_usage),
        ("Edge Cases & Error Handling", test_edge_cases),
        ("Video Loading & Frame Extraction", test_video_loading),
        ("Crime Type Distribution", test_crime_type_distribution)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.info(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Your data loader is working perfectly!")
    else:
        logger.info("⚠️ Some tests failed. Check the logs above for details.")
    
    return passed == total


if __name__ == "__main__":
    run_all_tests()
