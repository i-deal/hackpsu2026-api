#!/usr/bin/env python3
"""
Performance Comparison: Original vs Optimized Data Loader
Shows the speed improvements achieved
"""
import time
import torch
from dcsass_data_loader import DCSASSDataLoader
from optimized_data_loader import OptimizedDCSASSDataLoader
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_original_data_loader():
    """Test original data loader performance"""
    logger.info("🐌 Testing Original Data Loader...")
    
    start_time = time.time()
    
    # Create original data loader
    data_loader = DCSASSDataLoader(
        data_root="data/DCSASS Dataset/",
        batch_size=8,
        max_videos_per_crime=1,
        random_seed=42
    )
    
    init_time = time.time()
    
    # Test loading batches
    batch_count = 0
    for videos, labels in data_loader.train_loader:
        batch_count += 1
        if batch_count >= 5:  # Test first 5 batches
            break
    
    end_time = time.time()
    
    return {
        'init_time': init_time - start_time,
        'loading_time': end_time - init_time,
        'total_time': end_time - start_time,
        'batches_loaded': batch_count,
        'samples_per_second': (batch_count * 8) / (end_time - init_time)
    }


def test_optimized_data_loader():
    """Test optimized data loader performance"""
    logger.info("🚀 Testing Optimized Data Loader...")
    
    start_time = time.time()
    
    # Create optimized data loader
    data_loader = OptimizedDCSASSDataLoader(
        data_root="data/DCSASS Dataset/",
        batch_size=8,
        num_workers=4,
        max_videos_per_crime=1,
        random_seed=42,
        use_cache=True,
        pin_memory=True,
        persistent_workers=True
    )
    
    init_time = time.time()
    
    # Test loading batches
    batch_count = 0
    for videos, labels in data_loader.train_loader:
        batch_count += 1
        if batch_count >= 5:  # Test first 5 batches
            break
    
    end_time = time.time()
    
    return {
        'init_time': init_time - start_time,
        'loading_time': end_time - init_time,
        'total_time': end_time - start_time,
        'batches_loaded': batch_count,
        'samples_per_second': (batch_count * 8) / (end_time - init_time)
    }


def run_performance_comparison():
    """Run comprehensive performance comparison"""
    logger.info("📊 Data Loader Performance Comparison")
    logger.info("=" * 60)
    
    # Test original data loader
    logger.info("\n🐌 Original Data Loader:")
    original_results = test_original_data_loader()
    
    # Test optimized data loader
    logger.info("\n🚀 Optimized Data Loader:")
    optimized_results = test_optimized_data_loader()
    
    # Calculate improvements
    init_speedup = original_results['init_time'] / optimized_results['init_time']
    loading_speedup = original_results['loading_time'] / optimized_results['loading_time']
    total_speedup = original_results['total_time'] / optimized_results['total_time']
    throughput_improvement = optimized_results['samples_per_second'] / original_results['samples_per_second']
    
    # Display results
    logger.info("\n📈 Performance Results:")
    logger.info("=" * 60)
    
    logger.info(f"Original Data Loader:")
    logger.info(f"  Initialization: {original_results['init_time']:.2f}s")
    logger.info(f"  Batch Loading: {original_results['loading_time']:.2f}s")
    logger.info(f"  Total Time: {original_results['total_time']:.2f}s")
    logger.info(f"  Throughput: {original_results['samples_per_second']:.1f} samples/sec")
    
    logger.info(f"\nOptimized Data Loader:")
    logger.info(f"  Initialization: {optimized_results['init_time']:.2f}s")
    logger.info(f"  Batch Loading: {optimized_results['loading_time']:.2f}s")
    logger.info(f"  Total Time: {optimized_results['total_time']:.2f}s")
    logger.info(f"  Throughput: {optimized_results['samples_per_second']:.1f} samples/sec")
    
    logger.info(f"\n🚀 Performance Improvements:")
    logger.info(f"  Initialization Speedup: {init_speedup:.1f}x")
    logger.info(f"  Loading Speedup: {loading_speedup:.1f}x")
    logger.info(f"  Total Speedup: {total_speedup:.1f}x")
    logger.info(f"  Throughput Improvement: {throughput_improvement:.1f}x")
    
    # Memory usage comparison
    logger.info(f"\n💾 Memory Optimizations:")
    logger.info(f"  Frame Caching: Enabled")
    logger.info(f"  Parallel Processing: 4 workers")
    logger.info(f"  Pinned Memory: Enabled")
    logger.info(f"  Persistent Workers: Enabled")
    logger.info(f"  Prefetch Factor: 2")
    
    # Recommendations
    logger.info(f"\n💡 Recommendations:")
    if total_speedup > 2:
        logger.info(f"  ✅ Use Optimized Data Loader for training")
    else:
        logger.info(f"  ⚠️ Performance improvement is minimal")
    
    if optimized_results['samples_per_second'] > 50:
        logger.info(f"  ✅ Throughput is excellent for real-time training")
    else:
        logger.info(f"  ⚠️ Consider reducing batch size or workers")
    
    return {
        'original': original_results,
        'optimized': optimized_results,
        'improvements': {
            'init_speedup': init_speedup,
            'loading_speedup': loading_speedup,
            'total_speedup': total_speedup,
            'throughput_improvement': throughput_improvement
        }
    }


if __name__ == "__main__":
    results = run_performance_comparison()
    
    # Summary
    logger.info(f"\n🎯 Summary:")
    logger.info(f"  Total Speedup: {results['improvements']['total_speedup']:.1f}x faster")
    logger.info(f"  Throughput: {results['improvements']['throughput_improvement']:.1f}x more samples/sec")
    logger.info(f"  Recommendation: {'Use Optimized Data Loader' if results['improvements']['total_speedup'] > 1.5 else 'Both are similar'}")
