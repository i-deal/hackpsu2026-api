#!/usr/bin/env python3
"""
Shuffling Comparison: Original vs Fixed Data Loader
Shows the difference in data diversity and shuffling
"""
import time
import torch
from dcsass_data_loader import DCSASSDataLoader
from fixed_data_loader import FixedDCSASSDataLoader
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_data_diversity(data_loader, name: str):
    """Analyze data diversity in the first few batches"""
    logger.info(f"\n📊 Analyzing {name} Data Diversity:")
    
    # Collect labels from first 5 batches
    all_labels = []
    batch_labels = []
    
    for batch_idx, (videos, labels) in enumerate(data_loader.train_loader):
        batch_labels.append(labels.tolist())
        all_labels.extend(labels.tolist())
        
        if batch_idx >= 4:  # Analyze first 5 batches
            break
    
    # Calculate diversity metrics
    unique_labels = len(set(all_labels))
    total_samples = len(all_labels)
    label_counts = {}
    
    for label in all_labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    # Calculate label distribution
    label_distribution = {k: v/total_samples for k, v in label_counts.items()}
    
    logger.info(f"  Total samples analyzed: {total_samples}")
    logger.info(f"  Unique labels: {unique_labels}/13")
    logger.info(f"  Label diversity: {unique_labels/13*100:.1f}%")
    logger.info(f"  First batch labels: {batch_labels[0]}")
    logger.info(f"  Label distribution: {dict(sorted(label_distribution.items()))}")
    
    return {
        'total_samples': total_samples,
        'unique_labels': unique_labels,
        'diversity_percentage': unique_labels/13*100,
        'label_distribution': label_distribution,
        'first_batch': batch_labels[0]
    }


def run_shuffling_comparison():
    """Run comprehensive shuffling comparison"""
    logger.info("🔄 Data Shuffling Comparison")
    logger.info("=" * 60)
    
    # Test original data loader
    logger.info("\n🐌 Original Data Loader (Limited Diversity):")
    original_loader = DCSASSDataLoader(
        data_root="data/DCSASS Dataset/",
        batch_size=8,
        max_videos_per_crime=1,  # Only 1 video per crime type
        random_seed=42
    )
    original_results = analyze_data_diversity(original_loader, "Original")
    
    # Test fixed data loader
    logger.info("\n🚀 Fixed Data Loader (Proper Shuffling):")
    fixed_loader = FixedDCSASSDataLoader(
        data_root="data/DCSASS Dataset/",
        batch_size=8,
        max_videos_per_crime=3,  # 3 videos per crime type for diversity
        random_seed=42
    )
    fixed_results = analyze_data_diversity(fixed_loader, "Fixed")
    
    # Compare results
    logger.info("\n📈 Shuffling Comparison Results:")
    logger.info("=" * 60)
    
    logger.info(f"Original Data Loader:")
    logger.info(f"  Total samples: {original_results['total_samples']}")
    logger.info(f"  Unique labels: {original_results['unique_labels']}/13")
    logger.info(f"  Diversity: {original_results['diversity_percentage']:.1f}%")
    logger.info(f"  Videos per crime: 1")
    
    logger.info(f"\nFixed Data Loader:")
    logger.info(f"  Total samples: {fixed_results['total_samples']}")
    logger.info(f"  Unique labels: {fixed_results['unique_labels']}/13")
    logger.info(f"  Diversity: {fixed_results['diversity_percentage']:.1f}%")
    logger.info(f"  Videos per crime: 3")
    
    # Calculate improvements
    diversity_improvement = fixed_results['diversity_percentage'] - original_results['diversity_percentage']
    sample_increase = fixed_results['total_samples'] - original_results['total_samples']
    
    logger.info(f"\n🚀 Improvements:")
    logger.info(f"  Diversity increase: +{diversity_improvement:.1f}%")
    logger.info(f"  Sample increase: +{sample_increase} samples")
    logger.info(f"  Better shuffling: {'Yes' if fixed_results['unique_labels'] > original_results['unique_labels'] else 'No'}")
    
    # Training impact analysis
    logger.info(f"\n🎯 Training Impact Analysis:")
    if fixed_results['diversity_percentage'] > 90:
        logger.info(f"  ✅ Excellent diversity - should prevent training plateau")
    elif fixed_results['diversity_percentage'] > 70:
        logger.info(f"  ✅ Good diversity - should improve training")
    else:
        logger.info(f"  ⚠️ Limited diversity - may still plateau")
    
    if fixed_results['unique_labels'] == 13:
        logger.info(f"  ✅ All crime types represented in first 5 batches")
    else:
        logger.info(f"  ⚠️ Missing {13 - fixed_results['unique_labels']} crime types in first 5 batches")
    
    # Recommendations
    logger.info(f"\n💡 Recommendations:")
    if diversity_improvement > 20:
        logger.info(f"  ✅ Use Fixed Data Loader for training")
        logger.info(f"  ✅ Should resolve 25% accuracy plateau")
    else:
        logger.info(f"  ⚠️ Consider further data augmentation")
    
    return {
        'original': original_results,
        'fixed': fixed_results,
        'improvements': {
            'diversity_increase': diversity_improvement,
            'sample_increase': sample_increase
        }
    }


if __name__ == "__main__":
    results = run_shuffling_comparison()
    
    # Summary
    logger.info(f"\n🎯 Summary:")
    logger.info(f"  Diversity Improvement: +{results['improvements']['diversity_increase']:.1f}%")
    logger.info(f"  Sample Increase: +{results['improvements']['sample_increase']} samples")
    logger.info(f"  Recommendation: {'Use Fixed Data Loader' if results['improvements']['diversity_increase'] > 10 else 'Both are similar'}")
