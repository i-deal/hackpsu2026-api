#!/usr/bin/env python3
"""
DCSASS Dataset Data Loader
Handles loading video clips from the DCSASS Dataset structure
"""
import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DCSASSVideoDataset(Dataset):
    """
    Dataset class for DCSASS video clips
    """
    
    def __init__(
        self,
        data_root: None,
        crime_types: Optional[List[str]] = None,
        max_videos_per_crime: int = 1,  # Always 1 folder per crime type
        clip_duration: float = 2.0,  # Existing 2-second clips
        target_size: Tuple[int, int] = (224, 224),
        transform=None,
        random_seed: int = 42
    ):
        """
        Initialize DCSASS Dataset
        
        Args:
            data_root: Path to DCSASS Dataset directory
            crime_types: List of crime types to include (None = all)
            max_videos_per_crime: Always 1 - randomly select one video folder per crime type
            clip_duration: Duration of existing clips (2.0 seconds)
            target_size: Target frame size (height, width)
            transform: PyTorch transforms to apply
            random_seed: Random seed for reproducibility
        """
        self.data_root = Path(data_root)
        self.clip_duration = clip_duration
        self.target_size = target_size
        self.transform = transform or self._get_default_transform()
        self.random_seed = random_seed

        if data_root is None:
            data_root = Path(__file__).parent.parent / "data" / "DCSASS Dataset"
        self.data_root = Path(data_root)
        
        # Set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Get available crime types
        if crime_types is None:
            self.crime_types = [d.name for d in self.data_root.iterdir() 
                              if d.is_dir() and not d.name.startswith('.')]
        else:
            self.crime_types = crime_types
            
        logger.info(f"Found crime types: {self.crime_types}")
        
        # Build dataset
        self.samples = self._build_dataset(max_videos_per_crime)
        logger.info(f"Dataset built with {len(self.samples)} samples")
        
    def _get_default_transform(self):
        """Get default transforms for video frames"""
        return transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    
    def _build_dataset(self, max_videos_per_crime: int) -> List[Dict]:
        """Build dataset by randomly selecting ONE video folder per crime type and loading ALL clips from it"""
        samples = []
        
        for crime_type in self.crime_types:
            crime_dir = self.data_root / crime_type
            
            if not crime_dir.exists():
                logger.warning(f"Crime directory not found: {crime_dir}")
                continue
                
            # Get all video folders for this crime type
            video_folders = [d for d in crime_dir.iterdir() 
                           if d.is_dir() and d.name.endswith('.mp4')]
            
            if not video_folders:
                logger.warning(f"No video folders found for {crime_type}")
                continue
            
            # Randomly select ONE video folder per crime type
            selected_folder = random.choice(video_folders)
            
            logger.info(f"Selected video folder for {crime_type}: {selected_folder.name}")
            
            # Get ALL clips from the selected video folder
            clips = self._get_clips_from_folder(selected_folder, crime_type)
            samples.extend(clips)
            logger.info(f"  Loaded {len(clips)} clips from {selected_folder.name}")
                
        return samples
    
    def _get_clips_from_folder(self, video_folder: Path, crime_type: str) -> List[Dict]:
        """Get all clips from a video folder"""
        clips = []
        
        # Get all video files in the folder
        video_files = [f for f in video_folder.iterdir() 
                      if f.suffix == '.mp4']
        
        for video_file in video_files:
            clip_info = {
                'video_path': str(video_file),
                'crime_type': crime_type,
                'video_folder': video_folder.name,
                'clip_name': video_file.name,
                'label': self._get_crime_label(crime_type)
            }
            clips.append(clip_info)
            
        return clips
    
    def _get_crime_label(self, crime_type: str) -> int:
        """Convert crime type to numeric label"""
        crime_to_label = {
            'Abuse': 0,
            'Arrest': 1, 
            'Arson': 2,
            'Assault': 3,
            'Burglary': 4,
            'Explosion': 5,
            'Fighting': 6,
            'RoadAccidents': 7,
            'Robbery': 8,
            'Shooting': 9,
            'Shoplifting': 10,
            'Stealing': 11,
            'Vandalism': 12
        }
        return crime_to_label.get(crime_type, -1)
    
    def _load_video_clip(self, video_path: str) -> np.ndarray:
        """Load existing 2-second video clip and extract all frames"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        if not cap.isOpened():
            logger.warning(f"Could not open video: {video_path}")
            return np.array([])
        
        # Read all frames from the existing 2-second clip
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        if not frames:
            logger.warning(f"No frames extracted from: {video_path}")
            return np.array([])
            
        return np.array(frames)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset"""
        sample = self.samples[idx]
        
        # Load video clip
        frames = self._load_video_clip(sample['video_path'])
        
        if frames.size == 0:
            # Return dummy data if video couldn't be loaded
            dummy_frames = np.zeros((15, *self.target_size, 3), dtype=np.uint8)
            frames = dummy_frames
        
        # Convert to tensor and apply transforms
        frame_tensors = []
        for frame in frames:
            frame_pil = Image.fromarray(frame)
            frame_tensor = self.transform(frame_pil)
            frame_tensors.append(frame_tensor)
        
        # Stack frames into video tensor
        video_tensor = torch.stack(frame_tensors)
        
        # Cap all videos to 15 frames
        target_frames = 15
        if video_tensor.shape[0] < target_frames:
            # Pad with last frame
            padding = video_tensor[-1:].repeat(target_frames - video_tensor.shape[0], 1, 1, 1)
            video_tensor = torch.cat([video_tensor, padding], dim=0)
        elif video_tensor.shape[0] > target_frames:
            # Truncate to target length
            video_tensor = video_tensor[:target_frames]
        
        return video_tensor, sample['label']


class DCSASSDataLoader:
    """
    Main data loader class for DCSASS Dataset
    """
    
    def __init__(
        self,
        data_root: str = "data/DCSASS Dataset/",
        batch_size: int = 1,
        num_workers: int = 4,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        max_videos_per_crime: int = 5,
        clip_duration: float = 2.5,
        target_size: Tuple[int, int] = (224, 224),
        random_seed: int = 42
    ):
        """
        Initialize DCSASS Data Loader
        
        Args:
            data_root: Path to DCSASS Dataset
            batch_size: Batch size for DataLoader
            num_workers: Number of worker processes
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            test_split: Fraction of data for testing
            max_videos_per_crime: Max videos per crime type
            clip_duration: Duration of each clip in seconds
            target_size: Target frame size
            random_seed: Random seed for reproducibility
        """
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_seed = random_seed
        
        # Create dataset
        self.dataset = DCSASSVideoDataset(
            data_root=data_root,
            max_videos_per_crime=max_videos_per_crime,
            clip_duration=clip_duration,
            target_size=target_size,
            random_seed=random_seed
        )
        
        # Split dataset
        self.train_dataset, self.val_dataset, self.test_dataset = self._split_dataset(
            train_split, val_split, test_split
        )
        
        # Create data loaders
        self.train_loader = self._create_data_loader(self.train_dataset, shuffle=True)
        self.val_loader = self._create_data_loader(self.val_dataset, shuffle=False)
        self.test_loader = self._create_data_loader(self.test_dataset, shuffle=False)
        
        logger.info(f"Data loaders created:")
        logger.info(f"  Train: {len(self.train_dataset)} samples")
        logger.info(f"  Val: {len(self.val_dataset)} samples")
        logger.info(f"  Test: {len(self.test_dataset)} samples")
    
    def _split_dataset(self, train_split: float, val_split: float, test_split: float):
        """Split dataset into train/val/test"""
        total_samples = len(self.dataset)
        
        train_size = int(total_samples * train_split)
        val_size = int(total_samples * val_split)
        test_size = total_samples - train_size - val_size
        
        # Create indices for splitting
        indices = list(range(total_samples))
        random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Create subset datasets
        train_dataset = torch.utils.data.Subset(self.dataset, train_indices)
        val_dataset = torch.utils.data.Subset(self.dataset, val_indices)
        test_dataset = torch.utils.data.Subset(self.dataset, test_indices)
        
        return train_dataset, val_dataset, test_dataset
    
    def _create_data_loader(self, dataset, shuffle: bool = False) -> DataLoader:
        """Create PyTorch DataLoader"""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    def get_dataset_info(self) -> Dict:
        """Get information about the dataset"""
        crime_counts = {}
        for sample in self.dataset.samples:
            crime_type = sample['crime_type']
            crime_counts[crime_type] = crime_counts.get(crime_type, 0) + 1
        
        return {
            'total_samples': len(self.dataset),
            'crime_type_counts': crime_counts,
            'num_crime_types': len(crime_counts),
            'batch_size': self.batch_size,
            'clip_duration': self.dataset.clip_duration,
            'target_size': self.dataset.target_size
        }
    
    def save_dataset_info(self, output_path: str = "dataset_info.json"):
        """Save dataset information to JSON file"""
        info = self.get_dataset_info()
        with open(output_path, 'w') as f:
            json.dump(info, f, indent=2)
        logger.info(f"Dataset info saved to {output_path}")


def test_data_loader():
    """Test the data loader"""
    logger.info("🧪 Testing DCSASS Data Loader...")
    
    try:
        # Create data loader
        data_loader = DCSASSDataLoader(
            data_root="data/DCSASS Dataset/",
            batch_size=4,
            max_videos_per_crime=2,  # Small number for testing
            clip_duration=2.0,
            target_size=(224, 224)
        )
        
        # Get dataset info
        info = data_loader.get_dataset_info()
        logger.info(f"✅ Dataset loaded successfully!")
        logger.info(f"   Total samples: {info['total_samples']}")
        logger.info(f"   Crime types: {info['num_crime_types']}")
        logger.info(f"   Batch size: {info['batch_size']}")
        
        # Test loading a batch
        logger.info("🔄 Testing batch loading...")
        for batch_idx, (videos, labels) in enumerate(data_loader.train_loader):
            logger.info(f"   Batch {batch_idx}: videos shape {videos.shape}, labels shape {labels.shape}")
            if batch_idx >= 2:  # Test only first 3 batches
                break
        
        logger.info("✅ Data loader test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Data loader test failed: {e}")
        return False


if __name__ == "__main__":
    # Test the data loader
    test_data_loader()
