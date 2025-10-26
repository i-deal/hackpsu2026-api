"""
Real Video Processor
Processes real videos from DCSASS dataset for violence detection
"""
import os
import cv2
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

from config import settings
from utils.logger import setup_logger
from utils.constants import MULTI_CLASS_CATEGORIES, DEFAULT_CONFIDENCE_THRESHOLD
from .crime_analyzer import CrimeAnalyzer


class RealVideoProcessor:
    """Processes real videos for violence detection"""
    
    def __init__(self):
        self.logger = setup_logger("real_video_processor")
        self.data_dir = settings.DATA_DIR
        self.crime_analyzer = CrimeAnalyzer()
        
        # Processing settings
        self.max_frames_per_video = settings.MAX_FRAMES_PER_VIDEO
        self.confidence_threshold = settings.CONFIDENCE_THRESHOLD
        
        # Analysis results storage
        self.analysis_results = {}
        
        self.logger.info("✅ Real video processor initialized")
    
    def process_video_file(self, video_path: str) -> Dict:
        """Process a single video file for violence detection"""
        if not os.path.exists(video_path):
            return {"status": "error", "message": f"Video file not found: {video_path}"}
        
        self.logger.info(f"🎬 Processing video: {video_path}")
        
        try:
            # Get video properties
            video_info = self._get_video_properties(video_path)
            
            # Analyze video for crime patterns
            crime_analysis = self.crime_analyzer.analyze_video_crime_patterns(video_path)
            
            # Extract frames for detailed analysis
            frames = self._extract_analysis_frames(video_path)
            
            # Analyze frames for violence indicators
            violence_indicators = self._analyze_frames_for_violence(frames)
            
            # Determine threat type and confidence
            threat_type = self._classify_threat_type(violence_indicators, crime_analysis)
            confidence = self._calculate_confidence(violence_indicators, crime_analysis)
            severity = self._assess_severity(violence_indicators, crime_analysis)
            
            # Generate indicators
            indicators = self._generate_indicators(violence_indicators, crime_analysis)
            
            result = {
                "status": "success",
                "video_path": video_path,
                "video_info": video_info,
                "threat_type": threat_type,
                "confidence": confidence,
                "severity": severity,
                "indicators": indicators,
                "analysis_timestamp": datetime.now().isoformat(),
                "frames_analyzed": len(frames),
                "crime_analysis": crime_analysis
            }
            
            self.logger.info(f"✅ Video processed: {threat_type} (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error processing video: {e}")
            return {"status": "error", "message": str(e)}
    
    def _get_video_properties(self, video_path: str) -> Dict:
        """Get video properties"""
        cap = cv2.VideoCapture(video_path)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration": duration,
            "file_size": os.path.getsize(video_path)
        }
    
    def _extract_analysis_frames(self, video_path: str) -> List[np.ndarray]:
        """Extract frames for analysis"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_interval = max(1, frame_count // self.max_frames_per_video)
        
        frame_idx = 0
        while len(frames) < self.max_frames_per_video:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_interval == 0:
                frames.append(frame)
            
            frame_idx += 1
        
        cap.release()
        return frames
    
    def _analyze_frames_for_violence(self, frames: List[np.ndarray]) -> Dict:
        """Analyze frames for violence indicators"""
        indicators = {
            "motion_intensity": [],
            "color_analysis": [],
            "object_detection": [],
            "temporal_patterns": [],
            "violence_indicators": []
        }
        
        for frame in frames:
            # Motion analysis
            motion = self._detect_motion_intensity(frame)
            indicators["motion_intensity"].append(motion)
            
            # Color analysis
            color_analysis = self._analyze_colors(frame)
            indicators["color_analysis"].append(color_analysis)
            
            # Object detection
            objects = self._detect_objects(frame)
            indicators["object_detection"].append(objects)
            
            # Violence indicators
            violence_indicators = self._detect_violence_indicators(frame)
            indicators["violence_indicators"].append(violence_indicators)
        
        # Analyze temporal patterns
        indicators["temporal_patterns"] = self._analyze_temporal_patterns(indicators)
        
        return indicators
    
    def _detect_motion_intensity(self, frame: np.ndarray) -> float:
        """Detect motion intensity in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        return float(np.mean(magnitude))
    
    def _analyze_colors(self, frame: np.ndarray) -> Dict:
        """Analyze colors for violence indicators"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect red colors (potential blood/violence)
        red_mask = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
        red_pixels = np.sum(red_mask > 0)
        
        # Detect skin tones (people)
        skin_mask = cv2.inRange(hsv, (0, 20, 70), (20, 255, 255))
        skin_pixels = np.sum(skin_mask > 0)
        
        # Calculate color variance
        color_variance = np.var(frame)
        
        return {
            "red_intensity": float(red_pixels / (frame.shape[0] * frame.shape[1])),
            "skin_detection": float(skin_pixels / (frame.shape[0] * frame.shape[1])),
            "color_variance": float(color_variance),
            "dominant_colors": self._get_dominant_colors(frame)
        }
    
    def _get_dominant_colors(self, frame: np.ndarray) -> List[Tuple[int, int, int]]:
        """Get dominant colors in frame"""
        # Reshape image to be a list of pixels
        pixels = frame.reshape(-1, 3)
        
        # Use K-means to find dominant colors
        from sklearn.cluster import KMeans
        
        try:
            kmeans = KMeans(n_clusters=3, random_state=42)
            kmeans.fit(pixels)
            
            # Get cluster centers (dominant colors)
            dominant_colors = kmeans.cluster_centers_.astype(int)
            return [tuple(color) for color in dominant_colors]
        except:
            return [(0, 0, 0), (128, 128, 128), (255, 255, 255)]
    
    def _detect_objects(self, frame: np.ndarray) -> List[str]:
        """Detect objects in frame"""
        objects = []
        
        # Edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Significant object
                # Analyze shape
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    if circularity > 0.7:
                        objects.append("circular_object")
                    elif circularity < 0.3:
                        objects.append("linear_object")
                    else:
                        objects.append("irregular_object")
        
        return objects
    
    def _detect_violence_indicators(self, frame: np.ndarray) -> List[str]:
        """Detect specific violence indicators"""
        indicators = []
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect high contrast areas (potential violence)
        contrast = np.std(gray)
        if contrast > 50:
            indicators.append("high_contrast")
        
        # Detect rapid color changes
        color_variance = np.var(frame)
        if color_variance > 10000:
            indicators.append("rapid_color_changes")
        
        # Detect motion blur (potential violence)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100:
            indicators.append("motion_blur")
        
        # Detect red colors (potential blood)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red_mask = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
        red_pixels = np.sum(red_mask > 0)
        if red_pixels > (frame.shape[0] * frame.shape[1] * 0.05):
            indicators.append("red_colors_detected")
        
        return indicators
    
    def _analyze_temporal_patterns(self, indicators: Dict) -> Dict:
        """Analyze temporal patterns in the indicators"""
        patterns = {
            "motion_trend": "stable",
            "escalation": False,
            "peak_activity": 0,
            "duration": 0,
            "violence_escalation": False
        }
        
        # Analyze motion trends
        motion_data = indicators["motion_intensity"]
        if len(motion_data) > 1:
            # Check for escalation
            if motion_data[-1] > motion_data[0] * 1.5:
                patterns["escalation"] = True
                patterns["motion_trend"] = "escalating"
            
            # Find peak activity
            patterns["peak_activity"] = max(motion_data)
            
            # Calculate duration of high activity
            threshold = np.mean(motion_data) + np.std(motion_data)
            high_activity_frames = [m for m in motion_data if m > threshold]
            patterns["duration"] = len(high_activity_frames)
        
        # Analyze violence indicators
        violence_data = indicators["violence_indicators"]
        if violence_data:
            # Check for increasing violence indicators
            violence_counts = [len(v) for v in violence_data]
            if len(violence_counts) > 1 and violence_counts[-1] > violence_counts[0] * 1.5:
                patterns["violence_escalation"] = True
        
        return patterns
    
    def _classify_threat_type(self, violence_indicators: Dict, crime_analysis: Dict) -> str:
        """Classify the type of threat"""
        # Get indicators from both sources
        motion_avg = np.mean(violence_indicators["motion_intensity"]) if violence_indicators["motion_intensity"] else 0
        red_intensity = np.mean([c["red_intensity"] for c in violence_indicators["color_analysis"]]) if violence_indicators["color_analysis"] else 0
        escalation = violence_indicators["temporal_patterns"].get("escalation", False)
        violence_escalation = violence_indicators["temporal_patterns"].get("violence_escalation", False)
        
        # Get crime analysis indicators
        crime_confidence = crime_analysis.get("confidence", 0.0)
        predicted_category = crime_analysis.get("predicted_category", "Normal")
        
        # Classification logic
        if red_intensity > 0.1 and motion_avg > 50:
            return "Fighting"
        elif motion_avg > 100 and escalation:
            return "Assault"
        elif violence_escalation and motion_avg > 30:
            return "Violence"
        elif predicted_category != "Normal" and crime_confidence > 0.5:
            return predicted_category
        elif motion_avg > 30:
            return "Disturbance"
        else:
            return "Normal"
    
    def _calculate_confidence(self, violence_indicators: Dict, crime_analysis: Dict) -> float:
        """Calculate confidence score for the classification"""
        confidence_factors = []
        
        # Motion confidence
        motion_data = violence_indicators["motion_intensity"]
        if motion_data:
            motion_confidence = min(1.0, np.mean(motion_data) / 100.0)
            confidence_factors.append(motion_confidence)
        
        # Color confidence
        color_data = violence_indicators["color_analysis"]
        if color_data:
            red_confidence = min(1.0, np.mean([c["red_intensity"] for c in color_data]) * 10)
            confidence_factors.append(red_confidence)
        
        # Temporal confidence
        temporal = violence_indicators["temporal_patterns"]
        if temporal.get("escalation"):
            confidence_factors.append(0.8)
        if temporal.get("violence_escalation"):
            confidence_factors.append(0.9)
        
        # Crime analysis confidence
        crime_confidence = crime_analysis.get("confidence", 0.0)
        if crime_confidence > 0:
            confidence_factors.append(crime_confidence)
        
        # Return average confidence
        return np.mean(confidence_factors) if confidence_factors else 0.0
    
    def _assess_severity(self, violence_indicators: Dict, crime_analysis: Dict) -> str:
        """Assess the severity of the detected threat"""
        motion_avg = np.mean(violence_indicators["motion_intensity"]) if violence_indicators["motion_intensity"] else 0
        red_intensity = np.mean([c["red_intensity"] for c in violence_indicators["color_analysis"]]) if violence_indicators["color_analysis"] else 0
        escalation = violence_indicators["temporal_patterns"].get("escalation", False)
        violence_escalation = violence_indicators["temporal_patterns"].get("violence_escalation", False)
        
        if red_intensity > 0.1 or (motion_avg > 100 and violence_escalation):
            return "high"
        elif motion_avg > 50 or red_intensity > 0.05 or escalation:
            return "medium"
        else:
            return "low"
    
    def _generate_indicators(self, violence_indicators: Dict, crime_analysis: Dict) -> List[str]:
        """Generate human-readable indicators"""
        indicators = []
        
        # Motion indicators
        motion_avg = np.mean(violence_indicators["motion_intensity"]) if violence_indicators["motion_intensity"] else 0
        if motion_avg > 50:
            indicators.append("High motion detected")
        elif motion_avg > 20:
            indicators.append("Moderate motion detected")
        
        # Color indicators
        color_data = violence_indicators["color_analysis"]
        if color_data:
            red_avg = np.mean([c["red_intensity"] for c in color_data])
            if red_avg > 0.05:
                indicators.append("Red colors detected")
        
        # Temporal indicators
        temporal = violence_indicators["temporal_patterns"]
        if temporal.get("escalation"):
            indicators.append("Motion escalation detected")
        if temporal.get("violence_escalation"):
            indicators.append("Violence escalation detected")
        
        # Crime analysis indicators
        crime_indicators = crime_analysis.get("crime_indicators", {})
        if crime_indicators:
            if crime_indicators.get("motion_intensity"):
                indicators.append("Crime-related motion patterns")
            if crime_indicators.get("color_changes"):
                indicators.append("Suspicious color changes")
        
        return indicators
    
    def process_dataset_directory(self, dataset_path: str) -> Dict:
        """Process all videos in a dataset directory"""
        self.logger.info(f"📁 Processing dataset directory: {dataset_path}")
        
        results = {
            "total_videos": 0,
            "processed_videos": 0,
            "threats_detected": 0,
            "categories": {},
            "videos": []
        }
        
        try:
            dataset_dir = Path(dataset_path)
            if not dataset_dir.exists():
                return {"status": "error", "message": f"Dataset directory not found: {dataset_path}"}
            
            # Process each category
            for category_dir in dataset_dir.iterdir():
                if category_dir.is_dir():
                    category_name = category_dir.name
                    results["categories"][category_name] = {
                        "total_videos": 0,
                        "threats_detected": 0,
                        "videos": []
                    }
                    
                    # Process videos in category
                    for video_file in category_dir.iterdir():
                        if video_file.is_file() and video_file.suffix.lower() in ['.mp4', '.avi', '.mov']:
                            results["total_videos"] += 1
                            results["categories"][category_name]["total_videos"] += 1
                            
                            # Process video
                            video_result = self.process_video_file(str(video_file))
                            
                            if video_result.get("status") == "success":
                                results["processed_videos"] += 1
                                results["categories"][category_name]["videos"].append(video_result)
                                
                                if video_result.get("confidence", 0) > self.confidence_threshold:
                                    results["threats_detected"] += 1
                                    results["categories"][category_name]["threats_detected"] += 1
                                
                                results["videos"].append(video_result)
            
            results["status"] = "success"
            results["processing_timestamp"] = datetime.now().isoformat()
            
            self.logger.info(f"✅ Dataset processing completed: {results['threats_detected']} threats detected")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error processing dataset: {e}")
            return {"status": "error", "message": str(e)}
    
    def save_analysis_results(self, results: Dict, output_path: str = None):
        """Save analysis results to JSON file"""
        if output_path is None:
            output_path = settings.DATA_DIR / "analysis_results.json"
        
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"✅ Analysis results saved: {output_path}")
        except Exception as e:
            self.logger.error(f"❌ Error saving analysis results: {e}")


# Global video processor instance
video_processor = RealVideoProcessor()

def process_video_file(video_path: str) -> Dict:
    """Process a single video file"""
    return video_processor.process_video_file(video_path)

def process_dataset_directory(dataset_path: str) -> Dict:
    """Process all videos in a dataset directory"""
    return video_processor.process_dataset_directory(dataset_path)

# Test function
def test_video_processor():
    """Test video processor functionality"""
    print("🧪 Testing Real Video Processor...")
    
    # Test with a sample video (if available)
    sample_video = "data/sample_video.mp4"
    if os.path.exists(sample_video):
        print(f"\n🎬 Processing sample video: {sample_video}")
        result = process_video_file(sample_video)
        
        if result.get("status") == "success":
            print(f"  Threat type: {result['threat_type']}")
            print(f"  Confidence: {result['confidence']:.2f}")
            print(f"  Severity: {result['severity']}")
            print(f"  Indicators: {result['indicators']}")
        else:
            print(f"  Error: {result.get('message')}")
    else:
        print("  No sample video found for testing")
    
    print("\n✅ Video processor test completed!")

if __name__ == "__main__":
    test_video_processor()
