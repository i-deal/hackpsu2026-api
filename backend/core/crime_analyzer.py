"""
Crime Activity Analyzer
Analyzes crime patterns and categories from video data
"""
import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

from config import settings
from utils.logger import setup_logger
from utils.constants import MULTI_CLASS_CATEGORIES, THREAT_LEVELS


class CrimeAnalyzer:
    """Analyzes crime activity patterns from video data"""
    
    def __init__(self):
        self.logger = setup_logger("crime_analyzer")
        self.data_dir = settings.DATA_DIR
        
        # Crime categories from DCSASS dataset
        self.crime_categories = {
            "Abuse": 39,
            "Arrest": 26, 
            "Arson": 22,
            "Assault": 23,
            "Burglary": 47,
            "Explosion": 23,
            "Fighting": 8,  # Main focus - needs augmentation
            "RoadAccidents": 76,
            "Robbery": 105,
            "Shooting": 30,
            "Shoplifting": 28,
            "Stealing": 64,
            "Vandalism": 29
        }
        
        # Normal vs Abnormal video counts
        self.normal_videos = 9676
        self.abnormal_videos = 7177
        
        # Analysis results storage
        self.analysis_results = {}
        self.crime_patterns = {}
        
        self.logger.info("Crime analyzer initialized")
    
    def analyze_dataset_structure(self, dataset_path: str = None) -> Dict:
        """Analyze the structure of the crime dataset"""
        if dataset_path:
            self.data_dir = Path(dataset_path)
        
        self.logger.info("🔍 Analyzing crime dataset structure...")
        
        analysis = {
            "total_categories": len(self.crime_categories),
            "total_abnormal_videos": sum(self.crime_categories.values()),
            "total_normal_videos": self.normal_videos,
            "total_videos": self.normal_videos + sum(self.crime_categories.values()),
            "categories": self.crime_categories,
            "insights": self._generate_insights(),
            "data_quality": self._assess_data_quality(),
            "recommendations": self._generate_recommendations()
        }
        
        self.analysis_results = analysis
        return analysis
    
    def _generate_insights(self) -> List[str]:
        """Generate insights about the dataset"""
        insights = []
        
        # Fighting category analysis
        fighting_count = self.crime_categories.get("Fighting", 0)
        insights.append(f"Fighting category has only {fighting_count} videos - needs data augmentation")
        
        # Most common crimes
        max_crimes = max(self.crime_categories.items(), key=lambda x: x[1])
        insights.append(f"Most common crime: {max_crimes[0]} with {max_crimes[1]} videos")
        
        # Data imbalance
        total_abnormal = sum(self.crime_categories.values())
        imbalance_ratio = self.normal_videos / total_abnormal
        insights.append(f"Data imbalance: {imbalance_ratio:.1f}:1 normal to abnormal ratio")
        
        # Violence categories
        violence_categories = ["Fighting", "Assault", "Abuse", "Shooting"]
        violence_count = sum(self.crime_categories.get(cat, 0) for cat in violence_categories)
        insights.append(f"Violence-related crimes: {violence_count} videos")
        
        return insights
    
    def _assess_data_quality(self) -> Dict:
        """Assess the quality of the dataset"""
        quality_metrics = {
            "completeness": 0.0,
            "balance": 0.0,
            "diversity": 0.0,
            "coverage": 0.0
        }
        
        # Completeness - check if all categories have videos
        total_categories = len(self.crime_categories)
        non_zero_categories = len([c for c in self.crime_categories.values() if c > 0])
        quality_metrics["completeness"] = non_zero_categories / total_categories
        
        # Balance - check distribution
        values = list(self.crime_categories.values())
        if values:
            mean_val = sum(values) / len(values)
            variance = sum((x - mean_val) ** 2 for x in values) / len(values)
            quality_metrics["balance"] = 1.0 - (variance / (mean_val ** 2)) if mean_val > 0 else 0.0
        
        # Diversity - number of unique categories
        quality_metrics["diversity"] = min(1.0, total_categories / 20.0)  # Normalize to 20 categories
        
        # Coverage - total video count
        total_videos = sum(self.crime_categories.values())
        quality_metrics["coverage"] = min(1.0, total_videos / 10000.0)  # Normalize to 10k videos
        
        return quality_metrics
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for dataset improvement"""
        recommendations = []
        
        # Fighting category needs more data
        if self.crime_categories.get("Fighting", 0) < 20:
            recommendations.append("Increase Fighting category videos through data augmentation")
        
        # Balance recommendations
        max_crimes = max(self.crime_categories.items(), key=lambda x: x[1])
        min_crimes = min(self.crime_categories.items(), key=lambda x: x[1])
        
        if max_crimes[1] > min_crimes[1] * 5:
            recommendations.append(f"Balance dataset - {max_crimes[0]} has {max_crimes[1]} videos vs {min_crimes[0]} with {min_crimes[1]}")
        
        # Violence focus
        recommendations.append("Focus on violence detection categories: Fighting, Assault, Abuse")
        recommendations.append("Implement data augmentation for underrepresented categories")
        
        return recommendations
    
    def analyze_video_crime_patterns(self, video_path: str) -> Dict:
        """Analyze crime patterns in a specific video"""
        if not os.path.exists(video_path):
            return {"error": f"Video not found: {video_path}"}
        
        self.logger.info(f"🎬 Analyzing crime patterns in: {video_path}")
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Analyze frames for crime indicators
            crime_indicators = self._analyze_frames_for_crime(cap, frame_count)
            
            cap.release()
            
            # Determine crime category based on indicators
            predicted_category = self._classify_crime_type(crime_indicators)
            confidence = self._calculate_confidence(crime_indicators)
            severity = self._assess_severity(crime_indicators)
            
            analysis_result = {
                "video_path": video_path,
                "video_properties": {
                    "fps": fps,
                    "frame_count": frame_count,
                    "width": width,
                    "height": height,
                    "duration": duration
                },
                "crime_indicators": crime_indicators,
                "predicted_category": predicted_category,
                "confidence": confidence,
                "severity": severity,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error analyzing video: {e}")
            return {"error": str(e)}
    
    def _analyze_frames_for_crime(self, cap: cv2.VideoCapture, total_frames: int) -> Dict:
        """Analyze video frames for crime indicators"""
        indicators = {
            "motion_intensity": [],
            "color_changes": [],
            "object_detection": [],
            "scene_changes": [],
            "temporal_patterns": []
        }
        
        # Sample frames for analysis (every 30th frame)
        sample_interval = max(1, total_frames // 30)
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_interval == 0:
                # Analyze frame for crime indicators
                motion = self._detect_motion_intensity(frame)
                color_changes = self._detect_color_changes(frame)
                objects = self._detect_suspicious_objects(frame)
                
                indicators["motion_intensity"].append(motion)
                indicators["color_changes"].append(color_changes)
                indicators["object_detection"].append(objects)
            
            frame_count += 1
        
        # Calculate temporal patterns
        indicators["temporal_patterns"] = self._analyze_temporal_patterns(indicators)
        
        return indicators
    
    def _detect_motion_intensity(self, frame: np.ndarray) -> float:
        """Detect motion intensity in frame"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        return float(np.mean(magnitude))
    
    def _detect_color_changes(self, frame: np.ndarray) -> Dict:
        """Detect significant color changes (potential violence indicators)"""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect red colors (potential blood/violence)
        red_mask = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
        red_pixels = np.sum(red_mask > 0)
        
        # Detect skin tones (people)
        skin_mask = cv2.inRange(hsv, (0, 20, 70), (20, 255, 255))
        skin_pixels = np.sum(skin_mask > 0)
        
        return {
            "red_intensity": float(red_pixels / (frame.shape[0] * frame.shape[1])),
            "skin_detection": float(skin_pixels / (frame.shape[0] * frame.shape[1])),
            "color_variance": float(np.var(frame))
        }
    
    def _detect_suspicious_objects(self, frame: np.ndarray) -> List[str]:
        """Detect potentially suspicious objects"""
        objects = []
        
        # Simple edge detection for object analysis
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
    
    def _analyze_temporal_patterns(self, indicators: Dict) -> Dict:
        """Analyze temporal patterns in the indicators"""
        patterns = {
            "motion_trend": "stable",
            "escalation": False,
            "peak_activity": 0,
            "duration": 0
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
        
        return patterns
    
    def _classify_crime_type(self, indicators: Dict) -> str:
        """Classify the type of crime based on indicators"""
        # Simple rule-based classification
        motion_avg = np.mean(indicators["motion_intensity"]) if indicators["motion_intensity"] else 0
        red_intensity = np.mean([c["red_intensity"] for c in indicators["color_changes"]]) if indicators["color_changes"] else 0
        escalation = indicators["temporal_patterns"].get("escalation", False)
        
        # Classification logic
        if red_intensity > 0.1 and motion_avg > 50:
            return "Fighting"
        elif motion_avg > 100 and escalation:
            return "Assault"
        elif red_intensity > 0.05:
            return "Violence"
        elif motion_avg > 30:
            return "Disturbance"
        else:
            return "Normal"
    
    def _calculate_confidence(self, indicators: Dict) -> float:
        """Calculate confidence score for the classification"""
        confidence_factors = []
        
        # Motion confidence
        motion_data = indicators["motion_intensity"]
        if motion_data:
            motion_confidence = min(1.0, np.mean(motion_data) / 100.0)
            confidence_factors.append(motion_confidence)
        
        # Color confidence
        color_data = indicators["color_changes"]
        if color_data:
            red_confidence = min(1.0, np.mean([c["red_intensity"] for c in color_data]) * 10)
            confidence_factors.append(red_confidence)
        
        # Temporal confidence
        temporal = indicators["temporal_patterns"]
        if temporal.get("escalation"):
            confidence_factors.append(0.8)
        
        # Return average confidence
        return np.mean(confidence_factors) if confidence_factors else 0.0
    
    def _assess_severity(self, indicators: Dict) -> str:
        """Assess the severity of the detected crime"""
        motion_avg = np.mean(indicators["motion_intensity"]) if indicators["motion_intensity"] else 0
        red_intensity = np.mean([c["red_intensity"] for c in indicators["color_changes"]]) if indicators["color_changes"] else 0
        escalation = indicators["temporal_patterns"].get("escalation", False)
        
        if red_intensity > 0.1 or (motion_avg > 100 and escalation):
            return "high"
        elif motion_avg > 50 or red_intensity > 0.05:
            return "medium"
        else:
            return "low"
    
    def generate_crime_report(self, analysis_results: Dict = None) -> Dict:
        """Generate a comprehensive crime analysis report"""
        if analysis_results is None:
            analysis_results = self.analysis_results
        
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "dataset_analysis": analysis_results,
            "crime_statistics": self._calculate_crime_statistics(),
            "recommendations": self._generate_recommendations(),
            "data_quality_score": self._calculate_data_quality_score()
        }
        
        return report
    
    def _calculate_crime_statistics(self) -> Dict:
        """Calculate crime statistics"""
        total_videos = sum(self.crime_categories.values())
        
        # Violence-related crimes
        violence_categories = ["Fighting", "Assault", "Abuse", "Shooting"]
        violence_count = sum(self.crime_categories.get(cat, 0) for cat in violence_categories)
        
        # Property crimes
        property_categories = ["Burglary", "Robbery", "Shoplifting", "Stealing", "Vandalism"]
        property_count = sum(self.crime_categories.get(cat, 0) for cat in property_categories)
        
        return {
            "total_crimes": total_videos,
            "violence_crimes": violence_count,
            "property_crimes": property_count,
            "violence_percentage": (violence_count / total_videos) * 100 if total_videos > 0 else 0,
            "property_percentage": (property_count / total_videos) * 100 if total_videos > 0 else 0
        }
    
    def _calculate_data_quality_score(self) -> float:
        """Calculate overall data quality score"""
        quality_metrics = self._assess_data_quality()
        return np.mean(list(quality_metrics.values()))


# Global crime analyzer instance
crime_analyzer = CrimeAnalyzer()

def analyze_dataset_structure(dataset_path: str = None) -> Dict:
    """Analyze crime dataset structure"""
    return crime_analyzer.analyze_dataset_structure(dataset_path)

def analyze_video_crime_patterns(video_path: str) -> Dict:
    """Analyze crime patterns in a video"""
    return crime_analyzer.analyze_video_crime_patterns(video_path)

def generate_crime_report() -> Dict:
    """Generate comprehensive crime report"""
    return crime_analyzer.generate_crime_report()

# Test function
def test_crime_analyzer():
    """Test crime analyzer functionality"""
    print("🧪 Testing Crime Analyzer...")
    
    # Test dataset analysis
    print("\n📊 Analyzing dataset structure...")
    analysis = analyze_dataset_structure()
    print(f"  Total categories: {analysis['total_categories']}")
    print(f"  Total videos: {analysis['total_videos']}")
    print(f"  Data quality: {analysis['data_quality']}")
    
    # Test insights
    print("\n💡 Key insights:")
    for insight in analysis['insights']:
        print(f"  • {insight}")
    
    # Test recommendations
    print("\n📋 Recommendations:")
    for rec in analysis['recommendations']:
        print(f"  • {rec}")
    
    print("\n✅ Crime analyzer test completed!")

if __name__ == "__main__":
    test_crime_analyzer()
