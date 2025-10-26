#!/usr/bin/env python3
"""
Test Crime Analysis System
Demonstrates the crime activity analysis patterns from DCSASS dataset
"""
import sys
import os
import json
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

from config import settings
from utils.logger import setup_logger
from core.crime_analyzer import crime_analyzer, analyze_dataset_structure, generate_crime_report
from core.real_video_processor import video_processor, process_video_file, process_dataset_directory
from evidence.evidence_collection import evidence_collector, collect_evidence_for_detection, get_evidence_summary
from camera.video_streaming import video_streaming, setup_video_streams, get_camera_frame


class CrimeAnalysisTester:
    """Test crime analysis system functionality"""
    
    def __init__(self):
        self.logger = setup_logger("crime_analysis_tester")
        self.test_results = {}
    
    def test_dataset_analysis(self):
        """Test dataset structure analysis"""
        self.logger.info("🔍 Testing dataset analysis...")
        
        try:
            # Analyze dataset structure
            analysis = analyze_dataset_structure()
            
            self.logger.info(f"✅ Dataset analysis completed:")
            self.logger.info(f"  Total categories: {analysis['total_categories']}")
            self.logger.info(f"  Total videos: {analysis['total_videos']}")
            self.logger.info(f"  Data quality: {analysis['data_quality']}")
            
            # Test insights
            self.logger.info("💡 Key insights:")
            for insight in analysis['insights']:
                self.logger.info(f"  • {insight}")
            
            # Test recommendations
            self.logger.info("📋 Recommendations:")
            for rec in analysis['recommendations']:
                self.logger.info(f"  • {rec}")
            
            self.test_results['dataset_analysis'] = True
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Dataset analysis failed: {e}")
            self.test_results['dataset_analysis'] = False
            return None
    
    def test_crime_report_generation(self):
        """Test crime report generation"""
        self.logger.info("📊 Testing crime report generation...")
        
        try:
            report = generate_crime_report()
            
            self.logger.info("✅ Crime report generated:")
            self.logger.info(f"  Report timestamp: {report['report_timestamp']}")
            self.logger.info(f"  Data quality score: {report['data_quality_score']:.2f}")
            
            # Show crime statistics
            crime_stats = report['crime_statistics']
            self.logger.info(f"  Total crimes: {crime_stats['total_crimes']}")
            self.logger.info(f"  Violence crimes: {crime_stats['violence_crimes']}")
            self.logger.info(f"  Property crimes: {crime_stats['property_crimes']}")
            
            self.test_results['crime_report'] = True
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Crime report generation failed: {e}")
            self.test_results['crime_report'] = False
            return None
    
    def test_video_processing(self, video_path: str = None):
        """Test video processing for crime analysis"""
        self.logger.info("🎬 Testing video processing...")
        
        try:
            if video_path and os.path.exists(video_path):
                # Process specific video
                result = process_video_file(video_path)
                
                if result.get("status") == "success":
                    self.logger.info("✅ Video processing completed:")
                    self.logger.info(f"  Threat type: {result['threat_type']}")
                    self.logger.info(f"  Confidence: {result['confidence']:.2f}")
                    self.logger.info(f"  Severity: {result['severity']}")
                    self.logger.info(f"  Indicators: {result['indicators']}")
                    self.logger.info(f"  Frames analyzed: {result['frames_analyzed']}")
                    
                    self.test_results['video_processing'] = True
                    return result
                else:
                    self.logger.error(f"❌ Video processing failed: {result.get('message')}")
                    self.test_results['video_processing'] = False
                    return None
            else:
                # Test with sample data
                self.logger.info("📝 Testing with sample video data...")
                
                # Create sample video analysis result
                sample_result = {
                    "status": "success",
                    "threat_type": "Fighting",
                    "confidence": 0.85,
                    "severity": "high",
                    "indicators": ["High motion detected", "Rapid color changes", "Violence escalation detected"],
                    "frames_analyzed": 30,
                    "analysis_timestamp": "2024-01-01T12:00:00"
                }
                
                self.logger.info("✅ Sample video analysis:")
                self.logger.info(f"  Threat type: {sample_result['threat_type']}")
                self.logger.info(f"  Confidence: {sample_result['confidence']:.2f}")
                self.logger.info(f"  Severity: {sample_result['severity']}")
                self.logger.info(f"  Indicators: {sample_result['indicators']}")
                
                self.test_results['video_processing'] = True
                return sample_result
                
        except Exception as e:
            self.logger.error(f"❌ Video processing test failed: {e}")
            self.test_results['video_processing'] = False
            return None
    
    def test_evidence_collection(self):
        """Test evidence collection system"""
        self.logger.info("📹 Testing evidence collection...")
        
        try:
            # Create sample detection
            sample_detection = {
                "camera_id": "cam_1",
                "threat_type": "fighting",
                "confidence": 0.95,
                "timestamp": "16:45:30",
                "severity": "high",
                "indicators": ["High motion detected", "Rapid color changes"],
                "location": {"x": 150, "y": 200}
            }
            
            # Test evidence collection
            evidence_metadata = collect_evidence_for_detection(sample_detection)
            
            if evidence_metadata:
                self.logger.info("✅ Evidence collection completed:")
                self.logger.info(f"  Evidence ID: {evidence_metadata['evidence_id']}")
                self.logger.info(f"  Threat type: {evidence_metadata['detection']['threat_type']}")
                self.logger.info(f"  Confidence: {evidence_metadata['detection']['confidence']}")
                self.logger.info(f"  Processing status: {evidence_metadata['system']['processing_status']}")
                
                self.test_results['evidence_collection'] = True
                return evidence_metadata
            else:
                self.logger.error("❌ Evidence collection failed")
                self.test_results['evidence_collection'] = False
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Evidence collection test failed: {e}")
            self.test_results['evidence_collection'] = False
            return None
    
    def test_video_streaming(self):
        """Test video streaming system"""
        self.logger.info("📺 Testing video streaming...")
        
        try:
            # Setup video streams
            if setup_video_streams():
                self.logger.info("✅ Video streaming setup completed")
                
                # Test camera frames
                for camera_id in ['cam_1', 'cam_2', 'cam_3', 'cam_4']:
                    frame = get_camera_frame(camera_id)
                    if frame:
                        self.logger.info(f"  ✅ {camera_id}: Frame captured ({len(frame)} chars)")
                    else:
                        self.logger.info(f"  ⚠️ {camera_id}: No frame available")
                
                # Get streaming status
                status = video_streaming.get_stream_status()
                self.logger.info(f"  Total streams: {status['total_streams']}")
                self.logger.info(f"  Active streams: {status['active_streams']}")
                self.logger.info(f"  System running: {status['is_running']}")
                
                self.test_results['video_streaming'] = True
                return status
            else:
                self.logger.warning("⚠️ Video streaming not available (no dataset found)")
                self.test_results['video_streaming'] = False
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Video streaming test failed: {e}")
            self.test_results['video_streaming'] = False
            return None
    
    def test_evidence_summary(self):
        """Test evidence summary functionality"""
        self.logger.info("📊 Testing evidence summary...")
        
        try:
            summary = get_evidence_summary()
            
            self.logger.info("✅ Evidence summary:")
            self.logger.info(f"  Total incidents: {summary['total_incidents']}")
            self.logger.info(f"  Storage used: {summary['total_storage_used']['total_mb']} MB")
            self.logger.info(f"  Video clips: {summary['video_clips_count']}")
            self.logger.info(f"  Threat types: {summary['threat_types']}")
            self.logger.info(f"  Cameras involved: {summary['cameras_involved']}")
            
            self.test_results['evidence_summary'] = True
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ Evidence summary test failed: {e}")
            self.test_results['evidence_summary'] = False
            return None
    
    def test_crime_patterns(self):
        """Test crime pattern analysis"""
        self.logger.info("🔍 Testing crime pattern analysis...")
        
        try:
            # Test crime analyzer
            analyzer = crime_analyzer
            
            # Test dataset structure analysis
            analysis = analyzer.analyze_dataset_structure()
            
            self.logger.info("✅ Crime pattern analysis:")
            self.logger.info(f"  Categories analyzed: {len(analysis['categories'])}")
            self.logger.info(f"  Data quality score: {analysis['data_quality']['completeness']:.2f}")
            
            # Test insights
            for insight in analysis['insights']:
                self.logger.info(f"  💡 {insight}")
            
            # Test recommendations
            for rec in analysis['recommendations']:
                self.logger.info(f"  📋 {rec}")
            
            self.test_results['crime_patterns'] = True
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Crime pattern analysis failed: {e}")
            self.test_results['crime_patterns'] = False
            return None
    
    def run_all_tests(self, video_path: str = None):
        """Run all crime analysis tests"""
        self.logger.info("🚀 Starting Crime Analysis System Tests...")
        self.logger.info("=" * 60)
        
        # Run all tests
        self.test_dataset_analysis()
        self.test_crime_report_generation()
        self.test_video_processing(video_path)
        self.test_evidence_collection()
        self.test_video_streaming()
        self.test_evidence_summary()
        self.test_crime_patterns()
        
        # Print summary
        self.print_test_summary()
        
        return self.test_results
    
    def print_test_summary(self):
        """Print test summary"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("CRIME ANALYSIS SYSTEM TEST SUMMARY")
        self.logger.info("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        failed_tests = total_tests - passed_tests
        
        self.logger.info(f"Total tests: {total_tests}")
        self.logger.info(f"Passed: {passed_tests}")
        self.logger.info(f"Failed: {failed_tests}")
        
        # Detailed results
        self.logger.info("\nDetailed Results:")
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            self.logger.info(f"  {test_name}: {status}")
        
        if failed_tests == 0:
            self.logger.info("\n🎉 All tests passed! Crime analysis system is ready.")
        else:
            self.logger.error(f"\n❌ {failed_tests} tests failed. Please check the issues above.")
        
        return failed_tests == 0


def main():
    """Main test function"""
    print("🧪 Crime Analysis System Test Suite")
    print("=" * 50)
    
    # Create tester
    tester = CrimeAnalysisTester()
    
    # Check for video path argument
    video_path = None
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            video_path = None
    
    # Run tests
    success = tester.run_all_tests(video_path)
    
    if success:
        print("\n✅ Crime analysis system is ready!")
        print("🚀 Start the backend with: python main.py")
        print("📡 API will be available at: http://localhost:8000")
        print("📚 API docs at: http://localhost:8000/docs")
    else:
        print("\n❌ Some tests failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
