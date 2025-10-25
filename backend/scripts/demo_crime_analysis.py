#!/usr/bin/env python3
"""
Demo Crime Analysis System
Shows how to use the violence detection backend with crime analysis
"""
import sys
import os
import json
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

from config import settings
from utils.logger import setup_logger
from core.crime_analyzer import analyze_dataset_structure, generate_crime_report
from core.real_video_processor import process_video_file
from evidence.evidence_collection import collect_evidence_for_detection, get_evidence_summary
from camera.video_streaming import setup_video_streams, get_camera_frame


def demo_crime_analysis():
    """Demonstrate crime analysis capabilities"""
    print("🚨 Crime Analysis System Demo")
    print("=" * 50)
    
    # 1. Analyze dataset structure
    print("\n📊 1. Analyzing Dataset Structure...")
    analysis = analyze_dataset_structure()
    print(f"   Total categories: {analysis['total_categories']}")
    print(f"   Total videos: {analysis['total_videos']}")
    print(f"   Data quality: {analysis['data_quality']['completeness']:.2f}")
    
    # Show key insights
    print("\n💡 Key Insights:")
    for insight in analysis['insights'][:3]:  # Show first 3 insights
        print(f"   • {insight}")
    
    # 2. Generate crime report
    print("\n📋 2. Generating Crime Report...")
    report = generate_crime_report()
    print(f"   Report timestamp: {report['report_timestamp']}")
    print(f"   Data quality score: {report['data_quality_score']:.2f}")
    
    crime_stats = report['crime_statistics']
    print(f"   Violence crimes: {crime_stats['violence_crimes']}")
    print(f"   Property crimes: {crime_stats['property_crimes']}")
    
    # 3. Simulate video analysis
    print("\n🎬 3. Simulating Video Analysis...")
    sample_detection = {
        "camera_id": "cam_1",
        "threat_type": "fighting",
        "confidence": 0.95,
        "timestamp": "16:45:30",
        "severity": "high",
        "indicators": ["High motion detected", "Rapid color changes"],
        "location": {"x": 150, "y": 200}
    }
    
    print(f"   Threat type: {sample_detection['threat_type']}")
    print(f"   Confidence: {sample_detection['confidence']:.2f}")
    print(f"   Severity: {sample_detection['severity']}")
    print(f"   Indicators: {sample_detection['indicators']}")
    
    # 4. Collect evidence
    print("\n📹 4. Collecting Evidence...")
    evidence = collect_evidence_for_detection(sample_detection)
    print(f"   Evidence ID: {evidence['evidence_id']}")
    print(f"   Processing status: {evidence['system']['processing_status']}")
    
    # 5. Get evidence summary
    print("\n📊 5. Evidence Summary...")
    summary = get_evidence_summary()
    print(f"   Total incidents: {summary['total_incidents']}")
    print(f"   Storage used: {summary['total_storage_used']['total_mb']} MB")
    print(f"   Threat types: {summary['threat_types']}")
    
    # 6. Test video streaming (if available)
    print("\n📺 6. Testing Video Streaming...")
    if setup_video_streams():
        print("   ✅ Video streaming setup successful")
        for camera_id in ['cam_1', 'cam_2', 'cam_3', 'cam_4']:
            frame = get_camera_frame(camera_id)
            if frame:
                print(f"   ✅ {camera_id}: Frame available ({len(frame)} chars)")
            else:
                print(f"   ⚠️ {camera_id}: No frame available")
    else:
        print("   ⚠️ Video streaming not available (no dataset found)")
    
    print("\n🎉 Demo completed successfully!")
    print("\n📡 To start the API server:")
    print("   python3 main.py")
    print("\n🌐 Then visit:")
    print("   http://localhost:8000/docs")


def demo_api_endpoints():
    """Demonstrate API endpoint usage"""
    print("\n🌐 API Endpoint Demo")
    print("=" * 30)
    
    print("Available endpoints:")
    print("  GET  /                    - API information")
    print("  GET  /health              - Health check")
    print("  GET  /crime/analysis     - Crime analysis")
    print("  GET  /crime/report        - Crime report")
    print("  POST /video/analyze       - Video analysis")
    print("  GET  /evidence/summary    - Evidence summary")
    print("  GET  /evidence/files      - List evidence files")
    print("  GET  /streaming/status    - Streaming status")
    print("  GET  /stats               - System statistics")
    
    print("\nExample API calls:")
    print("  curl http://localhost:8000/health")
    print("  curl http://localhost:8000/crime/analysis")
    print("  curl http://localhost:8000/evidence/summary")
    
    print("\n📚 Interactive API documentation:")
    print("  http://localhost:8000/docs")


def demo_with_real_data():
    """Demo with real dataset (if available)"""
    print("\n📁 Real Dataset Demo")
    print("=" * 25)
    
    data_dir = Path("data")
    if data_dir.exists():
        print("✅ Data directory found!")
        
        # List available categories
        categories = [d.name for d in data_dir.iterdir() if d.is_dir()]
        print(f"   Available categories: {categories}")
        
        # Find video files
        video_files = []
        for category in categories:
            category_dir = data_dir / category
            videos = list(category_dir.glob("*.mp4"))
            if videos:
                video_files.extend(videos[:2])  # Take first 2 videos from each category
        
        if video_files:
            print(f"   Found {len(video_files)} video files")
            print("   Sample videos:")
            for video in video_files[:5]:  # Show first 5
                print(f"     • {video.name} ({video.parent.name})")
        else:
            print("   No video files found")
    else:
        print("⚠️ No data directory found")
        print("   Create a 'data' directory with video categories:")
        print("   data/")
        print("   ├── Fighting/")
        print("   ├── Robbery/")
        print("   ├── Assault/")
        print("   └── Normal/")


def main():
    """Main demo function"""
    print("🚨 Violence Detection Backend - Crime Analysis Demo")
    print("=" * 60)
    
    # Run demos
    demo_crime_analysis()
    demo_api_endpoints()
    demo_with_real_data()
    
    print("\n" + "=" * 60)
    print("🎯 Next Steps:")
    print("1. Start the API server: python3 main.py")
    print("2. Visit the API docs: http://localhost:8000/docs")
    print("3. Test with real videos: Add videos to data/ directory")
    print("4. Configure notifications: Set up Discord/Email/SMS")
    print("5. Deploy to production: Use Gunicorn")
    
    print("\n🚀 Your crime analysis system is ready!")


if __name__ == "__main__":
    main()
