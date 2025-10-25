#!/usr/bin/env python3
"""
System testing script for Violence Detection Backend
"""
import sys
import asyncio
import logging
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

from config import settings
from utils.logger import setup_logger
from utils.constants import SYSTEM_STATUS


class SystemTester:
    """System testing class"""
    
    def __init__(self):
        self.logger = setup_logger("system_tester")
        self.test_results = {}
    
    def test_imports(self):
        """Test all module imports"""
        self.logger.info("Testing module imports...")
        
        try:
            # Test core imports
            from config import settings
            self.logger.info("✅ Config module imported")
            self.test_results['config'] = True
        except Exception as e:
            self.logger.error(f"❌ Config import failed: {e}")
            self.test_results['config'] = False
        
        try:
            from utils.logger import setup_logger
            self.logger.info("✅ Logger module imported")
            self.test_results['logger'] = True
        except Exception as e:
            self.logger.error(f"❌ Logger import failed: {e}")
            self.test_results['logger'] = False
        
        try:
            from utils.constants import SYSTEM_STATUS
            self.logger.info("✅ Constants module imported")
            self.test_results['constants'] = True
        except Exception as e:
            self.logger.error(f"❌ Constants import failed: {e}")
            self.test_results['constants'] = False
    
    def test_directories(self):
        """Test directory structure"""
        self.logger.info("Testing directory structure...")
        
        required_dirs = [
            settings.DATA_DIR,
            settings.EVIDENCE_DIR,
            settings.MODEL_DIR,
            settings.LOGS_DIR,
            settings.TEMP_DIR,
            settings.CACHE_DIR,
        ]
        
        for directory in required_dirs:
            if directory.exists():
                self.logger.info(f"✅ Directory exists: {directory}")
                self.test_results[f'dir_{directory.name}'] = True
            else:
                self.logger.error(f"❌ Directory missing: {directory}")
                self.test_results[f'dir_{directory.name}'] = False
    
    def test_configuration(self):
        """Test configuration settings"""
        self.logger.info("Testing configuration...")
        
        # Test API settings
        if settings.API_HOST and settings.API_PORT:
            self.logger.info(f"✅ API configured: {settings.API_HOST}:{settings.API_PORT}")
            self.test_results['api_config'] = True
        else:
            self.logger.error("❌ API configuration incomplete")
            self.test_results['api_config'] = False
        
        # Test Redis settings
        if settings.REDIS_URL:
            self.logger.info(f"✅ Redis configured: {settings.REDIS_URL}")
            self.test_results['redis_config'] = True
        else:
            self.logger.error("❌ Redis configuration missing")
            self.test_results['redis_config'] = False
    
    def test_dependencies(self):
        """Test required dependencies"""
        self.logger.info("Testing dependencies...")
        
        required_packages = [
            'fastapi',
            'uvicorn',
            'pydantic',
            'opencv-python',
            'numpy',
            'torch',
            'redis',
            'loguru',
            'colorlog'
        ]
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                self.logger.info(f"✅ {package} available")
                self.test_results[f'dep_{package}'] = True
            except ImportError:
                self.logger.error(f"❌ {package} not available")
                self.test_results[f'dep_{package}'] = False
    
    async def test_redis_connection(self):
        """Test Redis connection"""
        self.logger.info("Testing Redis connection...")
        
        try:
            import redis
            r = redis.Redis.from_url(settings.REDIS_URL)
            r.ping()
            self.logger.info("✅ Redis connection successful")
            self.test_results['redis_connection'] = True
        except Exception as e:
            self.logger.error(f"❌ Redis connection failed: {e}")
            self.logger.info("💡 Make sure Redis is running: redis-server")
            self.test_results['redis_connection'] = False
    
    def test_fastapi_app(self):
        """Test FastAPI application"""
        self.logger.info("Testing FastAPI application...")
        
        try:
            from main import app
            self.logger.info("✅ FastAPI app created successfully")
            self.test_results['fastapi_app'] = True
        except Exception as e:
            self.logger.error(f"❌ FastAPI app creation failed: {e}")
            self.test_results['fastapi_app'] = False
    
    def print_summary(self):
        """Print test summary"""
        self.logger.info("\n" + "="*50)
        self.logger.info("SYSTEM TEST SUMMARY")
        self.logger.info("="*50)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        failed_tests = total_tests - passed_tests
        
        self.logger.info(f"Total tests: {total_tests}")
        self.logger.info(f"Passed: {passed_tests}")
        self.logger.info(f"Failed: {failed_tests}")
        
        if failed_tests == 0:
            self.logger.info("🎉 All tests passed! System is ready.")
            return True
        else:
            self.logger.error(f"❌ {failed_tests} tests failed. Please fix the issues above.")
            return False
    
    async def run_all_tests(self):
        """Run all system tests"""
        self.logger.info("🚀 Starting system tests...")
        
        # Run synchronous tests
        self.test_imports()
        self.test_directories()
        self.test_configuration()
        self.test_dependencies()
        self.test_fastapi_app()
        
        # Run asynchronous tests
        await self.test_redis_connection()
        
        # Print summary
        return self.print_summary()


async def main():
    """Main test function"""
    tester = SystemTester()
    success = await tester.run_all_tests()
    
    if success:
        print("\n✅ System is ready to run!")
        print("🚀 Start the backend with: python main.py")
        print("📡 API will be available at: http://localhost:8000")
        print("📚 API docs at: http://localhost:8000/docs")
    else:
        print("\n❌ System tests failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
