#!/usr/bin/env python3
"""
Simple test script to validate the workflow structure and configuration.
This script tests core functionality without requiring all dependencies.
"""

import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_config_loading():
    """Test configuration loading."""
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        print("✅ Configuration loading: PASSED")
        print(f"   Study area: {config['study_area']['name']}")
        print(f"   Risk factors: {len(config['analysis']['risk_factors']['weights'])} configured")
        return True
    except Exception as e:
        print(f"❌ Configuration loading: FAILED - {e}")
        return False

def test_module_structure():
    """Test that all modules can be imported."""
    try:
        # Test utils
        from src.utils.config import ConfigManager
        from src.utils.logging_utils import Logger
        from src.utils.validation import DataValidator
        print("✅ Utils modules: PASSED")
        
        # Test basic functionality
        config_manager = ConfigManager('config.yaml')
        bounds = config_manager.get_study_area_bounds()
        print(f"   Study area bounds: {bounds}")
        
        logger = Logger.get_logger(__name__)
        logger.info("Test log message")
        print("✅ Logger initialization: PASSED")
        
        return True
    except Exception as e:
        print(f"❌ Module structure: FAILED - {e}")
        return False

def test_directory_structure():
    """Test that required directories exist or can be created."""
    try:
        required_dirs = [
            'src',
            'src/utils',
            'src/data_processing', 
            'src/spatial_analysis',
            'src/visualization',
            'data',
            'logs',
            'notebooks'
        ]
        
        for directory in required_dirs:
            dir_path = Path(directory)
            if not dir_path.exists():
                print(f"⚠️  Directory missing: {directory}")
                return False
        
        print("✅ Directory structure: PASSED")
        return True
    except Exception as e:
        print(f"❌ Directory structure: FAILED - {e}")
        return False

def test_file_structure():
    """Test that core files exist."""
    try:
        required_files = [
            'config.yaml',
            'requirements.txt',
            'main.py',
            'src/__init__.py',
            'src/utils/__init__.py',
            'src/data_processing/__init__.py',
            'src/spatial_analysis/__init__.py',
            'src/visualization/__init__.py'
        ]
        
        for file_path in required_files:
            if not Path(file_path).exists():
                print(f"⚠️  File missing: {file_path}")
                return False
        
        print("✅ File structure: PASSED")
        return True
    except Exception as e:
        print(f"❌ File structure: FAILED - {e}")
        return False

def main():
    """Run all tests."""
    print("Methane Emissions Workflow - Structure Validation")
    print("=" * 50)
    
    tests = [
        test_directory_structure,
        test_file_structure,
        test_config_loading,
        test_module_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All core structure tests PASSED!")
        print("\nThe workflow structure is properly configured.")
        print("To run the full analysis, install dependencies with:")
        print("  pip install -r requirements.txt")
        print("Then run:")
        print("  python main.py --demo")
    else:
        print(f"❌ {total - passed} tests FAILED!")
        print("Please check the workflow structure and configuration.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)