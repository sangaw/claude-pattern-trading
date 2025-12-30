"""
Testing & Verification Module
==============================

Runs tests to verify system functionality and data integrity.
"""

import pandas as pd
import logging
from pathlib import Path
import sys
import os

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
modeling_dir = os.path.dirname(current_dir)
src_dir = os.path.dirname(modeling_dir)
project_root = os.path.dirname(src_dir)

sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)

from utils.data_loader import DataLoader
from modeling.config import TradingConfig
from modeling.utils.indicator_manager import IndicatorManager
from modeling.utils.model_loader import ModelLoader

logger = logging.getLogger(__name__)


def run_tests():
    """Run comprehensive tests"""
    print("\n" + "="*80)
    print("TESTING & VERIFICATION")
    print("="*80)
    
    config = TradingConfig()
    test_results = {
        'passed': [],
        'failed': [],
        'warnings': []
    }
    
    # Test 1: Data Loader
    print("\n[Test 1] Data Loader Test...")
    try:
        csv_path = config.DEFAULT_DATA_FILE
        if csv_path.exists():
            loader = DataLoader(csv_path=str(csv_path))
            summary = loader.summary()
            
            print(f"  ✓ Data loaded: {summary['total_rows']} rows")
            print(f"  ✓ Indicators present: {summary['indicators_present']}")
            print(f"  ✓ Missing indicators: {summary['indicators_missing']}")
            
            if summary['indicators_missing']:
                print(f"  ⚠️ Missing: {summary['missing_list']}")
                test_results['warnings'].append("Some indicators missing from CSV")
            else:
                print("  ✓ All indicators present - no recalculation needed")
            
            test_results['passed'].append("Data Loader")
        else:
            print(f"  ✗ Test file not found: {csv_path}")
            test_results['failed'].append("Data Loader - file not found")
    except Exception as e:
        print(f"  ✗ Data Loader test failed: {e}")
        test_results['failed'].append(f"Data Loader - {str(e)}")
    
    # Test 2: Indicator Manager
    print("\n[Test 2] Indicator Manager Test...")
    try:
        if csv_path.exists():
            loader = DataLoader(csv_path=str(csv_path))
            df = loader.get_dataframe()
            
            indicator_manager = IndicatorManager(df, config)
            required = ['RSI', 'MACD', 'SMA_20', 'SMA_50', 'ATR']
            df_with_indicators = indicator_manager.require(required)
            
            missing_after = [ind for ind in required if ind not in df_with_indicators.columns]
            
            if missing_after:
                print(f"  ✗ Failed to compute: {missing_after}")
                test_results['failed'].append("Indicator Manager - computation failed")
            else:
                print("  ✓ All required indicators available")
                test_results['passed'].append("Indicator Manager")
        else:
            test_results['failed'].append("Indicator Manager - no data file")
    except Exception as e:
        print(f"  ✗ Indicator Manager test failed: {e}")
        test_results['failed'].append(f"Indicator Manager - {str(e)}")
    
    # Test 3: Model Loading
    print("\n[Test 3] Model Loading Test...")
    try:
        model_loader = ModelLoader(config.ARTIFACTS_ROOT)
        model_info = model_loader.get_model_info()
        
        if model_info['ml_model']['found']:
            print(f"  ✓ ML model found: {model_info['ml_model']['path']}")
            test_results['passed'].append("ML Model Loading")
        else:
            print("  ⚠️ No ML model found (this is OK if no training has been done)")
            test_results['warnings'].append("No ML model found")
        
        if model_info['rl_model']['found']:
            print(f"  ✓ RL model found: {model_info['rl_model']['path']}")
            test_results['passed'].append("RL Model Loading")
        else:
            print("  ⚠️ No RL model found (this is OK if no training has been done)")
            test_results['warnings'].append("No RL model found")
    except Exception as e:
        print(f"  ✗ Model loading test failed: {e}")
        test_results['failed'].append(f"Model Loading - {str(e)}")
    
    # Test 4: Configuration
    print("\n[Test 4] Configuration Test...")
    try:
        run_dir = config.get_run_directory()
        print(f"  ✓ Run directory created: {run_dir}")
        
        # Check subdirectories
        for subdir in ['logs', 'reports', 'images', 'models', 'trades']:
            subdir_path = run_dir / subdir
            if subdir_path.exists():
                print(f"  ✓ {subdir} directory exists")
            else:
                print(f"  ✗ {subdir} directory missing")
                test_results['failed'].append(f"Config - {subdir} directory missing")
        
        test_results['passed'].append("Configuration")
    except Exception as e:
        print(f"  ✗ Configuration test failed: {e}")
        test_results['failed'].append(f"Configuration - {str(e)}")
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Passed: {len(test_results['passed'])}")
    print(f"Failed: {len(test_results['failed'])}")
    print(f"Warnings: {len(test_results['warnings'])}")
    
    if test_results['passed']:
        print("\nPassed Tests:")
        for test in test_results['passed']:
            print(f"  ✓ {test}")
    
    if test_results['failed']:
        print("\nFailed Tests:")
        for test in test_results['failed']:
            print(f"  ✗ {test}")
    
    if test_results['warnings']:
        print("\nWarnings:")
        for warning in test_results['warnings']:
            print(f"  ⚠️ {warning}")
    
    print("="*80)
    
    return test_results


if __name__ == "__main__":
    run_tests()

