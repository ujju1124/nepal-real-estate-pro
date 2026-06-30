#!/usr/bin/env python3
"""Quick verification that app dependencies and data files load correctly."""

import sys

def test_imports():
    """Test all critical imports."""
    try:
        print("Testing imports...")
        import streamlit
        import pandas
        import numpy
        import plotly
        import xgboost
        import catboost
        import sklearn
        print("✓ All core packages imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_data_files():
    """Test that required data files exist and load."""
    import pandas as pd
    import pickle
    
    data_files = [
        'data/housing_model_ready_after_outlier_treatment.csv',
        'data/cleaned_land_merged_final_after_eda.csv',
        'data/cleaned_lalpurja_house_v2_after_cleaning.csv',
        'data/cleaned_lalpurja_land_final_after_eda.csv',
    ]
    
    model_files = [
        'models/xgboost_housing_final.pkl',
        'models/catboost_land_model_final.pkl',
        'models/catboost_lalpurja_house_v2_final.pkl',
        'models/catboost_lalpurja_model_final.pkl',
    ]
    
    print("\nTesting data files...")
    for f in data_files:
        try:
            df = pd.read_csv(f)
            print(f"✓ {f}: {len(df)} rows")
        except Exception as e:
            print(f"✗ {f}: {e}")
            return False
    
    print("\nTesting model files...")
    for f in model_files:
        try:
            with open(f, 'rb') as file:
                pickle.load(file)
            print(f"✓ {f}")
        except Exception as e:
            print(f"✗ {f}: {e}")
            return False
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Nepal Real Estate Pro - Setup Verification")
    print("=" * 60)
    
    success = test_imports() and test_data_files()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All checks passed! App is ready to run.")
        print("\nTo start the app, run:")
        print("  streamlit run app_final.py")
        sys.exit(0)
    else:
        print("❌ Some checks failed. Please review errors above.")
        sys.exit(1)
