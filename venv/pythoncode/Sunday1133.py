Retail Data ETL Pipeline
Location: pythoncode/ (with venv activated)
Files must be in same folder as script
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# =============================================
# STEP 1: EXACT FILENAME DEFINITIONS
# =============================================
INPUT_FILES = {
    'features': 'features_data_set.csv',  # All lowercase with underscores
    'sales': 'sales_data_set.csv',       # All lowercase with underscores
    'stores': 'stores_data_set.csv'      # All lowercase with underscores
}

OUTPUT_FILES = {
    'features': 'cleansed_features_data_set.csv',
    'sales': 'cleansed_sales_data_set.csv',
    'stores': 'cleansed_stores_data_set.csv'
}

# =============================================
# STEP 2: VENV-COMPATIBLE FILE VERIFICATION
# =============================================
def verify_files():
    """Check files exist in script's directory"""
    script_dir = Path(__file__).parent
    print(f"\n🔍 Checking in: {script_dir}")
    
    missing = []
    for name, filename in INPUT_FILES.items():
        if (script_dir / filename).exists():
            print(f"✓ Found '{filename}'")
        else:
            print(f"✕ MISSING: '{filename}'")
            missing.append(filename)
    
    if missing:
        print("\n❌ Missing files detected!")
        print("Folder contents:")
        for f in script_dir.iterdir():
            print(f"- {f.name}")
        return False
    return True

# =============================================
# STEP 3: DATA PROCESSING (VENV-READY)
# =============================================
def load_data(file_type):
    """Load from script's directory"""
    return pd.read_csv(Path(__file__).parent / INPUT_FILES[file_type])

def clean_data(df, dataset_type):
    """Universal cleaner with venv-safe operations"""
    df = df.copy()
    
    # Handle dates
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Handle missing values
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col].fillna(df[col].median(), inplace=True)
        elif col != 'Date':
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Dataset-specific cleaning
    if dataset_type == 'sales' and 'Weekly_Sales' in df.columns:
        df['Weekly_Sales'] = df['Weekly_Sales'].ffill().bfill()
        df = df[df['Weekly_Sales'] >= 0]
    
    return df

def save_data(df, file_type):
    """Save to script's directory"""
    output_path = Path(__file__).parent / OUTPUT_FILES[file_type]
    df.to_csv(output_path, index=False)
    print(f"✓ Saved '{OUTPUT_FILES[file_type]}'")

# =============================================
# MAIN EXECUTION (VENV-COMPATIBLE)
# =============================================
if __name__ == "__main__":
    print("="*50)
    print(f"Running in venv from: {Path(__file__).parent}")
    print("="*50)
    
    if not verify_files():
        print("\n❌ Add missing files to script folder")
        sys.exit(1)
    
    try:
        # Process all datasets
        for dataset in ['features', 'sales', 'stores']:
            print(f"\nProcessing {dataset}...")
            data = load_data(dataset)
            cleaned = clean_data(data, dataset)
            save_data(cleaned, dataset)
        
        print("\n✅ Success! Created:")
        for f in OUTPUT_FILES.values():
            print(f"- {f}")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("Troubleshooting:")
        print("1. Activate venv: 'source venv/bin/activate' (Linux/Mac) or 'venv\\Scripts\\activate' (Windows)")
        print("2. Ensure pandas is installed in venv: 'pip install pandas numpy'")
        print("3. Verify CSV files are not corrupted")
        sys.exit(1)