"""
Step 1: Convert OBSERVATION_DATETIME to time format and extract NBP columns
"""

import pandas as pd
import numpy as np
from pathlib import Path

INFINITY_FILTERED_DIR = Path(r"C:\Users\katia\Desktop\min data\work\data\infinity_filtered\infinity_filtered")
OUTPUT_DIR = Path(r"C:\Users\katia\Desktop\min data\work\data\Patients-infinity\filtered\nbp_time_converted")

OUTPUT_DIR.mkdir(exist_ok=True)


def datetime_to_time(obs_datetime):
    """Convert 20240731134527 to 13:45:27"""
    if pd.isna(obs_datetime):
        return None
    try:
        if isinstance(obs_datetime, float):
            if np.isnan(obs_datetime):
                return None
            ts_str = str(int(obs_datetime))
        else:
            ts_str = str(obs_datetime)
        
        if '.' in ts_str:
            ts_str = ts_str.split('.')[0]
        
        if len(ts_str) >= 14:
            hour = ts_str[8:10]
            minute = ts_str[10:12]
            second = ts_str[12:14]
            return f"{hour}:{minute}:{second}"
        else:
            return None
    except:
        return None


def process_patient_folder(patient_folder):
    """Process one patient folder"""
    print(f"\nProcessing: {patient_folder.name}")
    
    # Find the BLOOD_PRESSURE csv file
    bp_file = None
    for file in patient_folder.glob("*.csv"):
        if 'BLOOD_PRESSURE' in file.name:
            bp_file = file
            break
    
    if bp_file is None:
        print(f"  No BLOOD_PRESSURE file found")
        return False
    
    print(f"  File: {bp_file.name}")
    
    # Read the file
    df = pd.read_csv(bp_file)
    
    # Find OBSERVATION_DATETIME column
    time_col = None
    for col in df.columns:
        if 'OBSERVATION_DATETIME' in col:
            time_col = col
            break
    
    if time_col is None:
        print(f"  No OBSERVATION_DATETIME column found")
        return False
    
    # Convert to time
    df['time'] = df[time_col].apply(datetime_to_time)
    
    # Remove rows with invalid time
    df = df.dropna(subset=['time'])
    
    print(f"  Total rows: {len(df)}")
    print(f"  Time range: {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
    
    # Select only time and NBP columns
    nbp_cols = ['NBP D (mm(hg)^^ISO+)', 'NBP S (mm(hg)^^ISO+)', 'NBP M (mm(hg)^^ISO+)']
    existing_cols = [col for col in nbp_cols if col in df.columns]
    
    if len(existing_cols) == 0:
        print(f"  No NBP columns found")
        return False
    
    # Create output dataframe with time and NBP columns
    df_output = df[['time'] + existing_cols].copy()
    
    # Clean NBP values (remove *)
    for col in existing_cols:
        df_output[col] = df_output[col].astype(str).str.replace('*', '').str.strip()
        df_output[col] = pd.to_numeric(df_output[col], errors='coerce')
    
    # Save to new file
    output_file = OUTPUT_DIR / f"{patient_folder.name}_nbp.csv"
    df_output.to_csv(output_file, index=False)
    
    print(f"  Saved: {output_file.name}")
    print(f"  Rows with NBP values: {df_output[existing_cols].notna().any(axis=1).sum()}")
    
    return True


def main():
    print("=" * 80)
    print("Step 1: Convert OBSERVATION_DATETIME to time format")
    print("Extract NBP columns only")
    print("=" * 80)
    
    # Find all patient folders
    patient_folders = [f for f in INFINITY_FILTERED_DIR.iterdir() if f.is_dir()]
    
    print(f"\nFound {len(patient_folders)} patient folders")
    
    success_count = 0
    
    for folder in patient_folders:
        if process_patient_folder(folder):
            success_count += 1
    
    print("\n" + "=" * 80)
    print(f"SUMMARY: {success_count} files created")
    print(f"Output folder: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()