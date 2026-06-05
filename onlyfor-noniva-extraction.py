"""
Extract Non-invasive data from original events_segments.csv files
- Keep only operation segment
- Convert timestamp to time (HH:MM:SS)
- Keep only rows with NBP values (remove NaN)
- Remove * from values
- Output to operation-csv-none-invasive folder
"""

import pandas as pd
import numpy as np
from pathlib import Path

SOURCE_DIR = Path(r"C:\Users\katia\Desktop\min data\work\data\Patients-infinity\original-files")
OUTPUT_DIR = Path(r"C:\Users\katia\Desktop\min data\work\data\Patients-infinity\filtered\operation-csv-none-invasive")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# NBP column names
NBP_D_COL = 'NBP D (mm(hg)^^ISO+)'
NBP_S_COL = 'NBP S (mm(hg)^^ISO+)'
NBP_M_COL = 'NBP M (mm(hg)^^ISO+)'


def parse_timestamp_to_time(ts):
    """Convert 20250605081547.0 to 08:15:47"""
    if pd.isna(ts):
        return None
    try:
        if isinstance(ts, float):
            ts_str = str(int(ts))
        else:
            ts_str = str(ts)
        
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


def clean_nbp_value(val):
    """Remove * and convert to float, return NaN if empty"""
    if pd.isna(val):
        return np.nan
    if val == '' or val == ' ':
        return np.nan
    if isinstance(val, str):
        val = val.replace('*', '').strip()
        if val == '':
            return np.nan
    try:
        return float(val)
    except:
        return np.nan


def process_patient(patient_folder):
    """Process one patient folder and extract non-invasive data"""
    events_file = patient_folder / "events_segments.csv"
    
    if not events_file.exists():
        return False, "No events_segments.csv"
    
    # Read the file
    df = pd.read_csv(events_file)
    
    # Check required columns
    if 'OBSERVATION_DATETIME' not in df.columns:
        return False, "Missing OBSERVATION_DATETIME column"
    
    if 'segment' not in df.columns:
        return False, "Missing segment column"
    
    # Filter to operation segment only
    df_op = df[df['segment'] == 'operation'].copy()
    
    if len(df_op) == 0:
        return False, "No operation rows found"
    
    # Convert timestamp to time
    df_op['time'] = df_op['OBSERVATION_DATETIME'].apply(parse_timestamp_to_time)
    
    # Remove rows with invalid time
    df_op = df_op.dropna(subset=['time'])
    
    if len(df_op) == 0:
        return False, "No valid timestamps"
    
    # Clean NBP values (remove *)
    for col in [NBP_S_COL, NBP_D_COL, NBP_M_COL]:
        if col in df_op.columns:
            df_op[col] = df_op[col].apply(clean_nbp_value)
    
    # Keep only rows where at least one NBP value exists (not NaN)
    nbp_cols = [col for col in [NBP_S_COL, NBP_D_COL, NBP_M_COL] if col in df_op.columns]
    
    if len(nbp_cols) == 0:
        return False, "No NBP columns found"
    
    df_nbp = df_op[df_op[nbp_cols].notna().any(axis=1)].copy()
    
    if len(df_nbp) == 0:
        return False, "No NBP values found"
    
    # Keep only time and NBP columns
    output_cols = ['time'] + nbp_cols
    df_output = df_nbp[output_cols].copy()
    
    # Get patient name from folder name
    patient_name = patient_folder.name
    
    # Save to CSV
    output_file = OUTPUT_DIR / f"{patient_name}_noninvasive.csv"
    df_output.to_csv(output_file, index=False)
    
    return True, f"{len(df_output)} NBP rows"


def main():
    print("=" * 80)
    print("Extract Non-invasive data from original events_segments.csv")
    print("Keep operation segment, convert time, remove *, save to CSV")
    print("=" * 80)
    
    # Find all patient folders
    patient_folders = []
    for item in SOURCE_DIR.iterdir():
        if item.is_dir():
            name = item.name
            # Skip non-patient folders
            if name in ['filtered', 'sceipts', 'other_analysis', 'scripts']:
                continue
            patient_folders.append(item)
    
    patient_folders.sort()
    print(f"\nFound {len(patient_folders)} patient folders\n")
    
    processed = 0
    failed = 0
    
    for folder in patient_folders:
        print(f"Processing: {folder.name}...", end=' ', flush=True)
        
        success, message = process_patient(folder)
        
        if success:
            processed += 1
            print(f"OK - {message}")
        else:
            failed += 1
            print(f"FAILED - {message}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Successfully processed: {processed}")
    print(f"Failed: {failed}")
    print(f"\nOutput folder: {OUTPUT_DIR}")
    print("Files created: *_noninvasive.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()