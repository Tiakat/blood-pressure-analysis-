"""
Generate cleaned CSV files for invasive data with artifacts removed
Based on SAI detection with expanded artifact region removal
Output: New CSV files with artifacts erased (gaps in data)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Input paths
INVASIVE_INPUT_DIR = Path(r"C:\Users\katia\Desktop\min data\work\data\Patients-infinity\filtered\operation-csv-invasive\beforesai")
SAI_OUTPUT_DIR = Path(r"C:\Users\katia\Desktop\min data\work\data\Patients-infinity\filtered\sai-invasive")
OUTPUT_DIR = Path(r"C:\Users\katia\Desktop\min data\work\data\Patients-infinity\filtered\operation-csv-invasive\aftersai")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXTENSION_BEFORE = 10
EXTENSION_AFTER = 10

# Column names
S_COL = 'ART S (mm(hg)^^ISO+)'
D_COL = 'ART D (mm(hg)^^ISO+)'
M_COL = 'ART M (mm(hg)^^ISO+)'


def time_to_datetime(time_str, base_date="2024-01-01"):
    """Convert HH:MM:SS to datetime for internal processing"""
    try:
        if '.' in str(time_str):
            time_str = str(time_str).split('.')[0]
        return datetime.strptime(f"{base_date} {time_str}", "%Y-%m-%d %H:%M:%S")
    except:
        return None


def apply_sai_to_invasive(df_raw):
    """Apply SAI criteria to invasive data and return cleaned dataframe"""
    if df_raw is None or len(df_raw) == 0:
        return None
    
    df = df_raw.copy()
    df = df.sort_values('time').reset_index(drop=True)
    
    # Convert to numeric
    df[S_COL] = pd.to_numeric(df[S_COL], errors='coerce')
    df[D_COL] = pd.to_numeric(df[D_COL], errors='coerce')
    df[M_COL] = pd.to_numeric(df[M_COL], errors='coerce')
    
    # Calculate changes
    df['S_change'] = df[S_COL].diff().abs()
    df['D_change'] = df[D_COL].diff().abs()
    df['M_change'] = df[M_COL].diff().abs()
    
    # Initialize keep flag
    df['keep'] = True
    df['artifact_reason'] = ''
    
    # SAI criteria
    for i in range(len(df)):
        reasons = []
        
        # C1: Systolic > 300
        if not pd.isna(df.iloc[i][S_COL]) and df.iloc[i][S_COL] > 300:
            reasons.append('SYS>300')
            df.iloc[i, df.columns.get_loc('keep')] = False
        
        # C2: Diastolic < 20
        if not pd.isna(df.iloc[i][D_COL]) and df.iloc[i][D_COL] < 20:
            reasons.append('DIA<20')
            df.iloc[i, df.columns.get_loc('keep')] = False
        
        # C3: Mean < 30
        if not pd.isna(df.iloc[i][M_COL]) and df.iloc[i][M_COL] < 30:
            reasons.append('MAP<30')
            df.iloc[i, df.columns.get_loc('keep')] = False
        
        # C4: Mean > 200
        if not pd.isna(df.iloc[i][M_COL]) and df.iloc[i][M_COL] > 200:
            reasons.append('MAP>200')
            df.iloc[i, df.columns.get_loc('keep')] = False
        
        # C7: Pulse pressure < 20
        if not pd.isna(df.iloc[i][S_COL]) and not pd.isna(df.iloc[i][D_COL]):
            pp = df.iloc[i][S_COL] - df.iloc[i][D_COL]
            if pp < 20:
                reasons.append(f'PP={pp:.0f}<20')
                df.iloc[i, df.columns.get_loc('keep')] = False
        
        # C9: Sudden systolic change > 20
        if i > 0 and not pd.isna(df.iloc[i]['S_change']) and df.iloc[i]['S_change'] > 20:
            reasons.append(f'ΔSYS={df.iloc[i]["S_change"]:.0f}>20')
            df.iloc[i, df.columns.get_loc('keep')] = False
        
        # C10: Sudden diastolic change > 20
        if i > 0 and not pd.isna(df.iloc[i]['D_change']) and df.iloc[i]['D_change'] > 20:
            reasons.append(f'ΔDIA={df.iloc[i]["D_change"]:.0f}>20')
            df.iloc[i, df.columns.get_loc('keep')] = False
        
        # C11: Sudden mean change > 10
        if i > 0 and not pd.isna(df.iloc[i]['M_change']) and df.iloc[i]['M_change'] > 10:
            reasons.append(f'ΔMAP={df.iloc[i]["M_change"]:.0f}>10')
            df.iloc[i, df.columns.get_loc('keep')] = False
        
        if reasons:
            df.iloc[i, df.columns.get_loc('artifact_reason')] = '; '.join(reasons)
    
    # Keep only non-artifact rows
    df_clean = df[df['keep'] == True].copy()
    df_clean = df_clean.reset_index(drop=True)
    
    return df_clean


def expand_artifact_regions(df_raw, df_clean, extension_before, extension_after):
    """Expand artifact detection to completely remove entire artifact regions"""
    df_raw['datetime'] = df_raw['time'].apply(time_to_datetime)
    df_clean['datetime'] = df_clean['time'].apply(time_to_datetime)
    
    df_raw = df_raw.dropna(subset=['datetime'])
    df_clean = df_clean.dropna(subset=['datetime'])
    
    artifact_times = set(df_raw[~df_raw['datetime'].isin(df_clean['datetime'])]['datetime'].values)
    
    if len(artifact_times) == 0:
        return df_clean.copy()
    
    artifact_indices = []
    for idx, row in df_raw.iterrows():
        if row['datetime'] in artifact_times:
            artifact_indices.append(idx)
    
    if len(artifact_indices) == 0:
        return df_clean.copy()
    
    expanded_indices = set()
    for idx in artifact_indices:
        for offset in range(-extension_before, extension_after + 1):
            neighbor_idx = idx + offset
            if 0 <= neighbor_idx < len(df_raw):
                expanded_indices.add(neighbor_idx)
    
    expanded_clean_times = set()
    for idx in range(len(df_raw)):
        if idx not in expanded_indices:
            expanded_clean_times.add(df_raw.iloc[idx]['datetime'])
    
    df_sharp = df_raw[df_raw['datetime'].isin(expanded_clean_times)].copy()
    df_sharp = df_sharp.reset_index(drop=True)
    
    # Keep only time and pressure columns
    df_sharp = df_sharp[['time', S_COL, D_COL, M_COL]]
    
    return df_sharp


def process_patient(patient_name, raw_file):
    """Process one patient: apply SAI, expand artifact regions, save cleaned CSV"""
    
    # Load raw data
    df_raw = pd.read_csv(raw_file)
    print(f"  Raw rows: {len(df_raw)}")
    
    # Apply SAI to detect artifacts
    df_clean_sai = apply_sai_to_invasive(df_raw)
    
    if df_clean_sai is None or len(df_clean_sai) == 0:
        print(f"  No clean data after SAI")
        return 0, 0
    
    print(f"  SAI clean rows: {len(df_clean_sai)}")
    
    # Expand artifact regions to remove neighbors
    df_final = expand_artifact_regions(df_raw, df_clean_sai, EXTENSION_BEFORE, EXTENSION_AFTER)
    
    if df_final is None or len(df_final) == 0:
        print(f"  No data after expansion")
        return 0, 0
    
    print(f"  Final clean rows: {len(df_final)} (after expanding window)")
    
    # Save cleaned CSV
    output_file = OUTPUT_DIR / f"{patient_name}_invasive_cleaned.csv"
    df_final.to_csv(output_file, index=False)
    
    artifacts_removed = len(df_raw) - len(df_final)
    
    return len(df_final), artifacts_removed


def main():
    print("=" * 80)
    print("Generate cleaned CSV files for invasive data")
    print(f"Artifact extension window: {EXTENSION_BEFORE} beats before, {EXTENSION_AFTER} beats after")
    print("=" * 80)
    
    # Get all invasive files
    invasive_files = list(INVASIVE_INPUT_DIR.glob("*_invasive.csv"))
    print(f"\nFound {len(invasive_files)} invasive files\n")
    
    results = []
    processed = 0
    
    for inv_file in invasive_files:
        patient_name = inv_file.stem.replace("_invasive", "")
        print(f"\nProcessing: {patient_name}")
        
        try:
            final_count, artifacts_removed = process_patient(patient_name, inv_file)
            
            results.append({
                'patient': patient_name,
                'final_rows': final_count,
                'artifacts_removed': artifacts_removed
            })
            processed += 1
            
        except Exception as e:
            print(f"  ERROR: {str(e)[:80]}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Successfully processed: {processed}/{len(invasive_files)} patients")
    print(f"\nOutput folder: {OUTPUT_DIR}")
    print("Files created: *_invasive_cleaned.csv")
    print("=" * 80)
    
    # Print summary of results
    print("\nResults summary:")
    for r in results:
        print(f"  {r['patient']}: {r['final_rows']} rows kept, {r['artifacts_removed']} artifacts removed")


if __name__ == "__main__":
    main()