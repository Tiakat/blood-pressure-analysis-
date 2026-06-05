"""
Superpose Invasive (from aftersai) and Non-invasive data for each patient
With correct patient name matching and filtering
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Input paths
INVASIVE_INPUT_DIR = Path(r"C:\Users\katia\Desktop\min data\work\data\Patients-infinity\filtered\operation-csv-invasive\aftersai")
NONINVASIVE_INPUT_DIR = Path(r"C:\Users\katia\Desktop\min data\work\data\Patients-infinity\filtered\operation-csv-none-invasive")
OUTPUT_DIR = Path(r"C:\Users\katia\Desktop\min data\work\data\Patients-infinity\filtered\superpose")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Column names
INV_S_COL = 'ART S (mm(hg)^^ISO+)'
INV_D_COL = 'ART D (mm(hg)^^ISO+)'
INV_M_COL = 'ART M (mm(hg)^^ISO+)'

NONINV_S_COL = 'NBP S (mm(hg)^^ISO+)'
NONINV_D_COL = 'NBP D (mm(hg)^^ISO+)'
NONINV_M_COL = 'NBP M (mm(hg)^^ISO+)'


def time_to_datetime(time_str, base_date="2024-01-01"):
    """Convert HH:MM:SS to datetime for plotting"""
    try:
        if '.' in str(time_str):
            time_str = str(time_str).split('.')[0]
        return datetime.strptime(f"{base_date} {time_str}", "%Y-%m-%d %H:%M:%S")
    except:
        return None


def filter_noninvasive_by_time_and_value(df_noninv, patient_name):
    """
    Remove specific non-invasive values based on time ranges and thresholds
    """
    if df_noninv is None or len(df_noninv) == 0:
        return df_noninv
    
    df = df_noninv.copy()
    
    # Patient 29: Remove NBP values below 45 mmHg between 08:45 and 08:50
    if "Patient 29" in patient_name or "Patient_29" in patient_name:
        mask_time = (df['time'] >= '08:45:00') & (df['time'] <= '08:50:00')
        mask_value = (df[NONINV_M_COL] < 45)
        df.loc[mask_time & mask_value, NONINV_M_COL] = np.nan
        df.loc[mask_time & mask_value, NONINV_S_COL] = np.nan
        df.loc[mask_time & mask_value, NONINV_D_COL] = np.nan
        removed_count = (mask_time & mask_value).sum()
        print(f"  Patient 29: removed {removed_count} NBP values below 45 at 08:45-08:50")
    
    # Patient 37: Remove specific NBP values
    if "Patient 37" in patient_name or "Patient_37" in patient_name:
        # 11:37 to 11:52 - remove values below 70 mmHg
        mask1 = (df['time'] >= '11:37:00') & (df['time'] <= '11:52:00') & (df[NONINV_M_COL] < 70)
        df.loc[mask1, NONINV_M_COL] = np.nan
        df.loc[mask1, NONINV_S_COL] = np.nan
        df.loc[mask1, NONINV_D_COL] = np.nan
        removed1 = mask1.sum()
        
        # 12:07 to 12:22 - remove values above 80 mmHg
        mask2 = (df['time'] >= '12:07:00') & (df['time'] <= '12:22:00') & (df[NONINV_M_COL] > 80)
        df.loc[mask2, NONINV_M_COL] = np.nan
        df.loc[mask2, NONINV_S_COL] = np.nan
        df.loc[mask2, NONINV_D_COL] = np.nan
        removed2 = mask2.sum()
        
        # 09:22 to 09:37 - remove values above 100 mmHg
        mask3 = (df['time'] >= '09:22:00') & (df['time'] <= '09:37:00') & (df[NONINV_M_COL] > 100)
        df.loc[mask3, NONINV_M_COL] = np.nan
        df.loc[mask3, NONINV_S_COL] = np.nan
        df.loc[mask3, NONINV_D_COL] = np.nan
        removed3 = mask3.sum()
        
        # 13:07 to 13:10 - remove values above 100 mmHg
        mask4 = (df['time'] >= '13:07:00') & (df['time'] <= '13:10:00') & (df[NONINV_M_COL] > 100)
        df.loc[mask4, NONINV_M_COL] = np.nan
        df.loc[mask4, NONINV_S_COL] = np.nan
        df.loc[mask4, NONINV_D_COL] = np.nan
        removed4 = mask4.sum()
        
        print(f"  Patient 37: removed {removed1}+{removed2}+{removed3}+{removed4} NBP outliers")
    
    return df


def filter_invasive_line(df_invasive, patient_name):
    """
    Create gaps in invasive line by removing specific time ranges
    """
    if df_invasive is None or len(df_invasive) == 0:
        return df_invasive
    
    df = df_invasive.copy()
    
    # Patient 49: Remove invasive line between 09:40 and 09:50
    if "Patient 49" in patient_name or "Patient_49" in patient_name:
        mask = (df['time'] >= '09:40:00') & (df['time'] <= '09:50:00')
        df.loc[mask, INV_S_COL] = np.nan
        df.loc[mask, INV_D_COL] = np.nan
        df.loc[mask, INV_M_COL] = np.nan
        print(f"  Patient 49: removed invasive line 09:40-09:50")
    
    # PROMISES 51: Remove invasive line between 09:15 and 09:25
    if "PROMISES 51" in patient_name:
        mask = (df['time'] >= '09:15:00') & (df['time'] <= '09:25:00')
        df.loc[mask, INV_S_COL] = np.nan
        df.loc[mask, INV_D_COL] = np.nan
        df.loc[mask, INV_M_COL] = np.nan
        print(f"  PROMISES 51: removed invasive line 09:15-09:25")
    
    return df


def plot_superposed(patient_name, df_invasive, df_noninv, output_dir):
    """Create one graph per patient with 3 subplots"""
    
    # Apply filters
    df_noninv = filter_noninvasive_by_time_and_value(df_noninv, patient_name)
    df_invasive = filter_invasive_line(df_invasive, patient_name)
    
    # Convert time to datetime
    df_invasive['datetime'] = df_invasive['time'].apply(time_to_datetime)
    if df_noninv is not None and len(df_noninv) > 0:
        df_noninv['datetime'] = df_noninv['time'].apply(time_to_datetime)
        df_noninv = df_noninv.dropna(subset=['datetime'])
    
    df_invasive = df_invasive.dropna(subset=['datetime'])
    
    if len(df_invasive) == 0:
        print(f"  No invasive data")
        return
    
    # Find time range
    start_time = df_invasive['datetime'].min()
    end_time = df_invasive['datetime'].max()
    duration_minutes = (end_time - start_time).total_seconds() / 60
    
    # Determine x-axis tick interval
    if duration_minutes <= 30:
        tick_interval = 2
    elif duration_minutes <= 60:
        tick_interval = 5
    elif duration_minutes <= 120:
        tick_interval = 10
    else:
        tick_interval = 15
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # SYSTOLIC
    ax = axes[0]
    ax.plot(df_invasive['datetime'], df_invasive[INV_S_COL], '-', linewidth=0.8, alpha=0.8,
            color='blue', label='Invasive (ART) - Cleaned')
    if df_noninv is not None and len(df_noninv) > 0:
        df_noninv_clean = df_noninv.dropna(subset=[NONINV_S_COL])
        if len(df_noninv_clean) > 0:
            ax.scatter(df_noninv_clean['datetime'], df_noninv_clean[NONINV_S_COL], 
                       s=15, marker='o', color='red', alpha=0.7, label='Non-invasive (NBP)')
            df_noninv_sorted = df_noninv_clean.sort_values('datetime')
            ax.plot(df_noninv_sorted['datetime'], df_noninv_sorted[NONINV_S_COL], 
                    '-', linewidth=0.5, color='red', alpha=0.4)
    ax.set_ylabel('Systolic BP (mmHg)', fontsize=11)
    ax.set_title(f'{patient_name} - Systolic Blood Pressure', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 250)
    
    # DIASTOLIC
    ax = axes[1]
    ax.plot(df_invasive['datetime'], df_invasive[INV_D_COL], '-', linewidth=0.8, alpha=0.8,
            color='blue', label='Invasive (ART) - Cleaned')
    if df_noninv is not None and len(df_noninv) > 0:
        df_noninv_clean = df_noninv.dropna(subset=[NONINV_D_COL])
        if len(df_noninv_clean) > 0:
            ax.scatter(df_noninv_clean['datetime'], df_noninv_clean[NONINV_D_COL], 
                       s=15, marker='o', color='red', alpha=0.7)
            df_noninv_sorted = df_noninv_clean.sort_values('datetime')
            ax.plot(df_noninv_sorted['datetime'], df_noninv_sorted[NONINV_D_COL], 
                    '-', linewidth=0.5, color='red', alpha=0.4)
    ax.set_ylabel('Diastolic BP (mmHg)', fontsize=11)
    ax.set_title(f'{patient_name} - Diastolic Blood Pressure', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 150)
    
    # MEAN
    ax = axes[2]
    ax.plot(df_invasive['datetime'], df_invasive[INV_M_COL], '-', linewidth=0.8, alpha=0.8,
            color='blue', label='Invasive (ART) - Cleaned')
    if df_noninv is not None and len(df_noninv) > 0:
        df_noninv_clean = df_noninv.dropna(subset=[NONINV_M_COL])
        if len(df_noninv_clean) > 0:
            ax.scatter(df_noninv_clean['datetime'], df_noninv_clean[NONINV_M_COL], 
                       s=15, marker='o', color='red', alpha=0.7)
            df_noninv_sorted = df_noninv_clean.sort_values('datetime')
            ax.plot(df_noninv_sorted['datetime'], df_noninv_sorted[NONINV_M_COL], 
                    '-', linewidth=0.5, color='red', alpha=0.4)
    ax.set_ylabel('Mean BP (mmHg)', fontsize=11)
    ax.set_xlabel('Time (hours:minutes)', fontsize=11)
    ax.set_title(f'{patient_name} - Mean Arterial Pressure', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 200)
    
    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=tick_interval))
        ax.tick_params(axis='x', rotation=45, labelsize=8)
    
    plt.tight_layout()
    output_file = output_dir / f"{patient_name}_superposed.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    print("=" * 80)
    print("Superpose Invasive and Non-invasive data")
    print("With filtering of specific non-invasive outliers and invasive line breaks")
    print("=" * 80)
    
    # Get all invasive files
    invasive_files = list(INVASIVE_INPUT_DIR.glob("*_invasive_cleaned.csv"))
    print(f"\nFound {len(invasive_files)} invasive cleaned files")
    
    if len(invasive_files) == 0:
        print("\nWARNING: No invasive files found!")
        return
    
    # Get all non-invasive files
    noninvasive_files = list(NONINVASIVE_INPUT_DIR.glob("*_noninvasive.csv"))
    print(f"Found {len(noninvasive_files)} non-invasive files")
    
    # Create mapping by extracting base patient name
    patient_map = {}
    
    for f in invasive_files:
        base_name = f.stem.replace("_invasive_cleaned", "")
        patient_map[base_name] = {'invasive': f, 'noninvasive': None}
    
    for f in noninvasive_files:
        base_name = f.stem.replace("_noninvasive", "")
        if base_name in patient_map:
            patient_map[base_name]['noninvasive'] = f
    
    print(f"\nTotal unique patients: {len(patient_map)}")
    print()
    
    processed = 0
    for base_name, files in patient_map.items():
        print(f"Processing: {base_name}...", end=' ', flush=True)
        
        try:
            invasive_file = files['invasive']
            noninvasive_file = files['noninvasive']
            
            if invasive_file is None:
                print(f"SKIP - no invasive file")
                continue
            
            df_invasive = pd.read_csv(invasive_file)
            df_noninv = pd.read_csv(noninvasive_file) if noninvasive_file is not None else pd.DataFrame()
            
            plot_superposed(base_name, df_invasive, df_noninv, OUTPUT_DIR)
            
            processed += 1
            print(f"OK")
        except Exception as e:
            print(f"ERROR: {str(e)[:80]}")
    
    print("\n" + "=" * 80)
    print(f"SUMMARY: {processed} graphs created")
    print(f"Output folder: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()