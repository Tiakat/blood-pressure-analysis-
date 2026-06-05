"""
SHARP SAI visualization - Separate graphs for Invasive and Non-invasive
Invasive graphs saved to: sai-invasive
Non-invasive graphs saved to: sai-none-invasive
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
INVASIVE_INPUT_DIR = Path(r"C:\Users\katia\Desktop\min data\work\data\Patients-infinity\filtered\operation-csv-invasive")
NONINVASIVE_INPUT_DIR = Path(r"C:\Users\katia\Desktop\min data\work\data\Patients-infinity\filtered\operation-csv-none-invasive")

# Output paths for graphs
INVASIVE_OUTPUT_DIR = Path(r"C:\Users\katia\Desktop\min data\work\data\Patients-infinity\filtered\sai-invasive")
NONINVASIVE_OUTPUT_DIR = Path(r"C:\Users\katia\Desktop\min data\work\data\Patients-infinity\filtered\sai-none-invasive")

INVASIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NONINVASIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXTENSION_BEFORE = 10
EXTENSION_AFTER = 10

# Invasive column names
INV_S_COL = 'ART S (mm(hg)^^ISO+)'
INV_D_COL = 'ART D (mm(hg)^^ISO+)'
INV_M_COL = 'ART M (mm(hg)^^ISO+)'

# Non-invasive column names
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


def apply_sai_to_data(df_raw, s_col, d_col, m_col, is_invasive=True):
    """Apply SAI criteria to data and return cleaned dataframe"""
    if df_raw is None or len(df_raw) == 0:
        return None
    
    df = df_raw.copy()
    df = df.sort_values('time').reset_index(drop=True)
    
    # Convert to numeric
    df[s_col] = pd.to_numeric(df[s_col], errors='coerce')
    df[d_col] = pd.to_numeric(df[d_col], errors='coerce')
    df[m_col] = pd.to_numeric(df[m_col], errors='coerce')
    
    # Calculate changes
    df['S_change'] = df[s_col].diff().abs()
    df['D_change'] = df[d_col].diff().abs()
    df['M_change'] = df[m_col].diff().abs()
    
    # Initialize keep flag
    df['keep'] = True
    df['artifact_reason'] = ''
    
    # SAI criteria
    for i in range(len(df)):
        reasons = []
        
        if is_invasive:
            # Invasive thresholds
            if not pd.isna(df.iloc[i][s_col]) and df.iloc[i][s_col] > 300:
                reasons.append('SYS>300')
                df.iloc[i, df.columns.get_loc('keep')] = False
            if not pd.isna(df.iloc[i][d_col]) and df.iloc[i][d_col] < 20:
                reasons.append('DIA<20')
                df.iloc[i, df.columns.get_loc('keep')] = False
        else:
            # Non-invasive thresholds
            if not pd.isna(df.iloc[i][s_col]) and df.iloc[i][s_col] > 250:
                reasons.append('SYS>250')
                df.iloc[i, df.columns.get_loc('keep')] = False
            if not pd.isna(df.iloc[i][d_col]) and df.iloc[i][d_col] < 20:
                reasons.append('DIA<20')
                df.iloc[i, df.columns.get_loc('keep')] = False
        
        # Common thresholds
        if not pd.isna(df.iloc[i][m_col]) and df.iloc[i][m_col] < 30:
            reasons.append('MAP<30')
            df.iloc[i, df.columns.get_loc('keep')] = False
        if not pd.isna(df.iloc[i][m_col]) and df.iloc[i][m_col] > 200:
            reasons.append('MAP>200')
            df.iloc[i, df.columns.get_loc('keep')] = False
        
        # Pulse pressure < 20
        if not pd.isna(df.iloc[i][s_col]) and not pd.isna(df.iloc[i][d_col]):
            pp = df.iloc[i][s_col] - df.iloc[i][d_col]
            if pp < 20:
                reasons.append(f'PP={pp:.0f}<20')
                df.iloc[i, df.columns.get_loc('keep')] = False
        
        # Sudden changes
        if i > 0 and not pd.isna(df.iloc[i]['S_change']) and df.iloc[i]['S_change'] > 20:
            reasons.append(f'ΔSYS={df.iloc[i]["S_change"]:.0f}>20')
            df.iloc[i, df.columns.get_loc('keep')] = False
        if i > 0 and not pd.isna(df.iloc[i]['D_change']) and df.iloc[i]['D_change'] > 20:
            reasons.append(f'ΔDIA={df.iloc[i]["D_change"]:.0f}>20')
            df.iloc[i, df.columns.get_loc('keep')] = False
        if i > 0 and not pd.isna(df.iloc[i]['M_change']) and df.iloc[i]['M_change'] > 10:
            reasons.append(f'ΔMAP={df.iloc[i]["M_change"]:.0f}>10')
            df.iloc[i, df.columns.get_loc('keep')] = False
        
        if reasons:
            df.iloc[i, df.columns.get_loc('artifact_reason')] = '; '.join(reasons)
    
    # Keep only non-artifact rows
    df_clean = df[df['keep'] == True].copy()
    df_clean = df_clean.reset_index(drop=True)
    
    return df_clean


def expand_artifact_regions_complete(df_raw, df_clean, extension_before, extension_after):
    """Expand artifact detection to completely remove entire artifact regions."""
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
    
    return df_sharp


def create_complete_gaps(df_raw, df_clean_sharp, col_name):
    """Create version where all artifact regions are replaced with NaN (complete gaps)"""
    df_gaps = df_raw.copy()
    clean_times = set(df_clean_sharp['datetime'].values)
    
    for idx in df_gaps.index:
        if df_gaps.loc[idx, 'datetime'] not in clean_times:
            df_gaps.loc[idx, col_name] = np.nan
    
    return df_gaps


def plot_patient(patient_name, raw_file, output_dir, is_invasive=True):
    """Create one PNG per patient with 6 graphs (3 pressures x 2 columns)"""
    
    df_raw = pd.read_csv(raw_file)
    
    if is_invasive:
        s_col = INV_S_COL
        d_col = INV_D_COL
        m_col = INV_M_COL
        data_type = "Invasive"
    else:
        s_col = NONINV_S_COL
        d_col = NONINV_D_COL
        m_col = NONINV_M_COL
        data_type = "Non-invasive"
    
    # Apply SAI to get artifact detection
    df_clean_original = apply_sai_to_data(df_raw, s_col, d_col, m_col, is_invasive)
    
    if df_clean_original is None or len(df_clean_original) == 0:
        print(f"  No clean data after SAI")
        return 0, 0, 0, 0
    
    df_raw['datetime'] = df_raw['time'].apply(time_to_datetime)
    df_clean_original['datetime'] = df_clean_original['time'].apply(time_to_datetime)
    
    df_raw = df_raw.dropna(subset=['datetime'])
    df_clean_original = df_clean_original.dropna(subset=['datetime'])
    
    if len(df_raw) == 0:
        return 0, 0, 0, 0
    
    # Expand artifact regions
    df_sharp_clean = expand_artifact_regions_complete(df_raw, df_clean_original, EXTENSION_BEFORE, EXTENSION_AFTER)
    
    # Find original artifact locations
    original_removed = df_raw[~df_raw['datetime'].isin(df_clean_original['datetime'])].copy()
    
    # Create gap data for each pressure
    df_gaps_s = create_complete_gaps(df_raw, df_sharp_clean, s_col)
    df_gaps_d = create_complete_gaps(df_raw, df_sharp_clean, d_col)
    df_gaps_m = create_complete_gaps(df_raw, df_sharp_clean, m_col)
    
    # Find time range for x-axis
    start_time = df_raw['datetime'].min()
    end_time = df_raw['datetime'].max()
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
    
    # Create figure with 3 rows x 2 columns
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    
    # Pressure configurations
    pressures = [
        (s_col, 'Systolic BP (mmHg)', 0, 250, axes[0, 0], axes[0, 1], df_gaps_s),
        (d_col, 'Diastolic BP (mmHg)', 0, 150, axes[1, 0], axes[1, 1], df_gaps_d),
        (m_col, 'Mean BP (mmHg)', 0, 200, axes[2, 0], axes[2, 1], df_gaps_m)
    ]
    
    for col_name, ylabel, ymin, ymax, ax_left, ax_right, df_gap in pressures:
        title_name = ylabel.split(' ')[0]
        
        # LEFT COLUMN: MARKED (raw with red X on artifacts)
        ax_left.plot(df_raw['datetime'], df_raw[col_name], '-', linewidth=0.6, alpha=0.7,
                     color='blue', label='Raw signal')
        if len(original_removed) > 0:
            ax_left.scatter(original_removed['datetime'], original_removed[col_name], 
                           s=10, marker='x', color='red', alpha=0.9, linewidth=1.2, 
                           label=f'Artifacts ({len(original_removed)})')
        ax_left.set_ylabel(ylabel, fontsize=10)
        ax_left.set_title(f'{title_name} - MARKED (red X = artifact locations)', fontsize=11)
        ax_left.legend(loc='upper right', fontsize=8)
        ax_left.grid(True, alpha=0.3)
        ax_left.set_ylim(ymin, ymax)
        
        # RIGHT COLUMN: CLEANSED (complete gaps, no residue)
        ax_right.plot(df_gap['datetime'], df_gap[col_name], '-', linewidth=0.8, alpha=0.9,
                      color='green', label='Clean signal (artifacts removed)')
        ax_right.set_ylabel(ylabel, fontsize=10)
        ax_right.set_title(f'{title_name} - CLEANSED (complete gaps, no residue)', fontsize=11)
        ax_right.legend(loc='upper right', fontsize=8)
        ax_right.grid(True, alpha=0.3)
        ax_right.set_ylim(ymin, ymax)
    
    # Format x-axis for all subplots
    for i in range(3):
        for j in range(2):
            axes[i, j].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            axes[i, j].xaxis.set_major_locator(mdates.MinuteLocator(interval=tick_interval))
            axes[i, j].tick_params(axis='x', rotation=45, labelsize=8)
    
    axes[2, 0].set_xlabel('Time (hours:minutes)', fontsize=10)
    axes[2, 1].set_xlabel('Time (hours:minutes)', fontsize=10)
    
    # Statistics
    original_removed_count = len(original_removed)
    expanded_removed_count = len(df_raw) - len(df_sharp_clean)
    total_count = len(df_raw)
    original_clean_count = len(df_clean_original)
    expanded_clean_count = len(df_sharp_clean)
    
    stats_text = (f'Original SAI: {original_clean_count}/{total_count} clean ({original_removed_count} artifacts) | '
                  f'Expanded: {expanded_clean_count}/{total_count} clean (+{expanded_removed_count - original_removed_count} neighbors removed)')
    fig.suptitle(f'{patient_name} - {data_type}\n{stats_text}', fontsize=10, y=0.98)
    
    plt.tight_layout()
    output_file = output_dir / f"{patient_name}_{data_type.lower()}_sai.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return original_removed_count, expanded_removed_count, total_count, expanded_clean_count


def main():
    print("=" * 80)
    print("SHARP SAI visualization - Invasive and Non-invasive")
    print(f"Artifact extension window: {EXTENSION_BEFORE} beats before, {EXTENSION_AFTER} beats after")
    print("=" * 80)
    
    # Process Invasive data
    print("\n" + "-" * 40)
    print("Processing INVASIVE data")
    print("-" * 40)
    
    invasive_files = list(INVASIVE_INPUT_DIR.glob("*_invasive.csv"))
    print(f"\nFound {len(invasive_files)} invasive files\n")
    
    inv_processed = 0
    for inv_file in invasive_files:
        patient_name = inv_file.stem.replace("_invasive", "")
        print(f"Processing: {patient_name}...", end=' ', flush=True)
        
        try:
            orig, expanded, total, clean = plot_patient(
                patient_name, inv_file, INVASIVE_OUTPUT_DIR, is_invasive=True
            )
            inv_processed += 1
            print(f"done - {orig} artifacts -> {expanded} removed")
        except Exception as e:
            print(f"ERROR: {str(e)[:80]}")
    
    # Process Non-invasive data
    print("\n" + "-" * 40)
    print("Processing NON-INVASIVE data")
    print("-" * 40)
    
    noninvasive_files = list(NONINVASIVE_INPUT_DIR.glob("*_noninvasive.csv"))
    print(f"\nFound {len(noninvasive_files)} non-invasive files\n")
    
    noninv_processed = 0
    for noninv_file in noninvasive_files:
        patient_name = noninv_file.stem.replace("_noninvasive", "")
        print(f"Processing: {patient_name}...", end=' ', flush=True)
        
        try:
            orig, expanded, total, clean = plot_patient(
                patient_name, noninv_file, NONINVASIVE_OUTPUT_DIR, is_invasive=False
            )
            noninv_processed += 1
            print(f"done - {orig} artifacts -> {expanded} removed")
        except Exception as e:
            print(f"ERROR: {str(e)[:80]}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Invasive graphs: {inv_processed} patients processed")
    print(f"Non-invasive graphs: {noninv_processed} patients processed")
    print(f"\nOutput folders:")
    print(f"  Invasive graphs: {INVASIVE_OUTPUT_DIR}")
    print(f"  Non-invasive graphs: {NONINVASIVE_OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()