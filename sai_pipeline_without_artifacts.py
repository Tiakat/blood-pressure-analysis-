"""
SHARP SAI visualization - Complete artifact region removal
One PNG per patient with 6 graphs: MARKED (red X) and CLEANSED (gaps)
All three pressures: Systolic, Diastolic, Mean
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path(r"C:\Users\katia\Desktop\min data\work\data\Patients-infinity\filtered")
SAI_OUTPUT_DIR = OUTPUT_DIR / "sai_filtered"
VIZ_OUTPUT_DIR = OUTPUT_DIR / "sai_visualizations_final"
VIZ_OUTPUT_DIR.mkdir(exist_ok=True)

EXTENSION_BEFORE = 10
EXTENSION_AFTER = 10


def time_to_datetime(time_str, base_date="2024-01-01"):
    try:
        if '.' in str(time_str):
            time_str = str(time_str).split('.')[0]
        return datetime.strptime(f"{base_date} {time_str}", "%Y-%m-%d %H:%M:%S")
    except:
        return None


def expand_artifact_regions_complete(df_raw, df_clean, extension_before, extension_after):
    """
    Expand artifact detection to completely remove entire artifact regions.
    When an artifact is detected, removes all beats within a window around it.
    """
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


def create_complete_gaps(df_raw, df_clean_sharp, pressure_col):
    """Create version where all artifact regions are replaced with NaN (complete gaps)"""
    df_gaps = df_raw.copy()
    clean_times = set(df_clean_sharp['datetime'].values)
    
    for idx in df_gaps.index:
        if df_gaps.loc[idx, 'datetime'] not in clean_times:
            df_gaps.loc[idx, pressure_col] = np.nan
    
    return df_gaps


def plot_single_patient(patient_name, raw_file, clean_file, output_dir):
    """
    Create one PNG per patient with 6 graphs:
    Left column: MARKED (raw data with red X on artifacts)
    Right column: CLEANSED (clean data with complete gaps, no residue)
    """
    df_raw = pd.read_csv(raw_file)
    df_clean_original = pd.read_csv(clean_file)

    df_raw['datetime'] = df_raw['time'].apply(time_to_datetime)
    df_clean_original['datetime'] = df_clean_original['time'].apply(time_to_datetime)

    df_raw = df_raw.dropna(subset=['datetime'])
    df_clean_original = df_clean_original.dropna(subset=['datetime'])

    if len(df_raw) == 0:
        return 0, 0, 0, 0

    # Expand artifact regions to completely remove entire artifact zones
    df_sharp_clean = expand_artifact_regions_complete(df_raw, df_clean_original, EXTENSION_BEFORE, EXTENSION_AFTER)
    
    # Find original artifact locations
    original_removed = df_raw[~df_raw['datetime'].isin(df_clean_original['datetime'])].copy()
    
    # Create gap data for each pressure
    df_gaps_s = create_complete_gaps(df_raw, df_sharp_clean, 'ART_S')
    df_gaps_d = create_complete_gaps(df_raw, df_sharp_clean, 'ART_D')
    df_gaps_m = create_complete_gaps(df_raw, df_sharp_clean, 'ART_M')
    
    # Find time range for x-axis
    start_time = df_raw['datetime'].min()
    end_time = df_raw['datetime'].max()
    duration_minutes = (end_time - start_time).total_seconds() / 60
    
    # Determine x-axis tick interval based on duration
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
        ('ART_S', 'Systolic BP (mmHg)', 0, 250, axes[0, 0], axes[0, 1], df_gaps_s),
        ('ART_D', 'Diastolic BP (mmHg)', 0, 150, axes[1, 0], axes[1, 1], df_gaps_d),
        ('ART_M', 'Mean BP (mmHg)', 0, 200, axes[2, 0], axes[2, 1], df_gaps_m)
    ]
    
    for col_name, ylabel, ymin, ymax, ax_left, ax_right, df_gap in pressures:
        title_name = 'Systolic' if col_name == 'ART_S' else 'Diastolic' if col_name == 'ART_D' else 'Mean'
        
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
                  f'Expanded (window={EXTENSION_BEFORE}/{EXTENSION_AFTER}): {expanded_clean_count}/{total_count} clean (+{expanded_removed_count - original_removed_count} neighbors removed)')
    fig.suptitle(f'{patient_name}\n{stats_text}', fontsize=10, y=0.98)
    
    plt.tight_layout()
    output_file = output_dir / f"{patient_name}_sai_final.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return len(original_removed), len(expanded_removed), len(df_raw), len(df_sharp_clean)


def main():
    print("=" * 80)
    print("SHARP SAI visualization - Complete artifact region removal")
    print(f"Artifact extension window: {EXTENSION_BEFORE} beats before, {EXTENSION_AFTER} beats after")
    print("=" * 80)

    # Get all operation files
    op_files = list(OUTPUT_DIR.glob("*_operation.csv"))
    op_files = [f for f in op_files if "sai_filtered" not in str(f)]

    print(f"Found {len(op_files)} operation files")
    
    # List available clean files for debugging
    clean_files = list(SAI_OUTPUT_DIR.glob("*_sai_filtered.csv"))
    print(f"Found {len(clean_files)} clean files in sai_filtered folder")

    all_results = []
    processed = 0
    skipped = 0

    for op_file in op_files:
        patient_name = op_file.stem.replace("_operation", "")
        
        # Find matching clean file
        clean_file = SAI_OUTPUT_DIR / f"{op_file.stem}_sai_filtered.csv"
        
        if not clean_file.exists():
            print(f"SKIP: no clean file for {patient_name}")
            skipped += 1
            continue

        print(f"Processing {patient_name}...", end=' ', flush=True)
        
        try:
            orig_removed, expanded_removed, total_count, clean_count = plot_single_patient(
                patient_name, op_file, clean_file, VIZ_OUTPUT_DIR
            )
            all_results.append({
                'patient': patient_name,
                'total': total_count,
                'original_removed': orig_removed,
                'expanded_removed': expanded_removed,
                'original_pct': orig_removed/total_count*100 if total_count > 0 else 0,
                'expanded_pct': expanded_removed/total_count*100 if total_count > 0 else 0
            })
            processed += 1
            print(f"done (window={EXTENSION_BEFORE}/{EXTENSION_AFTER}: {orig_removed} -> {expanded_removed} removed)")
        except Exception as e:
            print(f"ERROR: {str(e)[:50]}")
            skipped += 1

    print("\n" + "=" * 80)
    print(f"SUMMARY: {processed} patients processed, {skipped} skipped")
    print("=" * 80)

    for r in all_results:
        print(f"  {r['patient']}: Original {r['original_pct']:.1f}% -> Expanded {r['expanded_pct']:.1f}% removed")

    print("\n" + "=" * 80)
    print(f"Output folder: {VIZ_OUTPUT_DIR}")
    print("Each PNG shows 6 graphs (3 pressures x 2 columns):")
    print("  - Left column: MARKED (blue line with red X on artifact locations)")
    print("  - Right column: CLEANSED (green line with complete gaps, no residue)")
    print(f"  - Extension window: {EXTENSION_BEFORE} beats before, {EXTENSION_AFTER} beats after each artifact")
    print("=" * 80)


if __name__ == "__main__":
    main()