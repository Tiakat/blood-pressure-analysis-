"""
SHARP SAI visualization - Complete artifact region removal
One PNG per patient with 6 graphs: MARKED (red X) and CLEANSED (gaps)
All three pressures: Systolic, Diastolic, Mean
Avec statistiques de performance pour comparaison avec Khan
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION - CHEMINS CORRIGÉS
# ============================================================
BEFORE_SAI_DIR = Path(r"C:\Users\katia\Desktop\min data\work\data\Patients-infinity\filtered\invasive\operation-csv-invasive\beforesai")
AFTER_SAI_DIR = Path(r"C:\Users\katia\Desktop\min data\work\data\Patients-infinity\filtered\invasive\operation-csv-invasive\aftersai")
VIZ_OUTPUT_DIR = Path(r"C:\Users\katia\Desktop\min data\work\data\Patients-infinity\filtered\sai_visualizations_final")

VIZ_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXTENSION_BEFORE = 10
EXTENSION_AFTER = 10

# ============================================================
# PARAMÈTRES KHAN POUR COMPARAISON
# ============================================================
MAP_MIN = 30
MAP_MAX = 200
MAP_DELTA = 7.8


def time_to_datetime(time_str, base_date="2024-01-01"):
    try:
        if '.' in str(time_str):
            time_str = str(time_str).split('.')[0]
        return datetime.strptime(f"{base_date} {time_str}", "%Y-%m-%d %H:%M:%S")
    except:
        return None


def get_pressure_column(df):
    possible_cols = ['ART M (mm(hg)^^ISO+)', 'ART_M', 'ART M', 'MAP', 'Mean']
    for col in possible_cols:
        if col in df.columns:
            return col
    return None


def detect_artifacts_khan(values):
    """Détection selon Khan (2022) sur les valeurs MAP"""
    n = len(values)
    mask = np.zeros(n, dtype=bool)
    
    for i in range(n):
        if not np.isnan(values[i]):
            if values[i] < MAP_MIN or values[i] > MAP_MAX:
                mask[i] = True
    
    for i in range(1, n):
        if not np.isnan(values[i]) and not np.isnan(values[i-1]):
            if abs(values[i] - values[i-1]) > MAP_DELTA:
                mask[i] = True
    
    return mask


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


def calculate_sai_performance(patient_name, df_raw, df_clean_original):
    """
    Calcule les métriques de performance de SAI par rapport à Khan
    """
    map_col = get_pressure_column(df_raw)
    if map_col is None:
        return None
    
    values = df_raw[map_col].values
    n = len(values)
    
    # 1. Détection Khan
    mask_khan = detect_artifacts_khan(values)
    n_khan = mask_khan.sum()
    
    # 2. Détection SAI (NaN dans clean_original)
    mask_sai = np.zeros(n, dtype=bool)
    clean_times = set(df_clean_original['datetime'].values)
    for i, row in df_raw.iterrows():
        if row['datetime'] not in clean_times:
            mask_sai[i] = True
    n_sai = mask_sai.sum()
    
    # 3. Matrice de confusion
    tp = np.sum(mask_khan & mask_sai)
    fp = np.sum(mask_khan & ~mask_sai)
    fn = np.sum(~mask_khan & mask_sai)
    tn = np.sum(~mask_khan & ~mask_sai)
    
    # 4. Métriques
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    accuracy = (tp + tn) / n if n > 0 else 0
    
    return {
        'patient': patient_name,
        'n_total': n,
        'n_sai': n_sai,
        'pct_sai': n_sai / n * 100 if n > 0 else 0,
        'n_khan': n_khan,
        'pct_khan': n_khan / n * 100 if n > 0 else 0,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1': f1,
        'accuracy': accuracy
    }


def plot_single_patient(patient_name, raw_file, clean_file, output_dir):
    """
    Create one PNG per patient with 6 graphs:
    Left column: MARKED (raw data with red X on artifacts)
    Right column: CLEANSED (clean data with complete gaps, no residue)
    """
    df_raw = pd.read_csv(raw_file)
    df_clean_original = pd.read_csv(clean_file)

    # Trouver la colonne de temps
    if 'time' in df_raw.columns:
        df_raw['datetime'] = df_raw['time'].apply(time_to_datetime)
        df_clean_original['datetime'] = df_clean_original['time'].apply(time_to_datetime)
    elif 'OBSERVATION_DATETIME' in df_raw.columns:
        # À adapter selon le format
        pass

    df_raw = df_raw.dropna(subset=['datetime'])
    df_clean_original = df_clean_original.dropna(subset=['datetime'])

    if len(df_raw) == 0:
        return 0, 0, 0, 0, None

    # Expand artifact regions
    df_sharp_clean = expand_artifact_regions_complete(df_raw, df_clean_original, EXTENSION_BEFORE, EXTENSION_AFTER)
    
    # Find original artifact locations
    original_removed = df_raw[~df_raw['datetime'].isin(df_clean_original['datetime'])].copy()
    
    # Get pressure columns
    map_col = get_pressure_column(df_raw)
    if map_col is None:
        return 0, 0, 0, 0, None
    
    # Créer les colonnes pour chaque pression
    for col in ['ART_S', 'ART_D', 'ART_M', 'ART_S (mm(hg)^^ISO+)', 'ART_D (mm(hg)^^ISO+)', 'ART_M (mm(hg)^^ISO+)']:
        if col in df_raw.columns:
            # Déterminer le nom simplifié
            if 'S' in col:
                s_col = col
            elif 'D' in col:
                d_col = col
            elif 'M' in col:
                m_col = col
    
    # Si les colonnes standard ne sont pas trouvées, utiliser map_col pour toutes
    if 's_col' not in locals():
        s_col = d_col = m_col = map_col
    
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
        (s_col, 'Systolic BP (mmHg)', 0, 250, axes[0, 0], axes[0, 1], df_gaps_s, 'Systolic'),
        (d_col, 'Diastolic BP (mmHg)', 0, 150, axes[1, 0], axes[1, 1], df_gaps_d, 'Diastolic'),
        (m_col, 'Mean BP (mmHg)', 0, 200, axes[2, 0], axes[2, 1], df_gaps_m, 'Mean')
    ]
    
    for col_name, ylabel, ymin, ymax, ax_left, ax_right, df_gap, title_name in pressures:
        # LEFT COLUMN: MARKED
        ax_left.plot(df_raw['datetime'], df_raw[col_name], '-', linewidth=0.6, alpha=0.7,
                     color='blue', label='Raw signal')
        if len(original_removed) > 0 and col_name in original_removed.columns:
            ax_left.scatter(original_removed['datetime'], original_removed[col_name], 
                           s=10, marker='x', color='red', alpha=0.9, linewidth=1.2, 
                           label=f'Artifacts ({len(original_removed)})')
        ax_left.set_ylabel(ylabel, fontsize=10)
        ax_left.set_title(f'{title_name} - MARKED (red X = artifact locations)', fontsize=11)
        ax_left.legend(loc='upper right', fontsize=8)
        ax_left.grid(True, alpha=0.3)
        ax_left.set_ylim(ymin, ymax)
        
        # RIGHT COLUMN: CLEANSED
        ax_right.plot(df_gap['datetime'], df_gap[col_name], '-', linewidth=0.8, alpha=0.9,
                      color='green', label='Clean signal (artifacts removed)')
        ax_right.set_ylabel(ylabel, fontsize=10)
        ax_right.set_title(f'{title_name} - CLEANSED (complete gaps, no residue)', fontsize=11)
        ax_right.legend(loc='upper right', fontsize=8)
        ax_right.grid(True, alpha=0.3)
        ax_right.set_ylim(ymin, ymax)
    
    # Format x-axis
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
    
    # Performance
    perf = calculate_sai_performance(patient_name, df_raw, df_clean_original)
    
    stats_text = (f'Original SAI: {original_clean_count}/{total_count} clean ({original_removed_count} artifacts) | '
                  f'Expanded: {expanded_clean_count}/{total_count} clean (+{expanded_removed_count - original_removed_count} neighbors)')
    
    if perf:
        stats_text += f'\nSAI vs Khan: Sens={perf["sensitivity"]:.3f} | Spec={perf["specificity"]:.3f} | F1={perf["f1"]:.3f} | Acc={perf["accuracy"]:.3f}'
    
    fig.suptitle(f'{patient_name}\n{stats_text}', fontsize=10, y=0.98)
    
    plt.tight_layout()
    output_file = output_dir / f"{patient_name}_sai_final.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return original_removed_count, expanded_removed_count, total_count, expanded_clean_count, perf


def main():
    print("=" * 80)
    print("SHARP SAI visualization - Complete artifact region removal")
    print(f"Artifact extension window: {EXTENSION_BEFORE} beats before, {EXTENSION_AFTER} beats after")
    print("=" * 80)
    print(f"Fichiers bruts (beforesai): {BEFORE_SAI_DIR}")
    print(f"Fichiers SAI (aftersai): {AFTER_SAI_DIR}")
    print("=" * 80)

    # Récupérer les fichiers bruts
    raw_files = list(BEFORE_SAI_DIR.glob("*_invasive.csv"))
    print(f"Fichiers bruts trouvés: {len(raw_files)}")
    
    # Récupérer les fichiers SAI
    clean_files = list(AFTER_SAI_DIR.glob("*_invasive_cleaned.csv"))
    print(f"Fichiers SAI trouvés: {len(clean_files)}")

    all_results = []
    performance_data = []
    processed = 0
    skipped = 0

    for raw_file in raw_files:
        # Extraire le nom du patient
        patient_name = raw_file.stem.replace("_invasive", "")
        
        # Trouver le fichier SAI correspondant
        clean_file = AFTER_SAI_DIR / f"{patient_name}_invasive_cleaned.csv"
        
        if not clean_file.exists():
            print(f"SKIP: no clean file for {patient_name}")
            skipped += 1
            continue

        print(f"Processing {patient_name}...", end=' ', flush=True)
        
        try:
            orig_removed, expanded_removed, total_count, clean_count, perf = plot_single_patient(
                patient_name, raw_file, clean_file, VIZ_OUTPUT_DIR
            )
            all_results.append({
                'patient': patient_name,
                'total': total_count,
                'original_removed': orig_removed,
                'expanded_removed': expanded_removed,
                'original_pct': orig_removed/total_count*100 if total_count > 0 else 0,
                'expanded_pct': expanded_removed/total_count*100 if total_count > 0 else 0
            })
            
            if perf:
                performance_data.append(perf)
            
            processed += 1
            print(f"done: {orig_removed} -> {expanded_removed} removed")
            if perf:
                print(f"     SAI vs Khan: Sens={perf['sensitivity']:.3f}, Spec={perf['specificity']:.3f}, F1={perf['f1']:.3f}")
        except Exception as e:
            print(f"ERROR: {str(e)[:80]}")
            skipped += 1

    print("\n" + "=" * 80)
    print(f"SUMMARY: {processed} patients processed, {skipped} skipped")
    print("=" * 80)

    for r in all_results:
        print(f"  {r['patient']}: Original {r['original_pct']:.1f}% -> Expanded {r['expanded_pct']:.1f}% removed")

    # Performance SAI vs Khan
    if performance_data:
        print("\n" + "=" * 80)
        print("PERFORMANCE SAI vs KHAN")
        print("=" * 80)
        
        df_perf = pd.DataFrame(performance_data)
        df_perf.to_csv(VIZ_OUTPUT_DIR / 'sai_vs_khan_performance.csv', index=False)
        
        total_points = df_perf['n_total'].sum()
        total_sai = df_perf['n_sai'].sum()
        total_khan = df_perf['n_khan'].sum()
        total_tp = df_perf['tp'].sum()
        total_fp = df_perf['fp'].sum()
        total_fn = df_perf['fn'].sum()
        total_tn = df_perf['tn'].sum()
        
        print(f"Total points: {total_points:,}")
        print(f"Artéfacts SAI: {total_sai:,} ({total_sai/total_points*100:.2f}%)")
        print(f"Artéfacts Khan: {total_khan:,} ({total_khan/total_points*100:.2f}%)")
        print("")
        print("Matrice de confusion (SAI vs Khan):")
        print(f"  TP: {total_tp:,}")
        print(f"  FP: {total_fp:,}")
        print(f"  FN: {total_fn:,}")
        print(f"  TN: {total_tn:,}")
        print("")
        print("Métriques globales:")
        
        global_sensitivity = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        global_specificity = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0
        global_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        global_f1 = 2 * (global_precision * global_sensitivity) / (global_precision + global_sensitivity) if (global_precision + global_sensitivity) > 0 else 0
        global_accuracy = (total_tp + total_tn) / total_points if total_points > 0 else 0
        
        print(f"  Sensibilité: {global_sensitivity:.4f}")
        print(f"  Spécificité: {global_specificity:.4f}")
        print(f"  Précision: {global_precision:.4f}")
        print(f"  F1-score: {global_f1:.4f}")
        print(f"  Accuracy: {global_accuracy:.4f}")

    print("\n" + "=" * 80)
    print(f"Output folder: {VIZ_OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()