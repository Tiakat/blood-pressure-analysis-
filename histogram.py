"""
Histogram of gradients (MAP - NBP) for all patients combined and per patient
Gradient = Invasive MAP - Non-invasive MAP
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

# Input paths
INVASIVE_INPUT_DIR = Path(r"C:\Users\katia\Desktop\min data\work\data\Patients-infinity\filtered\operation-csv-invasive\aftersai")
NONINVASIVE_INPUT_DIR = Path(r"C:\Users\katia\Desktop\min data\work\data\Patients-infinity\filtered\operation-csv-none-invasive")
OUTPUT_DIR = Path(r"C:\Users\katia\Desktop\min data\work\data\Patients-infinity\filtered\gradient")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Column names
INV_M_COL = 'ART M (mm(hg)^^ISO+)'
NONINV_M_COL = 'NBP M (mm(hg)^^ISO+)'

TIME_TOLERANCE = 1  # minute


def time_to_minutes(time_str):
    """Convert HH:MM:SS to minutes since start of day"""
    try:
        h, m, s = map(int, str(time_str).split(':'))
        return h * 60 + m + s / 60
    except:
        return np.nan


def filter_noninvasive_by_time_and_value(df_noninv, patient_name):
    """Remove specific non-invasive values based on time ranges and thresholds"""
    if df_noninv is None or len(df_noninv) == 0:
        return df_noninv
    
    df = df_noninv.copy()
    
    if "Patient 29" in patient_name or "Patient_29" in patient_name:
        mask_time = (df['time'] >= '08:45:00') & (df['time'] <= '08:50:00')
        mask_value = (df[NONINV_M_COL] < 45)
        df.loc[mask_time & mask_value, NONINV_M_COL] = np.nan
    
    if "Patient 37" in patient_name or "Patient_37" in patient_name:
        mask1 = (df['time'] >= '11:37:00') & (df['time'] <= '11:52:00') & (df[NONINV_M_COL] < 70)
        df.loc[mask1, NONINV_M_COL] = np.nan
        mask2 = (df['time'] >= '12:07:00') & (df['time'] <= '12:22:00') & (df[NONINV_M_COL] > 80)
        df.loc[mask2, NONINV_M_COL] = np.nan
        mask3 = (df['time'] >= '09:22:00') & (df['time'] <= '09:37:00') & (df[NONINV_M_COL] > 100)
        df.loc[mask3, NONINV_M_COL] = np.nan
        mask4 = (df['time'] >= '13:07:00') & (df['time'] <= '13:10:00') & (df[NONINV_M_COL] > 100)
        df.loc[mask4, NONINV_M_COL] = np.nan
    
    return df


def filter_invasive_line(df_invasive, patient_name):
    """Create gaps in invasive line by removing specific time ranges"""
    if df_invasive is None or len(df_invasive) == 0:
        return df_invasive
    
    df = df_invasive.copy()
    
    if "Patient 49" in patient_name or "Patient_49" in patient_name:
        mask = (df['time'] >= '09:40:00') & (df['time'] <= '09:50:00')
        df.loc[mask, INV_M_COL] = np.nan
    
    if "PROMISES 51" in patient_name:
        mask = (df['time'] >= '09:15:00') & (df['time'] <= '09:25:00')
        df.loc[mask, INV_M_COL] = np.nan
    
    return df


def calculate_gradients(df_invasive, df_noninv):
    """Calculate gradient = MAP - NBP for each non-invasive point"""
    if df_invasive is None or df_noninv is None or len(df_invasive) == 0 or len(df_noninv) == 0:
        return None, None
    
    df_invasive['minutes'] = df_invasive['time'].apply(time_to_minutes)
    df_noninv['minutes'] = df_noninv['time'].apply(time_to_minutes)
    
    df_invasive = df_invasive.dropna(subset=['minutes', INV_M_COL])
    df_noninv = df_noninv.dropna(subset=['minutes', NONINV_M_COL])
    
    if len(df_invasive) == 0 or len(df_noninv) == 0:
        return None, None
    
    gradients = []
    
    for _, pni_row in df_noninv.iterrows():
        t_pni = pni_row['minutes']
        val_pni = pni_row[NONINV_M_COL]
        
        idx = (df_invasive['minutes'] - t_pni).abs().idxmin()
        t_inv = df_invasive.loc[idx, 'minutes']
        val_inv = df_invasive.loc[idx, INV_M_COL]
        
        if abs(t_inv - t_pni) <= TIME_TOLERANCE:
            gradient = val_inv - val_pni
            gradients.append(gradient)
    
    if len(gradients) == 0:
        return None, None
    
    return np.array(gradients), len(gradients)


def create_histogram_all_patients(all_gradients, output_dir):
    """Create histogram for all patients combined"""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Calculate statistics
    mean_grad = np.mean(all_gradients)
    median_grad = np.median(all_gradients)
    std_grad = np.std(all_gradients)
    
    # Create histogram
    n, bins, patches = ax.hist(all_gradients, bins=50, range=(-60, 60), 
                                color='steelblue', edgecolor='black', alpha=0.7)
    
    # Add vertical lines for mean and median
    ax.axvline(mean_grad, color='red', linestyle='-', linewidth=2, label=f'Moyenne = {mean_grad:.1f} mmHg')
    ax.axvline(median_grad, color='green', linestyle='--', linewidth=2, label=f'Médiane = {median_grad:.1f} mmHg')
    ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    # Add threshold lines at -10 and +10
    ax.axvline(-10, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='Seuil ±10 mmHg')
    ax.axvline(10, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    
    # Labels and title
    ax.set_xlabel('Gradient (MAP invasif - NBP non-invasif) [mmHg]', fontsize=12)
    ax.set_ylabel('Nombre de mesures', fontsize=12)
    ax.set_title(f'Distribution des gradients - Tous les patients confondus\n(n = {len(all_gradients)} mesures)', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add text box with statistics
    stats_text = f'Statistiques:\n'
    stats_text += f'Moyenne: {mean_grad:.1f} mmHg\n'
    stats_text += f'Médiane: {median_grad:.1f} mmHg\n'
    stats_text += f'Écart-type: {std_grad:.1f} mmHg\n'
    stats_text += f'Minimum: {np.min(all_gradients):.1f} mmHg\n'
    stats_text += f'Maximum: {np.max(all_gradients):.1f} mmHg\n'
    stats_text += f'Q1: {np.percentile(all_gradients, 25):.1f} mmHg\n'
    stats_text += f'Q3: {np.percentile(all_gradients, 75):.1f} mmHg'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    output_file = output_dir / "histogram_all_patients.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return mean_grad, median_grad, std_grad


def create_histogram_per_patient(patient_name, gradients, output_dir):
    """Create individual histogram for one patient"""
    
    if gradients is None or len(gradients) < 3:
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Calculate statistics
    mean_grad = np.mean(gradients)
    median_grad = np.median(gradients)
    std_grad = np.std(gradients)
    
    # Create histogram
    n, bins, patches = ax.hist(gradients, bins=30, range=(-60, 60), 
                                color='steelblue', edgecolor='black', alpha=0.7)
    
    # Color bins based on position relative to thresholds
    for i, (left, right) in enumerate(zip(bins[:-1], bins[1:])):
        if left < -10:
            patches[i].set_facecolor('darkred')
            patches[i].set_alpha(0.6)
        elif left > 10:
            patches[i].set_facecolor('darkred')
            patches[i].set_alpha(0.6)
        else:
            patches[i].set_facecolor('steelblue')
            patches[i].set_alpha(0.7)
    
    # Add vertical lines
    ax.axvline(mean_grad, color='red', linestyle='-', linewidth=2, label=f'Moyenne = {mean_grad:.1f} mmHg')
    ax.axvline(median_grad, color='green', linestyle='--', linewidth=2, label=f'Médiane = {median_grad:.1f} mmHg')
    ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.axvline(-10, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='Seuil ±10 mmHg')
    ax.axvline(10, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    
    # Labels and title
    ax.set_xlabel('Gradient (MAP invasif - NBP non-invasif) [mmHg]', fontsize=11)
    ax.set_ylabel('Nombre de mesures', fontsize=11)
    ax.set_title(f'{patient_name} - Distribution des gradients\n(n = {len(gradients)} mesures)', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Add text box with statistics
    pct_outside = (np.sum(np.abs(gradients) > 10) / len(gradients)) * 100
    pct_positive = (np.sum(gradients > 10) / len(gradients)) * 100
    pct_negative = (np.sum(gradients < -10) / len(gradients)) * 100
    
    stats_text = f'Statistiques:\n'
    stats_text += f'Moyenne: {mean_grad:.1f} mmHg\n'
    stats_text += f'Médiane: {median_grad:.1f} mmHg\n'
    stats_text += f'Écart-type: {std_grad:.1f} mmHg\n'
    stats_text += f'|gradient| > 10: {pct_outside:.1f}%\n'
    stats_text += f'  Gradient > 10: {pct_positive:.1f}%\n'
    stats_text += f'  Gradient < -10: {pct_negative:.1f}%'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    output_file = output_dir / f"{patient_name}_histogram.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)


def create_summary_histogram(all_patient_stats, output_dir):
    """Create histogram of patient percentages"""
    
    percentages = [s['pct_outside'] for s in all_patient_stats if s is not None]
    
    if len(percentages) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    n, bins, patches = ax.hist(percentages, bins=20, range=(0, 100), 
                                color='steelblue', edgecolor='black', alpha=0.7)
    
    ax.set_xlabel('Pourcentage de temps avec |gradient| > 10 mmHg (%)', fontsize=12)
    ax.set_ylabel('Nombre de patients', fontsize=12)
    ax.set_title(f'Distribution du temps passé en gradient significatif\n(n = {len(percentages)} patients)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_pct = np.mean(percentages)
    median_pct = np.median(percentages)
    
    ax.axvline(mean_pct, color='red', linestyle='-', linewidth=2, label=f'Moyenne = {mean_pct:.1f}%')
    ax.axvline(median_pct, color='green', linestyle='--', linewidth=2, label=f'Médiane = {median_pct:.1f}%')
    ax.legend(loc='upper right', fontsize=10)
    
    stats_text = f'Statistiques:\n'
    stats_text += f'Moyenne: {mean_pct:.1f}%\n'
    stats_text += f'Médiane: {median_pct:.1f}%\n'
    stats_text += f'Écart-type: {np.std(percentages):.1f}%\n'
    stats_text += f'Minimum: {np.min(percentages):.1f}%\n'
    stats_text += f'Maximum: {np.max(percentages):.1f}%'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    output_file = output_dir / "histogram_patient_percentages.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    print("=" * 80)
    print("HISTOGRAMME DES GRADIENTS MAP - NBP")
    print("Gradient = MAP invasif - NBP non-invasif")
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
    
    # Create mapping
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
    
    all_gradients = []
    all_patient_stats = []
    
    for base_name, files in patient_map.items():
        print(f"Processing: {base_name}...", end=' ', flush=True)
        
        try:
            invasive_file = files['invasive']
            noninvasive_file = files['noninvasive']
            
            if invasive_file is None or noninvasive_file is None:
                print(f"SKIP - missing files")
                continue
            
            df_invasive = pd.read_csv(invasive_file)
            df_noninv = pd.read_csv(noninvasive_file)
            
            df_noninv = filter_noninvasive_by_time_and_value(df_noninv, base_name)
            df_invasive = filter_invasive_line(df_invasive, base_name)
            
            gradients, n_points = calculate_gradients(df_invasive, df_noninv)
            
            if gradients is not None and len(gradients) > 0:
                all_gradients.extend(gradients)
                
                pct_outside = (np.sum(np.abs(gradients) > 10) / len(gradients)) * 100
                all_patient_stats.append({
                    'patient': base_name,
                    'pct_outside': pct_outside,
                    'n_points': len(gradients)
                })
                
                # Create individual histogram
                create_histogram_per_patient(base_name, gradients, OUTPUT_DIR)
                
                print(f"OK - {len(gradients)} mesures")
            else:
                print(f"SKIP - pas de gradients")
            
        except Exception as e:
            print(f"ERROR: {str(e)[:80]}")
    
    if len(all_gradients) > 0:
        # Create combined histogram for all patients
        print("\n" + "=" * 80)
        print("CRÉATION DES HISTOGRAMMES")
        print("=" * 80)
        
        mean_grad, median_grad, std_grad = create_histogram_all_patients(np.array(all_gradients), OUTPUT_DIR)
        create_summary_histogram(all_patient_stats, OUTPUT_DIR)
        
        print(f"\nHistogramme global: {len(all_gradients)} mesures")
        print(f"  Moyenne: {mean_grad:.1f} mmHg")
        print(f"  Médiane: {median_grad:.1f} mmHg")
        print(f"  Écart-type: {std_grad:.1f} mmHg")
        print(f"  |gradient| > 10 mmHg: {(np.sum(np.abs(all_gradients) > 10) / len(all_gradients) * 100):.1f}%")
        
        print(f"\nImages sauvegardées dans: {OUTPUT_DIR}")
        print("  - histogram_all_patients.png (tous les patients confondus)")
        print("  - histogram_patient_percentages.png (distribution des pourcentages)")
        print("  - *_histogram.png (un par patient)")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()