"""
COMPARAISON STATISTIQUE KHAN vs SAI
- Khan: détection sur données brutes avec bornes 30-200 + delta 7.8
- SAI: détection sur données aftersai (référence)
- Tableau comparatif détaillé par patient
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
BEFORE_SAI_DIR = Path(r"C:\Users\katia\Desktop\min data\work\data\Patients-infinity\filtered\invasive\operation-csv-invasive\beforesai")
AFTER_SAI_DIR = Path(r"C:\Users\katia\Desktop\min data\work\data\Patients-infinity\filtered\invasive\operation-csv-invasive\aftersai")
OUTPUT_DIR = Path(r"C:\Users\katia\Desktop\min data\work\data\Patients-infinity\filtered\khan_vs_sai_stats")

GRAPHS_DIR = OUTPUT_DIR / "graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

FS = 128

# ============================================================
# PARAMÈTRES KHAN
# ============================================================
MAP_MIN = 30
MAP_MAX = 200
MAP_DELTA = 7.8
EXPAND_WINDOW_BEFORE = 5
EXPAND_WINDOW_AFTER = 10

# ============================================================
# FONCTIONS
# ============================================================

def get_pressure_column(df):
    possible_cols = ['ART M (mm(hg)^^ISO+)', 'ART_M', 'ART M', 'MAP', 'Mean']
    for col in possible_cols:
        if col in df.columns:
            return col
    return None

def detect_artifacts_khan(values):
    """Détection selon Khan"""
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

def expand_artifact_window(mask, before=5, after=10):
    expanded = mask.copy()
    n = len(mask)
    artifact_indices = np.where(mask)[0]
    for idx in artifact_indices:
        start = max(0, idx - before)
        end = min(n, idx + after + 1)
        expanded[start:end] = True
    return expanded

def detect_artifacts_sai(df_before, df_after, pressure_col):
    """Détection SAI (NaN dans aftersai)"""
    if df_after is None:
        return None
    
    values_before = df_before[pressure_col].values
    values_after = df_after[pressure_col].values
    
    n_before = len(values_before)
    n_after = len(values_after)
    
    if n_after < n_before:
        values_after_padded = np.full(n_before, np.nan)
        values_after_padded[:n_after] = values_after
        return np.isnan(values_after_padded)
    elif n_after > n_before:
        return np.isnan(values_after[:n_before])
    else:
        return np.isnan(values_after)

def calculate_metrics(mask_khan, mask_sai):
    """Calculer les métriques de performance"""
    if mask_sai is None:
        return None
    
    n = min(len(mask_khan), len(mask_sai))
    khan = mask_khan[:n]
    sai = mask_sai[:n]
    
    tp = np.sum(khan & sai)
    fp = np.sum(khan & ~sai)
    fn = np.sum(~khan & sai)
    tn = np.sum(~khan & ~sai)
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1': f1
    }

def process_patient(patient_name):
    """Comparer Khan et SAI"""
    
    before_file = BEFORE_SAI_DIR / f"{patient_name}_invasive.csv"
    after_file = AFTER_SAI_DIR / f"{patient_name}_invasive_cleaned.csv"
    
    if not before_file.exists():
        return None
    
    df_before = pd.read_csv(before_file)
    df_after = pd.read_csv(after_file) if after_file.exists() else None
    
    pressure_col = get_pressure_column(df_before)
    if pressure_col is None:
        return None
    
    values = df_before[pressure_col].values
    n = len(values)
    
    # Khan
    mask_khan = detect_artifacts_khan(values)
    mask_khan_expanded = expand_artifact_window(mask_khan, EXPAND_WINDOW_BEFORE, EXPAND_WINDOW_AFTER)
    
    # SAI
    mask_sai = detect_artifacts_sai(df_before, df_after, pressure_col) if df_after is not None else None
    
    # Métriques
    metrics = calculate_metrics(mask_khan_expanded, mask_sai)
    
    return {
        'patient': patient_name,
        'n_total': n,
        'n_khan': mask_khan_expanded.sum(),
        'pct_khan': mask_khan_expanded.sum() / n * 100,
        'n_sai': mask_sai.sum() if mask_sai is not None else 0,
        'pct_sai': mask_sai.sum() / n * 100 if mask_sai is not None else 0,
        'tp': metrics['tp'] if metrics else 0,
        'fp': metrics['fp'] if metrics else 0,
        'fn': metrics['fn'] if metrics else 0,
        'tn': metrics['tn'] if metrics else 0,
        'sensitivity': metrics['sensitivity'] if metrics else 0,
        'specificity': metrics['specificity'] if metrics else 0,
        'precision': metrics['precision'] if metrics else 0,
        'f1': metrics['f1'] if metrics else 0,
        'accuracy': metrics['accuracy'] if metrics else 0
    }

def create_comparison_table(df_results):
    """Créer un tableau comparatif détaillé"""
    
    # Statistiques globales
    stats_summary = {
        'Métrique': ['Patients', 'Total points', 'Artéfacts Khan', 'Artéfacts SAI', 
                     'TP', 'FP', 'FN', 'TN',
                     'Sensibilité moyenne', 'Spécificité moyenne', 'F1 moyen', 'Accuracy moyenne'],
        'Valeur': [
            len(df_results),
            int(df_results['n_total'].sum()),
            int(df_results['n_khan'].sum()),
            int(df_results['n_sai'].sum()),
            int(df_results['tp'].sum()),
            int(df_results['fp'].sum()),
            int(df_results['fn'].sum()),
            int(df_results['tn'].sum()),
            round(df_results['sensitivity'].mean(), 4),
            round(df_results['specificity'].mean(), 4),
            round(df_results['f1'].mean(), 4),
            round(df_results['accuracy'].mean(), 4)
        ]
    }
    
    df_stats = pd.DataFrame(stats_summary)
    df_stats.to_csv(OUTPUT_DIR / 'stats_summary.csv', index=False)
    
    return df_stats

def create_graphs(df_results):
    """Créer des graphiques comparatifs"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Distribution des artéfacts
    ax1 = axes[0, 0]
    ax1.hist(df_results['pct_khan'], bins=20, alpha=0.5, label='Khan', color='red', edgecolor='black')
    ax1.hist(df_results['pct_sai'], bins=20, alpha=0.5, label='SAI', color='green', edgecolor='black')
    ax1.set_xlabel('Pourcentage d\'artéfacts (%)')
    ax1.set_ylabel('Nombre de patients')
    ax1.set_title('Distribution des artéfacts')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Boxplot comparatif
    ax2 = axes[0, 1]
    data = [df_results['pct_khan'], df_results['pct_sai']]
    ax2.boxplot(data, labels=['Khan', 'SAI'])
    ax2.set_ylabel('Pourcentage d\'artéfacts (%)')
    ax2.set_title('Boxplot des artéfacts')
    ax2.grid(True, alpha=0.3)
    
    # 3. Sensibilité et Spécificité
    ax3 = axes[1, 0]
    x = range(len(df_results))
    ax3.bar(x, df_results['sensitivity'], alpha=0.7, label='Sensibilité', color='steelblue')
    ax3.bar(x, df_results['specificity'], alpha=0.7, label='Spécificité', color='orange')
    ax3.axhline(y=0.5, color='red', linestyle='--', label='Seuil 0.5')
    ax3.set_xlabel('Patients')
    ax3.set_ylabel('Score')
    ax3.set_title('Sensibilité et Spécificité par patient')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. F1-score
    ax4 = axes[1, 1]
    ax4.bar(x, df_results['f1'], color='purple', alpha=0.7)
    ax4.axhline(y=0.5, color='red', linestyle='--', label='Seuil 0.5')
    ax4.set_xlabel('Patients')
    ax4.set_ylabel('F1-score')
    ax4.set_title('F1-score par patient')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / 'comparison_graphs.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Matrice de confusion globale
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = np.array([[df_results['tn'].sum(), df_results['fp'].sum()],
                   [df_results['fn'].sum(), df_results['tp'].sum()]])
    
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Normal (SAI)', 'Artéfact (SAI)'])
    ax.set_yticklabels(['Normal (Khan)', 'Artéfact (Khan)'])
    ax.set_xlabel('SAI (référence)')
    ax.set_ylabel('Khan')
    ax.set_title('Matrice de confusion globale')
    
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    print("=" * 70)
    print("STATISTIQUES COMPARATIVES KHAN vs SAI")
    print("=" * 70)
    
    files = list(BEFORE_SAI_DIR.glob("*_invasive.csv"))
    print(f"Fichiers trouvés: {len(files)}")
    
    results = []
    
    for i, file in enumerate(files, 1):
        patient_name = file.stem.replace("_invasive", "")
        print(f"\n[{i}/{len(files)}] {patient_name}")
        
        try:
            result = process_patient(patient_name)
            if result:
                results.append(result)
                print(f"  Khan: {result['n_khan']} ({result['pct_khan']:.1f}%) | SAI: {result['n_sai']} ({result['pct_sai']:.1f}%)")
                print(f"  Sens: {result['sensitivity']:.3f} | Spec: {result['specificity']:.3f} | F1: {result['f1']:.3f}")
            else:
                print("  ❌ Ignoré")
        except Exception as e:
            print(f"  ❌ Erreur: {e}")
    
    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_DIR / 'comparison_results.csv', index=False)
        
        print("\n" + "=" * 70)
        print("RÉSULTATS GLOBAUX")
        print("=" * 70)
        
        # Statistiques globales
        total_points = df['n_total'].sum()
        total_khan = df['n_khan'].sum()
        total_sai = df['n_sai'].sum()
        total_tp = df['tp'].sum()
        total_fp = df['fp'].sum()
        total_fn = df['fn'].sum()
        total_tn = df['tn'].sum()
        
        print(f"Total patients: {len(df)}")
        print(f"Total points: {total_points:,}")
        print(f"Artéfacts Khan: {total_khan:,} ({total_khan/total_points*100:.2f}%)")
        print(f"Artéfacts SAI: {total_sai:,} ({total_sai/total_points*100:.2f}%)")
        print("")
        print("Matrice de confusion (Khan vs SAI):")
        print(f"  TP (accord artéfact): {total_tp:,}")
        print(f"  FP (Khan artéfact, SAI normal): {total_fp:,}")
        print(f"  FN (Khan normal, SAI artéfact): {total_fn:,}")
        print(f"  TN (accord normal): {total_tn:,}")
        print("")
        print("Performances moyennes:")
        print(f"  Sensibilité: {df['sensitivity'].mean():.4f}")
        print(f"  Spécificité: {df['specificity'].mean():.4f}")
        print(f"  Précision: {df['precision'].mean():.4f}")
        print(f"  F1-score: {df['f1'].mean():.4f}")
        print(f"  Accuracy: {df['accuracy'].mean():.4f}")
        
        # Tableau détaillé
        create_comparison_table(df)
        create_graphs(df)
        
        print(f"\n✅ Résultats dans: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()