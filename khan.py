"""
KHAN ARTIFACT DETECTION - VERSION CORRIGÉE
- Appliqué sur les données brutes (beforesai)
- Détection avec bornes 30-200 + delta 7.8
- Nettoyage + interpolation
- Les artéfacts sont marqués en rouge sur le graphique 1
- Les artéfacts sont supprimés sur le graphique 2
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
INPUT_DIR = Path(r"C:\Users\katia\Desktop\min data\work\data\Patients-infinity\filtered\invasive\operation-csv-invasive\beforesai")
OUTPUT_DIR = Path(r"C:\Users\katia\Desktop\min data\work\data\Patients-infinity\filtered\khan_filtered")

GRAPHS_DIR = OUTPUT_DIR / "graphs"
CLEANED_DIR = OUTPUT_DIR / "cleaned_data"
REPORTS_DIR = OUTPUT_DIR / "reports"

for d in [OUTPUT_DIR, GRAPHS_DIR, CLEANED_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

FS = 128

# ============================================================
# PARAMÈTRES KHAN (2022)
# ============================================================
MAP_MIN = 30       # mmHg
MAP_MAX = 200      # mmHg
MAP_DELTA = 7.8    # mmHg
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
    """Détection selon Khan (2022)"""
    n = len(values)
    mask = np.zeros(n, dtype=bool)
    
    # Règle 1: Bornes absolues
    for i in range(n):
        if not np.isnan(values[i]):
            if values[i] < MAP_MIN or values[i] > MAP_MAX:
                mask[i] = True
    
    # Règle 2: Delta entre points consécutifs
    for i in range(1, n):
        if not np.isnan(values[i]) and not np.isnan(values[i-1]):
            if abs(values[i] - values[i-1]) > MAP_DELTA:
                mask[i] = True
    
    return mask

def expand_artifact_window(mask, before=5, after=10):
    """Étendre la fenêtre d'artéfact"""
    expanded = mask.copy()
    n = len(mask)
    artifact_indices = np.where(mask)[0]
    for idx in artifact_indices:
        start = max(0, idx - before)
        end = min(n, idx + after + 1)
        expanded[start:end] = True
    return expanded

def interpolate_artifacts(values, mask):
    """Interpoler les artéfacts supprimés"""
    n = len(values)
    cleaned = values.copy().astype(np.float64)
    cleaned[mask] = np.nan
    
    valid_indices = np.where(~mask)[0]
    valid_values = cleaned[valid_indices]
    
    if len(valid_indices) < 2:
        return cleaned
    
    try:
        f = interp1d(valid_indices, valid_values, kind='linear', fill_value='extrapolate')
        return f(np.arange(n))
    except:
        return cleaned

def create_graph(patient_name, values, mask, mask_expanded, interpolated, output_dir):
    """Créer un graphique 3 panneaux"""
    n = len(values)
    
    # Afficher les 10 premières secondes pour mieux voir les artéfacts
    display = min(n, 15 * FS)
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    y_min = np.nanmin(values[:display]) - 30
    y_max = np.nanmax(values[:display]) + 30
    
    # ============================================================
    # 1. Signal original avec les artéfacts en rouge
    # ============================================================
    ax1 = axes[0]
    
    # Tracer le signal original en bleu
    ax1.plot(values[:display], color='blue', linewidth=0.8, label='Signal original')
    
    # Marquer les artéfacts en rouge (points individuels)
    artifact_indices = np.where(mask[:display])[0]
    if len(artifact_indices) > 0:
        ax1.scatter(artifact_indices, values[artifact_indices], 
                   color='red', s=20, alpha=0.8, zorder=5, label=f'Artéfacts ({len(artifact_indices)})')
    
    # Zone étendue en rouge clair
    expanded_indices = np.where(mask_expanded[:display])[0]
    if len(expanded_indices) > 0:
        ax1.fill_between(expanded_indices, y_min, y_max, 
                         color='red', alpha=0.15, label='Zone étendue')
    
    ax1.set_title(f'{patient_name} - Original + Artéfacts en rouge (Khan)', fontsize=12)
    ax1.set_ylabel('MAP (mmHg)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(y_min, y_max)
    
    # ============================================================
    # 2. Signal nettoyé (les artéfacts sont supprimés)
    # ============================================================
    ax2 = axes[1]
    cleaned_no_interp = values.copy().astype(np.float64)
    cleaned_no_interp[mask_expanded] = np.nan
    
    ax2.plot(cleaned_no_interp[:display], color='orange', linewidth=0.8, label='Nettoyé sans interp')
    
    # Marquer les trous (points supprimés) en rouge
    if len(expanded_indices) > 0:
        ax2.fill_between(expanded_indices, y_min, y_max, 
                         color='red', alpha=0.2, label=f'{len(expanded_indices)} points supprimés')
    
    ax2.set_title(f'Signal nettoyé - {mask_expanded.sum()} points supprimés (trous en rouge)', fontsize=12)
    ax2.set_ylabel('MAP (mmHg)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(y_min, y_max)
    
    # ============================================================
    # 3. Signal nettoyé AVEC interpolation
    # ============================================================
    ax3 = axes[2]
    ax3.plot(interpolated[:display], color='green', linewidth=0.8, label='Nettoyé + interpolation')
    
    # Marquer les zones interpolées
    if len(expanded_indices) > 0:
        ax3.fill_between(expanded_indices, y_min, y_max, 
                         color='green', alpha=0.15, label='Interpolation')
    
    ax3.set_title('Signal nettoyé AVEC interpolation (vert)', fontsize=12)
    ax3.set_xlabel('Temps (échantillons)')
    ax3.set_ylabel('MAP (mmHg)')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{patient_name}_khan.png", dpi=150, bbox_inches='tight')
    plt.close()

def process_patient(patient_name):
    """Traiter un patient avec Khan"""
    
    file = INPUT_DIR / f"{patient_name}_invasive.csv"
    if not file.exists():
        return None
    
    df = pd.read_csv(file)
    pressure_col = get_pressure_column(df)
    if pressure_col is None:
        return None
    
    values = df[pressure_col].values
    n = len(values)
    
    print(f"  Points: {n}")
    
    # 1. Détection
    mask = detect_artifacts_khan(values)
    mask_expanded = expand_artifact_window(mask, EXPAND_WINDOW_BEFORE, EXPAND_WINDOW_AFTER)
    
    n_artifacts = mask_expanded.sum()
    print(f"  Artéfacts: {n_artifacts} ({n_artifacts/n*100:.1f}%)")
    
    # 2. Interpolation
    interpolated = interpolate_artifacts(values, mask_expanded)
    
    # 3. Sauvegarde
    cleaned_df = df.copy()
    cleaned_df[pressure_col] = interpolated
    cleaned_df.to_csv(CLEANED_DIR / f"{patient_name}_cleaned_khan.csv", index=False)
    
    # 4. Graphique
    create_graph(patient_name, values, mask, mask_expanded, interpolated, GRAPHS_DIR)
    
    return {
        'patient': patient_name,
        'n_total': n,
        'n_artifacts': n_artifacts,
        'pct_artifacts': n_artifacts / n * 100
    }

def main():
    print("=" * 70)
    print("KHAN ARTIFACT DETECTION")
    print(f"MAP bornes: [{MAP_MIN}, {MAP_MAX}] mmHg")
    print(f"MAP delta max: {MAP_DELTA} mmHg")
    print("=" * 70)
    print(f"Données utilisées: {INPUT_DIR}")
    print("=" * 70)
    
    files = list(INPUT_DIR.glob("*_invasive.csv"))
    print(f"Fichiers trouvés: {len(files)}")
    
    results = []
    
    for i, file in enumerate(files, 1):
        patient_name = file.stem.replace("_invasive", "")
        print(f"\n[{i}/{len(files)}] {patient_name}")
        
        try:
            result = process_patient(patient_name)
            if result:
                results.append(result)
                print("  ✅ OK")
            else:
                print("  ❌ Ignoré")
        except Exception as e:
            print(f"  ❌ Erreur: {e}")
    
    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_DIR / "khan_results.csv", index=False)
        
        print("\n" + "=" * 70)
        print("RÉSUMÉ")
        print("=" * 70)
        print(df.describe())
        
        # Histogramme
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df['pct_artifacts'], bins=20, color='steelblue', edgecolor='black')
        ax.axvline(df['pct_artifacts'].mean(), color='red', linestyle='--', 
                   label=f'Moyenne: {df["pct_artifacts"].mean():.1f}%')
        ax.set_xlabel('Pourcentage d\'artéfacts par patient')
        ax.set_ylabel('Nombre de patients')
        ax.set_title('Distribution du % d\'artéfacts (Khan)')
        ax.legend()
        plt.savefig(OUTPUT_DIR / "artifact_distribution.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Résultats dans: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()