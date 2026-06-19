"""
VALIDATION COMPLÈTE KHAN vs SAI
Vérifie pour chaque patient :
1. Correspondance des longueurs de fichiers
2. Détection des artéfacts SAI (NaN dans aftersai)
3. Détection des artéfacts Khan
4. Matrice de confusion
5. Rapport de confiance
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
BEFORE_SAI_DIR = Path(r"C:\Users\katia\Desktop\min data\work\data\Patients-infinity\filtered\invasive\operation-csv-invasive\beforesai")
AFTER_SAI_DIR = Path(r"C:\Users\katia\Desktop\min data\work\data\Patients-infinity\filtered\invasive\operation-csv-invasive\aftersai")
OUTPUT_DIR = Path(r"C:\Users\katia\Desktop\min data\work\data\Patients-infinity\filtered\validation_report")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# PARAMÈTRES KHAN
# ============================================================
MAP_MIN = 30
MAP_MAX = 200
MAP_DELTA = 7.8

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

def validate_patient(patient_name):
    """Valider un patient"""
    
    before_file = BEFORE_SAI_DIR / f"{patient_name}_invasive.csv"
    after_file = AFTER_SAI_DIR / f"{patient_name}_invasive_cleaned.csv"
    
    if not before_file.exists():
        return None
    
    df_before = pd.read_csv(before_file)
    df_after = pd.read_csv(after_file) if after_file.exists() else None
    
    pressure_col = get_pressure_column(df_before)
    if pressure_col is None:
        return None
    
    values_before = df_before[pressure_col].values
    n_before = len(values_before)
    
    # 1. Vérifier la longueur du fichier après SAI
    if df_after is not None:
        values_after = df_after[pressure_col].values
        n_after = len(values_after)
        
        # Vérifier si les longueurs correspondent
        if n_after != n_before:
            print(f"  ⚠️ Longueurs différentes: before={n_before}, after={n_after}")
            # Ajuster
            if n_after < n_before:
                values_after_padded = np.full(n_before, np.nan)
                values_after_padded[:n_after] = values_after
                values_after = values_after_padded
            else:
                values_after = values_after[:n_before]
    else:
        values_after = np.full(n_before, np.nan)
    
    # 2. Détection SAI (NaN dans aftersai)
    mask_sai = np.isnan(values_after)
    n_sai = mask_sai.sum()
    
    # 3. Détection Khan
    mask_khan = detect_artifacts_khan(values_before)
    n_khan = mask_khan.sum()
    
    # 4. Matrice de confusion
    tp = np.sum(mask_khan & mask_sai)
    fp = np.sum(mask_khan & ~mask_sai)
    fn = np.sum(~mask_khan & mask_sai)
    tn = np.sum(~mask_khan & ~mask_sai)
    
    # 5. Métriques
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    accuracy = (tp + tn) / n_before if n_before > 0 else 0
    
    # 6. Vérification de cohérence
    issues = []
    
    # Vérifier si les NaN dans aftersai sont bien des artéfacts
    # (les valeurs avant SAI doivent être différentes des valeurs après)
    if df_after is not None:
        for i in range(min(100, n_before)):
            if mask_sai[i] and not np.isnan(values_before[i]):
                # SAI a supprimé une valeur qui était présente avant → OK
                pass
            elif not mask_sai[i] and np.isnan(values_before[i]):
                # Valeur manquante avant mais présente après → anomalie
                issues.append(f"Valeur manquante avant SAI à l'index {i}")
    
    # 7. Calculer le taux d'accord
    agreement = (tp + tn) / n_before if n_before > 0 else 0
    
    return {
        'patient': patient_name,
        'n_before': n_before,
        'n_after': n_after if df_after is not None else 0,
        'n_sai': n_sai,
        'pct_sai': n_sai / n_before * 100 if n_before > 0 else 0,
        'n_khan': n_khan,
        'pct_khan': n_khan / n_before * 100 if n_before > 0 else 0,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1': f1,
        'accuracy': accuracy,
        'agreement': agreement,
        'issues': issues,
        'is_valid': len(issues) == 0
    }

def create_validation_report(all_results):
    """Créer un rapport de validation complet"""
    
    # Statistiques globales
    valid_patients = [r for r in all_results if r['is_valid']]
    invalid_patients = [r for r in all_results if not r['is_valid']]
    
    print("\n" + "=" * 70)
    print("RAPPORT DE VALIDATION")
    print("=" * 70)
    print(f"Patients valides: {len(valid_patients)}/{len(all_results)}")
    
    if invalid_patients:
        print(f"Patients avec anomalies: {len(invalid_patients)}")
        for p in invalid_patients:
            print(f"  - {p['patient']}: {len(p['issues'])} anomalies")
    
    print("\n" + "-" * 70)
    print("STATISTIQUES GLOBALES")
    print("-" * 70)
    
    total_points = sum(r['n_before'] for r in all_results)
    total_sai = sum(r['n_sai'] for r in all_results)
    total_khan = sum(r['n_khan'] for r in all_results)
    total_tp = sum(r['tp'] for r in all_results)
    total_fp = sum(r['fp'] for r in all_results)
    total_fn = sum(r['fn'] for r in all_results)
    total_tn = sum(r['tn'] for r in all_results)
    
    print(f"Total points: {total_points:,}")
    print(f"Artéfacts SAI: {total_sai:,} ({total_sai/total_points*100:.2f}%)")
    print(f"Artéfacts Khan: {total_khan:,} ({total_khan/total_points*100:.2f}%)")
    print(f"TP: {total_tp:,}")
    print(f"FP: {total_fp:,}")
    print(f"FN: {total_fn:,}")
    print(f"TN: {total_tn:,}")
    
    # Métriques globales
    global_sensitivity = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    global_specificity = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0
    global_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    global_f1 = 2 * (global_precision * global_sensitivity) / (global_precision + global_sensitivity) if (global_precision + global_sensitivity) > 0 else 0
    global_accuracy = (total_tp + total_tn) / total_points if total_points > 0 else 0
    
    print("\n" + "-" * 70)
    print("MÉTRIQUES GLOBALES")
    print("-" * 70)
    print(f"Sensibilité globale: {global_sensitivity:.4f}")
    print(f"Spécificité globale: {global_specificity:.4f}")
    print(f"Précision globale: {global_precision:.4f}")
    print(f"F1-score global: {global_f1:.4f}")
    print(f"Accuracy globale: {global_accuracy:.4f}")
    
    # Rapport de confiance
    print("\n" + "-" * 70)
    print("RAPPORT DE CONFIANCE")
    print("-" * 70)
    
    if global_sensitivity > 0.8 and global_specificity > 0.8:
        print("✅ Confiance ÉLEVÉE: Khan et SAI sont en bon accord")
    elif global_sensitivity > 0.5 and global_specificity > 0.8:
        print("⚠️ Confiance MODÉRÉE: Khan détecte moins d'artéfacts que SAI")
    else:
        print("❌ Confiance FAIBLE: Khan détecte beaucoup moins d'artéfacts que SAI")
    
    print(f"\nRaison: Sensibilité={global_sensitivity:.4f} (Khan ne détecte que {global_sensitivity*100:.1f}% des artéfacts SAI)")
    
    return {
        'total_points': total_points,
        'total_sai': total_sai,
        'total_khan': total_khan,
        'global_sensitivity': global_sensitivity,
        'global_specificity': global_specificity,
        'global_f1': global_f1,
        'global_accuracy': global_accuracy
    }

def create_validation_graph(all_results, output_dir):
    """Créer des graphiques de validation"""
    
    df = pd.DataFrame(all_results)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Comparaison Khan vs SAI par patient
    ax1 = axes[0, 0]
    x = range(len(df))
    ax1.bar(x, df['pct_khan'], alpha=0.7, label='Khan', color='red')
    ax1.bar(x, df['pct_sai'], alpha=0.7, label='SAI', color='green')
    ax1.set_xlabel('Patients')
    ax1.set_ylabel('Pourcentage d\'artéfacts (%)')
    ax1.set_title('Khan vs SAI - % d\'artéfacts par patient')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Sensibilité et Spécificité
    ax2 = axes[0, 1]
    ax2.scatter(df['sensitivity'], df['specificity'], alpha=0.7, s=50)
    ax2.axhline(y=0.8, color='red', linestyle='--', label='Seuil 0.8')
    ax2.axvline(x=0.8, color='red', linestyle='--')
    ax2.set_xlabel('Sensibilité')
    ax2.set_ylabel('Spécificité')
    ax2.set_title('Sensibilité vs Spécificité par patient')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. F1-score
    ax3 = axes[1, 0]
    ax3.bar(x, df['f1'], color='purple', alpha=0.7)
    ax3.axhline(y=0.5, color='red', linestyle='--', label='Seuil 0.5')
    ax3.set_xlabel('Patients')
    ax3.set_ylabel('F1-score')
    ax3.set_title('F1-score par patient')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Accord global
    ax4 = axes[1, 1]
    ax4.bar(x, df['agreement'], color='steelblue', alpha=0.7)
    ax4.axhline(y=0.8, color='red', linestyle='--', label='Seuil 0.8')
    ax4.set_xlabel('Patients')
    ax4.set_ylabel('Taux d\'accord')
    ax4.set_title('Accord Khan vs SAI par patient')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'validation_graphs.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    print("=" * 70)
    print("VALIDATION COMPLÈTE KHAN vs SAI")
    print("=" * 70)
    
    files = list(BEFORE_SAI_DIR.glob("*_invasive.csv"))
    print(f"Fichiers trouvés: {len(files)}")
    
    all_results = []
    
    for i, file in enumerate(files, 1):
        patient_name = file.stem.replace("_invasive", "")
        print(f"\n[{i}/{len(files)}] {patient_name}")
        
        try:
            result = validate_patient(patient_name)
            if result:
                all_results.append(result)
                status = "✅" if result['is_valid'] else "⚠️"
                print(f"  {status} SAI: {result['n_sai']} ({result['pct_sai']:.2f}%) | Khan: {result['n_khan']} ({result['pct_khan']:.2f}%)")
                print(f"     Sens: {result['sensitivity']:.4f} | Spec: {result['specificity']:.4f} | F1: {result['f1']:.4f}")
                if not result['is_valid']:
                    print(f"     ⚠️ Anomalies: {len(result['issues'])}")
            else:
                print("  ❌ Ignoré")
        except Exception as e:
            print(f"  ❌ Erreur: {e}")
    
    if all_results:
        # Sauvegarder les résultats
        df = pd.DataFrame(all_results)
        df.to_csv(OUTPUT_DIR / 'validation_results.csv', index=False)
        
        # Créer le rapport
        stats = create_validation_report(all_results)
        
        # Créer les graphiques
        create_validation_graph(all_results, OUTPUT_DIR)
        
        print(f"\n✅ Rapport de validation sauvegardé dans: {OUTPUT_DIR}")
        print(f"   - validation_results.csv")
        print(f"   - validation_graphs.png")

if __name__ == "__main__":
    main()