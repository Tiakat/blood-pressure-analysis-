"""
KALMAN FILTER WITH SIGNAL QUALITY INDEX (SQI) - VERSION 2
Pour données invasives brutes (beforesai) à 128 Hz
Correction des graphiques et de la détection
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
INPUT_DIR = Path(r"C:\Users\katia\Desktop\min data\work\data\Patients-infinity\filtered\invasive\operation-csv-invasive\beforesai")
OUTPUT_DIR = Path(r"C:\Users\katia\Desktop\min data\work\data\Patients-infinity\filtered\kalman_sqi_v2")

GRAPHS_DIR = OUTPUT_DIR / "graphs"
ESTIMATES_DIR = OUTPUT_DIR / "estimates"

for d in [OUTPUT_DIR, GRAPHS_DIR, ESTIMATES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

FS = 128

# ============================================================
# PARAMÈTRES DE DÉTECTION (ajustés)
# ============================================================
SBP_MAX = 200
SBP_MIN = 80
DBP_MIN = 40
MAP_MIN = 50
MAP_MAX = 150
HR_MIN = 40
HR_MAX = 120
PP_MIN = 30
DELTA_MAX = 20

# ============================================================
# FONCTIONS
# ============================================================

def get_pressure_column(df):
    possible_cols = ['ART M (mm(hg)^^ISO+)', 'ART_M', 'ART M', 'MAP', 'Mean']
    for col in possible_cols:
        if col in df.columns:
            return col
    return None

def get_sbp_column(df):
    possible_cols = ['ART S (mm(hg)^^ISO+)', 'ART_S', 'ART S', 'SBP']
    for col in possible_cols:
        if col in df.columns:
            return col
    return None

def detect_peaks(values, min_distance=int(0.3 * FS)):
    """Détecter les pics systoliques"""
    smoothed = gaussian_filter1d(values, sigma=2)
    peaks, _ = signal.find_peaks(smoothed, distance=min_distance, prominence=8, height=60)
    return peaks

def calculate_jsqi_simple(values, peaks):
    """
    jSQI simplifié avec des seuils ajustés
    """
    jsqi = []
    n = len(peaks)
    
    if n == 0:
        return jsqi
    
    for i, peak in enumerate(peaks):
        # Extraire SBP, DBP, MAP autour du pic
        start = max(0, peak - int(0.3 * FS))
        end = min(len(values), peak + int(0.3 * FS))
        window = values[start:end]
        
        if len(window) < 10:
            jsqi.append(0)
            continue
        
        sbp = values[peak]
        dbp = np.min(window[:int(0.3 * FS)]) if len(window) > int(0.3 * FS) else np.min(window)
        map_val = np.mean(window)
        pp = sbp - dbp
        
        # Calculer HR
        if i > 0:
            t = (peaks[i] - peaks[i-1]) / FS
            hr = 60 / t if t > 0 else 60
        else:
            hr = 70
        
        is_artifact = 0
        
        # Critères jSQI (seuils ajustés)
        if sbp > SBP_MAX or sbp < SBP_MIN:
            is_artifact = 1
        elif dbp < DBP_MIN:
            is_artifact = 1
        elif map_val < MAP_MIN or map_val > MAP_MAX:
            is_artifact = 1
        elif hr < HR_MIN or hr > HR_MAX:
            is_artifact = 1
        elif pp < PP_MIN:
            is_artifact = 1
        elif i > 0:
            prev_sbp = values[peaks[i-1]]
            if abs(sbp - prev_sbp) > DELTA_MAX:
                is_artifact = 1
        
        jsqi.append(1 - is_artifact)
    
    return jsqi

def kalman_filter_simple(measurements, psi):
    """
    Filtre de Kalman simplifié avec initialisation correcte
    """
    n = len(measurements)
    if n < 2:
        return measurements, np.ones(n)
    
    x = np.zeros(n)
    P = np.zeros(n)
    K = np.zeros(n)
    
    # Initialisation avec la première mesure
    x[0] = measurements[0]
    P[0] = 1.0
    
    for i in range(1, n):
        # Prédiction
        x_pred = x[i-1]
        P_pred = P[i-1] + 0.01
        
        # Mesure
        z = measurements[i]
        
        # Variance adaptative
        if i < len(psi):
            R = 0.5 / (psi[i] + 0.01)
        else:
            R = 1.0
        
        # Gain de Kalman
        K[i] = P_pred / (P_pred + R)
        
        # Mise à jour
        x[i] = x_pred + K[i] * (z - x_pred)
        P[i] = (1 - K[i]) * P_pred
    
    return x, K

def create_graph(patient_name, values, peaks, jsqi, psi, kalman_estimate, output_dir):
    """Créer un graphique avec 4 panneaux"""
    n = len(values)
    display = min(n, 20 * FS)  # 20 secondes
    
    fig, axes = plt.subplots(4, 1, figsize=(16, 14))
    
    # ============================================================
    # 1. Signal original + artéfacts en rouge
    # ============================================================
    ax1 = axes[0]
    ax1.plot(values[:display], color='blue', linewidth=0.8, label='Signal original')
    
    # Marquer les artéfacts en rouge
    artifact_peaks = []
    artifact_values = []
    for i, peak in enumerate(peaks):
        if i < len(jsqi) and jsqi[i] == 0 and peak < display:
            artifact_peaks.append(peak)
            artifact_values.append(values[peak])
    
    if len(artifact_peaks) > 0:
        ax1.scatter(artifact_peaks, artifact_values, color='red', s=30, 
                   marker='x', zorder=5, label=f'Artéfacts ({len(artifact_peaks)})')
    
    ax1.set_title(f'{patient_name} - Artéfacts en rouge (jSQI=0)', fontsize=12)
    ax1.set_ylabel('Pression (mmHg)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, display)
    
    # ============================================================
    # 2. jSQI (détection des artéfacts)
    # ============================================================
    ax2 = axes[1]
    if len(jsqi) > 0:
        times = [p for p in peaks[:len(jsqi)] if p < display]
        jsqi_display = jsqi[:len(times)]
        
        if len(times) > 0:
            # Barres de couleur
            ax2.bar(times, jsqi_display, width=20, color=['green' if j == 1 else 'red' for j in jsqi_display])
            ax2.set_ylim(-0.1, 1.1)
            ax2.set_ylabel('jSQI')
            ax2.set_title('jSQI - 1 = normal, 0 = artéfact')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, display)
    
    # ============================================================
    # 3. ψ (qualité fusionnée)
    # ============================================================
    ax3 = axes[2]
    if len(psi) > 0:
        times = [p for p in peaks[:len(psi)] if p < display]
        psi_display = psi[:len(times)]
        
        if len(times) > 0:
            ax3.plot(times, psi_display, 'o-', color='purple', markersize=3, linewidth=1)
            ax3.axhline(y=0.5, color='red', linestyle='--', label='Seuil 0.5')
            ax3.set_ylim(-0.1, 1.1)
            ax3.set_ylabel('ψ')
            ax3.set_title('Indice de qualité fusionné (ψ)')
            ax3.legend(loc='upper right')
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(0, display)
    
    # ============================================================
    # 4. Estimation Kalman
    # ============================================================
    ax4 = axes[3]
    if len(kalman_estimate) > 0:
        times = [p for p in peaks[:len(kalman_estimate)] if p < display]
        est_display = kalman_estimate[:len(times)]
        
        if len(times) > 0:
            # Signal original en bleu clair
            ax4.plot(values[:display], color='blue', alpha=0.4, linewidth=0.5, label='Original')
            # Mesures aux pics
            measured_values = [values[p] for p in times]
            ax4.scatter(times, measured_values, color='gray', s=10, alpha=0.5, label='Mesures')
            # Estimation Kalman en vert
            ax4.plot(times, est_display, 'o-', color='green', markersize=4, linewidth=1.5, label='Kalman')
            
            ax4.set_ylabel('Pression (mmHg)')
            ax4.set_title('Estimation Kalman vs Signal original')
            ax4.legend(loc='upper right')
            ax4.grid(True, alpha=0.3)
            ax4.set_xlim(0, display)
            ax4.set_xlabel('Temps (échantillons)')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{patient_name}_kalman.png", dpi=150, bbox_inches='tight')
    plt.close()

def process_patient(patient_name):
    """Traiter un patient"""
    
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
    
    # 1. Détection des pics
    peaks = detect_peaks(values)
    if len(peaks) < 10:
        print("  ❌ Pas assez de pics")
        return None
    
    print(f"  Pics: {len(peaks)}")
    
    # 2. jSQI
    jsqi = calculate_jsqi_simple(values, peaks)
    n_artifacts = sum(1 for j in jsqi if j == 0)
    print(f"  Artéfacts jSQI: {n_artifacts}/{len(jsqi)} ({n_artifacts/len(jsqi)*100:.1f}%)")
    
    # 3. wSQI (moyenne mobile de jSQI)
    window_size = min(20, len(jsqi))
    wsqi = np.convolve(jsqi, np.ones(window_size)/window_size, mode='same')
    
    # 4. ψ = fusion
    eta = 0.5
    psi = [wsqi[i] * (1 - eta * (1 - jsqi[i])) for i in range(len(jsqi))]
    psi = [max(0.01, min(1.0, p)) for p in psi]
    
    # 5. Extraire les mesures MAP aux pics
    measurements = [values[p] for p in peaks]
    
    # 6. Kalman
    kalman_estimate, gains = kalman_filter_simple(measurements, psi)
    
    # 7. Graphique
    create_graph(patient_name, values, peaks, jsqi, psi, kalman_estimate, GRAPHS_DIR)
    
    # 8. Sauvegarder
    results_df = pd.DataFrame({
        'peak_index': peaks[:len(kalman_estimate)],
        'map_measured': measurements[:len(kalman_estimate)],
        'jsqi': jsqi[:len(kalman_estimate)],
        'wsqi': wsqi[:len(kalman_estimate)],
        'psi': psi[:len(kalman_estimate)],
        'kalman_estimate': kalman_estimate
    })
    results_df.to_csv(ESTIMATES_DIR / f"{patient_name}_kalman_estimates.csv", index=False)
    
    return {
        'patient': patient_name,
        'n_pulses': len(peaks),
        'n_artifacts': n_artifacts,
        'pct_artifacts': n_artifacts / len(peaks) * 100,
        'mean_psi': np.mean(psi)
    }

def main():
    print("=" * 70)
    print("KALMAN FILTER + SQI - VERSION 2")
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
        df.to_csv(OUTPUT_DIR / "kalman_results.csv", index=False)
        
        print("\n" + "=" * 70)
        print("RÉSUMÉ")
        print("=" * 70)
        print(df.describe())
        print(f"\n✅ Résultats dans: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()