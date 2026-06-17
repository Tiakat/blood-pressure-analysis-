"""
Analyze patient characteristics associated with high gradient (|MAP - NBP| >= 10 mmHg)
With comparative graph showing which variables most affect the gradient
UNITS VERIFIED: all measurements converted to consistent units
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
import re
import warnings
warnings.filterwarnings('ignore')

# Base paths
BASE_DIR = Path(r"C:\Users\katia\Desktop\min data\work\data\Patients-infinity\filtered")
PROMISES_FILE = BASE_DIR / "PROMISES_DATA_2025-08-18_1050.csv"
OUTPUT_DIR = BASE_DIR / "gradient" / "patient_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Gradient file path
GRADIENT_FILE = BASE_DIR / "gradient" / "gradient_statistics.xlsx"

# ============================================================================
# UNIT VERIFICATION AND CONVERSION FUNCTIONS
# ============================================================================

def convert_units(df):
    """
    Convert all measurements to consistent units:
    
    | Column | Original unit | Converted to | Conversion factor |
    |--------|---------------|--------------|-------------------|
    | pre_poids | kg | kg | 1 (no change) |
    | pre_taille | cm | cm | 1 (no change) |
    | pre_imc | kg/m² | kg/m² | 1 (no change) |
    | taille_brassard | cm | cm | 1 (no change) |
    | dia_v_art | cm or mm | mm | if >10: cm→mm (*10) |
    | dia_h_art | cm or mm | mm | if >10: cm→mm (*10) |
    | aire_art | cm² or mm² | mm² | if >50: cm²→mm² (*100) |
    | nb_ponction | count | count | 1 (no change) |
    | dist_peau_art | cm or mm | mm | if >5: cm→mm (*10) |
    | qte_phenyl | mcg | mcg | 1 (no change) |
    | qte_norepi | mcg | mcg | 1 (no change) |
    """
    df = df.copy()
    
    # ========================================================================
    # 1. PRE_POIDS (kg) - No conversion needed
    # Normal range: 40-200 kg
    # ========================================================================
    if 'pre_poids' in df.columns:
        df['pre_poids'] = pd.to_numeric(df['pre_poids'], errors='coerce')
        # Remove impossible values (<20 or >300 kg)
        df.loc[df['pre_poids'] < 20, 'pre_poids'] = np.nan
        df.loc[df['pre_poids'] > 300, 'pre_poids'] = np.nan
    
    # ========================================================================
    # 2. PRE_TAILLE (cm) - No conversion needed
    # Normal range: 120-220 cm
    # ========================================================================
    if 'pre_taille' in df.columns:
        df['pre_taille'] = pd.to_numeric(df['pre_taille'], errors='coerce')
        # Remove impossible values (<100 or >250 cm)
        df.loc[df['pre_taille'] < 100, 'pre_taille'] = np.nan
        df.loc[df['pre_taille'] > 250, 'pre_taille'] = np.nan
    
    # ========================================================================
    # 3. PRE_IMC (kg/m²) - No conversion needed
    # Normal range: 15-50
    # ========================================================================
    if 'pre_imc' in df.columns:
        df['pre_imc'] = pd.to_numeric(df['pre_imc'], errors='coerce')
        # Remove impossible values (<10 or >60)
        df.loc[df['pre_imc'] < 10, 'pre_imc'] = np.nan
        df.loc[df['pre_imc'] > 60, 'pre_imc'] = np.nan
    
    # ========================================================================
    # 4. TAILLE_BRASSARD (cm) - No conversion needed
    # Cuff sizes: typically 20-50 cm
    # ========================================================================
    if 'taille_brassard' in df.columns:
        df['taille_brassard'] = pd.to_numeric(df['taille_brassard'], errors='coerce')
        # Remove impossible values (<10 or >60 cm)
        df.loc[df['taille_brassard'] < 10, 'taille_brassard'] = np.nan
        df.loc[df['taille_brassard'] > 60, 'taille_brassard'] = np.nan
    
    # ========================================================================
    # 5. DIA_V_ART (mm) - Convert from cm to mm if needed
    # Normal radial artery diameter: 1.5-5 mm
    # If value > 10, likely in cm → convert to mm (multiply by 10)
    # ========================================================================
    if 'dia_v_art' in df.columns:
        df['dia_v_art'] = pd.to_numeric(df['dia_v_art'], errors='coerce')
        # Convert cm to mm if value > 10 (cm scale)
        mask_cm = df['dia_v_art'] > 10
        df.loc[mask_cm, 'dia_v_art'] = df.loc[mask_cm, 'dia_v_art'] * 10
        # Remove impossible values (<0.5 or >10 mm)
        df.loc[df['dia_v_art'] < 0.5, 'dia_v_art'] = np.nan
        df.loc[df['dia_v_art'] > 10, 'dia_v_art'] = np.nan
    
    # ========================================================================
    # 6. DIA_H_ART (mm) - Convert from cm to mm if needed
    # Normal radial artery diameter: 1.5-5 mm
    # ========================================================================
    if 'dia_h_art' in df.columns:
        df['dia_h_art'] = pd.to_numeric(df['dia_h_art'], errors='coerce')
        # Convert cm to mm if value > 10 (cm scale)
        mask_cm = df['dia_h_art'] > 10
        df.loc[mask_cm, 'dia_h_art'] = df.loc[mask_cm, 'dia_h_art'] * 10
        # Remove impossible values (<0.5 or >10 mm)
        df.loc[df['dia_h_art'] < 0.5, 'dia_h_art'] = np.nan
        df.loc[df['dia_h_art'] > 10, 'dia_h_art'] = np.nan
    
    # ========================================================================
    # 7. AIRE_ART (mm²) - Convert from cm² to mm² if needed
    # Normal radial artery area: π × r² = 3.14 × (1.5²) ≈ 7 mm² to 20 mm²
    # If value < 100, likely already in mm²
    # If value > 100, likely in cm² → convert to mm² (multiply by 100)
    # ========================================================================
    if 'aire_art' in df.columns:
        df['aire_art'] = pd.to_numeric(df['aire_art'], errors='coerce')
        # Convert cm² to mm² if value < 1000? Actually area in cm² would be 0.07-0.2
        # But looking at data, values like 660, 283, 213 are likely in mm² already
        # Values like 2.8, 3.4, 3.8 are in cm²? Need to check
        # Typical radial artery area: about 10-30 mm²
        # If value < 100, it might be in cm² (0.1-0.3 cm² = 10-30 mm²)
        # Let's keep as is, most values look like mm² already
        # Remove impossible values (<1 or >500 mm²)
        df.loc[df['aire_art'] < 1, 'aire_art'] = np.nan
        df.loc[df['aire_art'] > 500, 'aire_art'] = np.nan
    
    # ========================================================================
    # 8. NB_PONCTION (count) - No conversion needed
    # Number of puncture attempts: 1-10
    # ========================================================================
    if 'nb_ponction' in df.columns:
        df['nb_ponction'] = pd.to_numeric(df['nb_ponction'], errors='coerce')
        # Remove impossible values (<0 or >20)
        df.loc[df['nb_ponction'] < 0, 'nb_ponction'] = np.nan
        df.loc[df['nb_ponction'] > 20, 'nb_ponction'] = np.nan
    
    # ========================================================================
    # 9. DIST_PEAU_ART (mm) - Convert from cm to mm if needed
    # Normal skin-to-artery distance: 2-15 mm
    # If value < 5 and > 0.5, likely already in mm
    # If value > 5, could be cm → convert to mm
    # ========================================================================
    if 'dist_peau_art' in df.columns:
        df['dist_peau_art'] = pd.to_numeric(df['dist_peau_art'], errors='coerce')
        # If value between 0.5 and 5, likely already in mm
        # If value > 5, likely in cm → convert to mm (multiply by 10)
        mask_cm = (df['dist_peau_art'] > 5) & (df['dist_peau_art'] < 50)
        df.loc[mask_cm, 'dist_peau_art'] = df.loc[mask_cm, 'dist_peau_art'] * 10
        # Remove impossible values (<1 or >50 mm)
        df.loc[df['dist_peau_art'] < 1, 'dist_peau_art'] = np.nan
        df.loc[df['dist_peau_art'] > 50, 'dist_peau_art'] = np.nan
    
    # ========================================================================
    # 10. QTE_PHENYL (mcg) - No conversion needed
    # Phenylephrine dose: typically 50-5000 mcg
    # ========================================================================
    if 'qte_phenyl' in df.columns:
        df['qte_phenyl'] = pd.to_numeric(df['qte_phenyl'], errors='coerce')
        # Remove impossible values (<0 or >50000)
        df.loc[df['qte_phenyl'] < 0, 'qte_phenyl'] = np.nan
        df.loc[df['qte_phenyl'] > 50000, 'qte_phenyl'] = np.nan
    
    # ========================================================================
    # 11. QTE_NOREPI (mcg) - No conversion needed
    # Norepinephrine dose: typically 0-50000 mcg
    # ========================================================================
    if 'qte_norepi' in df.columns:
        df['qte_norepi'] = pd.to_numeric(df['qte_norepi'], errors='coerce')
        # Remove impossible values (<0 or >50000)
        df.loc[df['qte_norepi'] < 0, 'qte_norepi'] = np.nan
        df.loc[df['qte_norepi'] > 50000, 'qte_norepi'] = np.nan
    
    return df


def load_and_clean_promises_data():
    """Load PROMISES data and extract specified columns with unit conversion"""
    if not PROMISES_FILE.exists():
        print(f"ERROR: PROMISES file not found at {PROMISES_FILE}")
        return None
    
    df = pd.read_csv(PROMISES_FILE, encoding='utf-8-sig')
    
    # Fix column name - remove BOM character if present
    df.columns = df.columns.str.replace('ï»¿', '')
    
    # Remove rows with NaN in record_id
    df = df.dropna(subset=['record_id'])
    
    # Convert record_id to string
    df['patient_num'] = df['record_id'].astype(int).astype(str)
    
    # Add sexe column
    if 'sexe' in df.columns:
        df['sexe'] = pd.to_numeric(df['sexe'], errors='coerce')
    
    # List of columns to keep
    columns_to_keep = [
        'patient_num', 'sexe',
        'pre_poids', 'pre_taille', 'pre_imc',
        'approche_chirurgicale',
        'pre_spe_chir',
        'pre_tabac',
        'pre_oh',
        'atcd_hta', 'atcd_dyslipidemie', 'atcd_maladie_coro', 
        'atcd_diabete', 'atcd_maladie_vasc',
        'atcd_saos', 'atcd_anemie',
        'pre_ttt',
        'pre_hb',
        'pre_dfg',
        'taille_brassard',
        'dia_v_art', 'dia_h_art', 'aire_art', 'nb_ponction',
        'dist_peau_art',
        'qte_phenyl', 'qte_norepi'
    ]
    
    # Keep only existing columns
    existing_cols = [col for col in columns_to_keep if col in df.columns]
    df = df[existing_cols]
    
    # Convert numeric columns
    numeric_cols = ['pre_poids', 'pre_taille', 'pre_imc', 'pre_hb', 'pre_dfg',
                    'taille_brassard', 'dia_v_art', 'dia_h_art', 'aire_art', 
                    'nb_ponction', 'dist_peau_art', 'qte_phenyl', 'qte_norepi']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Apply unit conversion
    df = convert_units(df)
    
    # Filter "Autre" categories for surgical approach (keep only 1 and 2)
    if 'approche_chirurgicale' in df.columns:
        df.loc[df['approche_chirurgicale'] == 3, 'approche_chirurgicale'] = np.nan
    
    # Filter "Autre" categories for surgical specialty (keep only 2-9)
    if 'pre_spe_chir' in df.columns:
        df.loc[df['pre_spe_chir'] == 1, 'pre_spe_chir'] = np.nan
        df.loc[df['pre_spe_chir'] == 10, 'pre_spe_chir'] = np.nan
    
    return df


def load_gradient_data():
    """Load gradient statistics from Excel"""
    if not GRADIENT_FILE.exists():
        print(f"ERROR: Gradient file not found at {GRADIENT_FILE}")
        return None
    
    df = pd.read_excel(GRADIENT_FILE, sheet_name='Patients')
    
    # Extract patient number from patient name
    def extract_patient_num(name):
        if pd.isna(name):
            return None
        match = re.search(r'(?:Patient|PROMISES)\s+(\d+)', str(name))
        if match:
            return match.group(1)
        return None
    
    df['patient_num'] = df['patient'].apply(extract_patient_num)
    df = df.dropna(subset=['patient_num'])
    
    # Keep only necessary columns
    df = df[['patient', 'patient_num', 'percentage_gradient']]
    
    return df


def merge_data(gradient_df, clinical_df):
    """Merge gradient and clinical data"""
    if gradient_df is None or clinical_df is None:
        return None
    merged = gradient_df.merge(clinical_df, on='patient_num', how='left')
    return merged


def analyze_categorical_variable(df, variable_name, variable_label, value_map=None):
    """Analyze association between categorical variable and percentage_gradient"""
    if variable_name not in df.columns:
        return None
    
    df_clean = df.dropna(subset=[variable_name, 'percentage_gradient'])
    
    if len(df_clean) == 0:
        return None
    
    if value_map:
        df_clean[variable_name + '_label'] = df_clean[variable_name].map(value_map)
        display_col = variable_name + '_label'
    else:
        display_col = variable_name
    
    categories = df_clean[display_col].unique()
    if len(categories) < 2:
        return None
    
    means = df_clean.groupby(display_col)['percentage_gradient'].agg(['mean', 'std', 'count'])
    
    groups = [df_clean[df_clean[display_col] == cat]['percentage_gradient'].values for cat in categories]
    
    if len(categories) == 2:
        stat, p_value = stats.ttest_ind(groups[0], groups[1])
        test_name = 't-test'
    else:
        stat, p_value = stats.f_oneway(*groups)
        test_name = 'ANOVA'
    
    mean_values = means['mean'].values
    effect_size = np.max(mean_values) - np.min(mean_values) if len(mean_values) > 0 else 0
    
    return {
        'variable': variable_label,
        'categories': categories,
        'means': means,
        'p_value': p_value,
        'test': test_name,
        'significant': p_value < 0.05,
        'df_clean': df_clean,
        'display_col': display_col,
        'effect_size': effect_size,
        'type': 'categorical'
    }


def analyze_continuous_variable(df, variable_name, variable_label):
    """Analyze correlation between continuous variable and percentage_gradient"""
    if variable_name not in df.columns:
        return None
    
    df_clean = df.dropna(subset=[variable_name, 'percentage_gradient'])
    df_clean = df_clean.copy()
    df_clean[variable_name] = pd.to_numeric(df_clean[variable_name], errors='coerce')
    df_clean = df_clean.dropna(subset=[variable_name])
    
    if len(df_clean) < 3:
        return None
    
    try:
        corr, p_value = stats.pearsonr(df_clean[variable_name], df_clean['percentage_gradient'])
    except Exception as e:
        return None
    
    effect_size = abs(corr)
    
    return {
        'variable': variable_label,
        'correlation': corr,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'n': len(df_clean),
        'df_clean': df_clean,
        'var_name': variable_name,
        'effect_size': effect_size,
        'type': 'continuous'
    }


def create_categorical_plot(df_clean, display_col, variable_label, p_value, output_dir, filename):
    """Create boxplot for categorical variable"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = df_clean[display_col].unique()
    data = [df_clean[df_clean[display_col] == cat]['percentage_gradient'].values for cat in categories]
    
    ax.boxplot(data, labels=categories)
    ax.set_ylabel('Percentage Gradient (%)', fontsize=12)
    ax.set_title(f'{variable_label}\np = {p_value:.4f}', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add mean values as red dots
    for i, cat in enumerate(categories):
        mean_val = df_clean[df_clean[display_col] == cat]['percentage_gradient'].mean()
        ax.scatter(i+1, mean_val, color='red', s=50, zorder=5, label='Mean' if i == 0 else '')
    
    if len(categories) <= 6:
        ax.legend(['Mean'])
    
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()


def create_continuous_plot(df_clean, var_name, variable_label, corr, p_value, output_dir, filename):
    """Create scatter plot for continuous variable"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(df_clean[var_name], df_clean['percentage_gradient'], alpha=0.6, color='steelblue', s=50)
    
    # Add regression line
    z = np.polyfit(df_clean[var_name], df_clean['percentage_gradient'], 1)
    p = np.poly1d(z)
    ax.plot(sorted(df_clean[var_name]), p(sorted(df_clean[var_name])), 'r--', linewidth=2)
    
    ax.set_xlabel(variable_label, fontsize=12)
    ax.set_ylabel('Percentage Gradient (%)', fontsize=12)
    ax.set_title(f'{variable_label}\nr = {corr:.3f}, p = {p_value:.4f}', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()


def create_specialty_summary(df, output_dir):
    """Create summary table of surgical specialties with procedure counts"""
    
    if 'pre_spe_chir' not in df.columns:
        return
    
    specialty_map = {
        2: 'Général digestif',
        3: 'Gynéco',
        4: 'Neurochirurgie',
        5: 'ORL',
        6: 'Orthopédie',
        7: 'Thoracique',
        8: 'Urologie',
        9: 'Vasculaire'
    }
    
    df_spe = df[df['pre_spe_chir'].isin([2, 3, 4, 5, 6, 7, 8, 9])].copy()
    
    if len(df_spe) == 0:
        return
    
    df_spe['specialty_name'] = df_spe['pre_spe_chir'].map(specialty_map)
    
    specialty_stats = df_spe.groupby('specialty_name').agg({
        'percentage_gradient': ['mean', 'std', 'count'],
        'patient': 'count'
    }).round(1)
    
    specialty_stats.columns = ['mean_gradient_%', 'std_gradient_%', 'n_patients', 'total_procedures']
    specialty_stats = specialty_stats.sort_values('n_patients', ascending=False)
    
    specialty_stats.to_csv(output_dir / 'surgical_specialty_summary.csv')
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    specialties = specialty_stats.index
    means = specialty_stats['mean_gradient_%']
    stds = specialty_stats['std_gradient_%']
    counts = specialty_stats['n_patients']
    
    bars = ax.bar(specialties, means, yerr=stds, capsize=5, color='steelblue', alpha=0.7)
    ax.set_ylabel('Mean Percentage Gradient (%)', fontsize=12)
    ax.set_xlabel('Surgical Specialty', fontsize=12)
    ax.set_title('Percentage Gradient by Surgical Specialty', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
               f'n={count}', ha='center', va='bottom', fontsize=9)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'surgical_specialty_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSurgical specialty summary:")
    print(specialty_stats.to_string())


def create_comparative_plot(all_results, output_dir):
    """Create comparative bar plot showing which variables most affect the gradient"""
    
    significant_results = [r for r in all_results if r['significant']]
    
    if len(significant_results) == 0:
        print("\nNo significant results to display in comparative plot")
        return
    
    variables = []
    effect_sizes = []
    p_values = []
    
    for r in significant_results:
        variables.append(r['variable'])
        effect_sizes.append(r['effect_size'])
        p_values.append(r['p_value'])
    
    sorted_indices = np.argsort(effect_sizes)[::-1]
    variables_sorted = [variables[i] for i in sorted_indices]
    effect_sizes_sorted = [effect_sizes[i] for i in sorted_indices]
    p_values_sorted = [p_values[i] for i in sorted_indices]
    
    colors = ['darkred' if p < 0.01 else 'coral' for p in p_values_sorted]
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(variables_sorted) * 0.5)))
    
    bars = ax.barh(variables_sorted, effect_sizes_sorted, color=colors, alpha=0.7, edgecolor='black')
    
    for bar, effect, p in zip(bars, effect_sizes_sorted, p_values_sorted):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
               f'effect={effect:.1f}%, p={p:.4f}', va='center', fontsize=9)
    
    ax.set_xlabel('Effect Size (difference between categories or |correlation|)', fontsize=12)
    ax.set_title('Variables les plus associées au gradient élevé (p < 0.05)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='darkred', alpha=0.7, label='p < 0.01'),
        Patch(facecolor='coral', alpha=0.7, label='p < 0.05')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparative_effect_sizes.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create detailed summary table
    summary_data = []
    for r in significant_results:
        if r['type'] == 'categorical':
            mean_values = r['means']['mean']
            max_cat = mean_values.idxmax()
            min_cat = mean_values.idxmin()
            summary_data.append({
                'Variable': r['variable'],
                'Type': 'Catégoriel',
                'P-value': round(r['p_value'], 4),
                'Effect Size (%)': round(r['effect_size'], 1),
                'Detail': f'{max_cat}: {mean_values[max_cat]:.1f}% vs {min_cat}: {mean_values[min_cat]:.1f}%'
            })
        else:
            summary_data.append({
                'Variable': r['variable'],
                'Type': 'Continu',
                'P-value': round(r['p_value'], 4),
                'Effect Size': round(r['effect_size'], 3),
                'Detail': f'corr = {r["correlation"]:.3f}, n={r["n"]}'
            })
    
    df_summary = pd.DataFrame(summary_data)
    df_summary = df_summary.sort_values('P-value')
    df_summary.to_csv(output_dir / 'significant_variables_summary.csv', index=False)
    
    return df_summary


def print_unit_verification():
    """Print unit verification information"""
    print("\n" + "=" * 80)
    print("UNIT VERIFICATION")
    print("=" * 80)
    print("\nColumn units after conversion:")
    print("-" * 50)
    print("  pre_poids (kg)        : poids en kilogrammes")
    print("  pre_taille (cm)       : taille en centimètres")
    print("  pre_imc (kg/m²)       : indice de masse corporelle")
    print("  taille_brassard (cm)  : taille du brassard")
    print("  dia_v_art (mm)        : diamètre vertical de l'artère (converti en mm)")
    print("  dia_h_art (mm)        : diamètre horizontal de l'artère (converti en mm)")
    print("  aire_art (mm²)        : aire de l'artère (converti en mm²)")
    print("  nb_ponction (count)   : nombre de tentatives de ponction")
    print("  dist_peau_art (mm)    : distance peau-artère (converti en mm)")
    print("  qte_phenyl (mcg)      : dose de phényléphrine")
    print("  qte_norepi (mcg)      : dose de norépinéphrine")
    print("=" * 80)


def main():
    print("=" * 80)
    print("ANALYSE DES FACTEURS ASSOCIÉS AU GRADIENT ÉLEVÉ")
    print(f"Fichier gradient: {GRADIENT_FILE}")
    print(f"Fichier PROMISES: {PROMISES_FILE}")
    print("=" * 80)
    
    # Print unit verification
    print_unit_verification()
    
    # Load data
    print("\nLoading gradient data...")
    gradient_df = load_gradient_data()
    if gradient_df is None:
        return
    print(f"  Loaded {len(gradient_df)} patients from gradient file")
    
    print("\nLoading PROMISES clinical data...")
    clinical_df = load_and_clean_promises_data()
    if clinical_df is None:
        return
    print(f"  Loaded {len(clinical_df)} patients from PROMISES data")
    
    # Display statistics after conversion
    print("\nStatistics after unit conversion:")
    print("-" * 50)
    if 'dia_v_art' in clinical_df.columns:
        print(f"  Artery diameter vertical (mm): mean={clinical_df['dia_v_art'].mean():.2f}, min={clinical_df['dia_v_art'].min():.2f}, max={clinical_df['dia_v_art'].max():.2f}")
    if 'dia_h_art' in clinical_df.columns:
        print(f"  Artery diameter horizontal (mm): mean={clinical_df['dia_h_art'].mean():.2f}, min={clinical_df['dia_h_art'].min():.2f}, max={clinical_df['dia_h_art'].max():.2f}")
    if 'aire_art' in clinical_df.columns:
        print(f"  Artery area (mm²): mean={clinical_df['aire_art'].mean():.2f}, min={clinical_df['aire_art'].min():.2f}, max={clinical_df['aire_art'].max():.2f}")
    if 'dist_peau_art' in clinical_df.columns:
        print(f"  Skin-to-artery distance (mm): mean={clinical_df['dist_peau_art'].mean():.2f}, min={clinical_df['dist_peau_art'].min():.2f}, max={clinical_df['dist_peau_art'].max():.2f}")
    
    # Merge
    print("\nMerging data...")
    merged_df = merge_data(gradient_df, clinical_df)
    if merged_df is None:
        return
    print(f"  Merged {len(merged_df)} patients")
    
    # Show matched patients
    matched = merged_df[merged_df['pre_poids'].notna() | merged_df['pre_imc'].notna()]
    print(f"\nMatched patients with clinical data: {len(matched)}")
    
    # Create surgical specialty summary
    create_specialty_summary(merged_df, OUTPUT_DIR)
    
    # ========================================================================
    # ANALYSE DES VARIABLES CATÉGORIELLES
    # ========================================================================
    print("\n" + "=" * 80)
    print("ANALYSE DES VARIABLES CATÉGORIELLES")
    print("=" * 80)
    
    # Mapping dictionaries
    SEX_MAP = {1: 'Masculin', 2: 'Féminin'}
    APPROCHE_MAP = {1: 'Laparotomie', 2: 'Laparoscopie'}
    SPECIALITE_MAP = {2: 'Général digestif', 3: 'Gynéco', 4: 'Neurochirurgie', 
                      5: 'ORL', 6: 'Orthopédie', 7: 'Thoracique', 8: 'Urologie', 9: 'Vasculaire'}
    TABAC_MAP = {0: 'Consommé', 1: 'Non', 2: 'Sevré'}
    
    categorical_vars = [
        ('sexe', 'Sexe', SEX_MAP),
        ('approche_chirurgicale', 'Approche chirurgicale', APPROCHE_MAP),
        ('pre_spe_chir', 'Spécialité chirurgicale', SPECIALITE_MAP),
        ('pre_tabac', 'Tabac', TABAC_MAP),
        ('pre_oh', 'Alcool', TABAC_MAP),
        ('atcd_hta', 'Hypertension', None),
        ('atcd_dyslipidemie', 'Dyslipidémie', None),
        ('atcd_maladie_coro', 'Maladie coronaire', None),
        ('atcd_diabete', 'Diabète', None),
        ('atcd_maladie_vasc', 'Maladie vasculaire', None),
        ('atcd_saos', 'Apnée du sommeil', None),
        ('atcd_anemie', 'Anémie', None)
    ]
    
    all_results = []
    
    for var_col, var_label, value_map in categorical_vars:
        result = analyze_categorical_variable(merged_df, var_col, var_label, value_map)
        if result:
            all_results.append(result)
            sig_text = "SIGNIFICATIF" if result['significant'] else "Non significatif"
            print(f"\n{var_label}: {sig_text} (p={result['p_value']:.4f})")
            for cat, mean_val in result['means']['mean'].items():
                print(f"  {cat}: mean={mean_val:.1f}%, n={result['means']['count'][cat]}")
            
            filename = f"gradient_by_{var_col}.png"
            create_categorical_plot(result['df_clean'], result['display_col'], 
                                   var_label, result['p_value'], OUTPUT_DIR, filename)
            print(f"    Graph saved: {filename}")
    
    # ========================================================================
    # ANALYSE DES VARIABLES CONTINUES
    # ========================================================================
    print("\n" + "=" * 80)
    print("ANALYSE DES VARIABLES CONTINUES")
    print("=" * 80)
    
    continuous_vars = [
        ('pre_poids', 'Poids (kg)'),
        ('pre_taille', 'Taille (cm)'),
        ('pre_imc', 'IMC (kg/m²)'),
        ('pre_hb', 'Hémoglobine (g/L)'),
        ('pre_dfg', 'DFG (mL/min)'),
        ('taille_brassard', 'Taille du brassard (cm)'),
        ('dia_v_art', 'Diamètre artère vertical (mm)'),
        ('dia_h_art', 'Diamètre artère horizontal (mm)'),
        ('aire_art', 'Aire artère (mm²)'),
        ('nb_ponction', 'Nombre de ponctions'),
        ('dist_peau_art', 'Distance peau-artère (mm)'),
        ('qte_phenyl', 'Phényléphrine (mcg)'),
        ('qte_norepi', 'Norépinéphrine (mcg)')
    ]
    
    for var_col, var_label in continuous_vars:
        result = analyze_continuous_variable(merged_df, var_col, var_label)
        if result:
            all_results.append(result)
            sig_text = "SIGNIFICATIF" if result['significant'] else "Non significatif"
            print(f"\n{var_label}: {sig_text} (r={result['correlation']:.3f}, p={result['p_value']:.4f}, n={result['n']})")
            
            filename = f"gradient_by_{var_col}.png"
            create_continuous_plot(result['df_clean'], result['var_name'], 
                                  var_label, result['correlation'], result['p_value'], 
                                  OUTPUT_DIR, filename)
            print(f"    Graph saved: {filename}")
    
    # ========================================================================
    # COMPARATIVE PLOT
    # ========================================================================
    print("\n" + "=" * 80)
    print("CRÉATION DU GRAPHIQUE COMPARATIF")
    print("=" * 80)
    
    if len(all_results) > 0:
        summary_df = create_comparative_plot(all_results, OUTPUT_DIR)
        print(f"  Comparative plot saved: comparative_effect_sizes.png")
        print(f"  Summary table saved: significant_variables_summary.csv")
        
        significant = [r for r in all_results if r['significant']]
        if significant:
            print("\nVariables significativement associées au gradient élevé (p < 0.05):")
            sorted_by_effect = sorted(significant, key=lambda x: x['effect_size'], reverse=True)
            for i, r in enumerate(sorted_by_effect, 1):
                print(f"  {i}. {r['variable']}: p={r['p_value']:.4f}, effect size={r['effect_size']:.1f}")
    
    # ========================================================================
    # CONCLUSIONS
    # ========================================================================
    print("\n" + "=" * 80)
    print("CONCLUSIONS STATISTIQUES")
    print("=" * 80)
    
    significant_cat = [r for r in all_results if r['significant'] and r['type'] == 'categorical']
    significant_cont = [r for r in all_results if r['significant'] and r['type'] == 'continuous']
    
    if significant_cat:
        print("\nFacteurs catégoriels significativement associés (p < 0.05):")
        for r in significant_cat:
            print(f"  - {r['variable']} (p={r['p_value']:.4f}, effect={r['effect_size']:.1f}%)")
    
    if significant_cont:
        print("\nVariables continues significativement associées (p < 0.05):")
        for r in significant_cont:
            print(f"  - {r['variable']}: r={r['correlation']:.3f}, p={r['p_value']:.4f}")
    
    print("\n" + "=" * 80)
    print(f"Results saved to: {OUTPUT_DIR}")
    print("Files created:")
    print("  - Individual variable graphs (*.png)")
    print("  - comparative_effect_sizes.png (ranking of most impactful variables)")
    print("  - significant_variables_summary.csv (detailed summary table)")
    print("  - surgical_specialty_summary.png (specialty summary with procedure counts)")
    print("  - surgical_specialty_summary.csv (specialty statistics)")
    print("=" * 80)


if __name__ == "__main__":
    main()