"""
Analyze patient characteristics associated with high gradient (|MAP - NBP| >= 10 mmHg)
With comparative graph showing which variables most affect the gradient
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
import re
import warnings
warnings.filterwarnings('ignore')

# Paths
GRADIENT_FILE = Path(r"C:\Users\katia\Desktop\min data\work\data\Patients-infinity\filtered\gradient\gradient_statistics.xlsx")
PROMISES_FILE = Path(r"C:\Users\katia\Desktop\min data\work\data\Patients-infinity\filtered\PROMISES_DATA_2025-08-18_1050.csv")
OUTPUT_DIR = Path(r"C:\Users\katia\Desktop\min data\work\data\Patients-infinity\filtered\gradient\patient_analysis")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Mapping dictionaries for categorical variables
SPECIALITE_MAP = {
    1: 'Plastique',
    2: 'Général digestif',
    3: 'Gynéco',
    4: 'Neurochirurgie',
    5: 'ORL',
    6: 'Orthopédie',
    7: 'Thoracique',
    8: 'Urologie',
    9: 'Vasculaire',
    10: 'Autre'
}

APPROCHE_MAP = {
    1: 'Laparotomie',
    2: 'Laparoscopie',
    3: 'Autre'
}

TABAC_MAP = {
    0: 'Consommé',
    1: 'Non',
    2: 'Sevré'
}

SEX_MAP = {
    1: 'Masculin',
    2: 'Féminin'
}


def load_and_clean_promises_data():
    """Load PROMISES data and extract specified columns"""
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
    
    # Filter BMI: remove values > 60 (weird values)
    if 'pre_imc' in df.columns:
        df.loc[df['pre_imc'] > 60, 'pre_imc'] = np.nan
    
    return df


def load_gradient_data():
    """Load gradient statistics from Excel"""
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
    merged = gradient_df.merge(clinical_df, on='patient_num', how='left')
    return merged


def analyze_categorical_variable(df, variable_name, variable_label, value_map=None):
    """Analyze association between categorical variable and percentage_gradient"""
    if variable_name not in df.columns:
        return None
    
    # Remove NaN
    df_clean = df.dropna(subset=[variable_name, 'percentage_gradient'])
    
    if len(df_clean) == 0:
        return None
    
    # Apply value mapping if provided
    if value_map:
        df_clean[variable_name + '_label'] = df_clean[variable_name].map(value_map)
        display_col = variable_name + '_label'
    else:
        display_col = variable_name
    
    # Get unique categories
    categories = df_clean[display_col].unique()
    if len(categories) < 2:
        return None
    
    # Calculate mean percentage per category
    means = df_clean.groupby(display_col)['percentage_gradient'].agg(['mean', 'std', 'count'])
    
    # Perform t-test or ANOVA
    groups = [df_clean[df_clean[display_col] == cat]['percentage_gradient'].values for cat in categories]
    
    if len(categories) == 2:
        stat, p_value = stats.ttest_ind(groups[0], groups[1])
        test_name = 't-test'
    else:
        stat, p_value = stats.f_oneway(*groups)
        test_name = 'ANOVA'
    
    # Calculate effect size (difference between highest and lowest category mean)
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
    
    # Remove NaN and ensure numeric
    df_clean = df.dropna(subset=[variable_name, 'percentage_gradient'])
    df_clean = df_clean.copy()
    df_clean[variable_name] = pd.to_numeric(df_clean[variable_name], errors='coerce')
    df_clean = df_clean.dropna(subset=[variable_name])
    
    if len(df_clean) < 3:
        return None
    
    # Calculate correlation
    try:
        corr, p_value = stats.pearsonr(df_clean[variable_name], df_clean['percentage_gradient'])
    except Exception as e:
        return None
    
    # For continuous variables, effect size is absolute correlation
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
    
    bp = ax.boxplot(data, labels=categories)
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
    ax.plot(sorted(df_clean[var_name]), p(sorted(df_clean[var_name])), 
            'r--', linewidth=2)
    
    ax.set_xlabel(variable_label, fontsize=12)
    ax.set_ylabel('Percentage Gradient (%)', fontsize=12)
    ax.set_title(f'{variable_label}\nr = {corr:.3f}, p = {p_value:.4f}', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()


def create_comparative_plot(all_results, output_dir):
    """Create comparative bar plot showing which variables most affect the gradient"""
    
    # Filter significant results (p < 0.05)
    significant_results = [r for r in all_results if r['significant']]
    
    if len(significant_results) == 0:
        print("\nNo significant results to display in comparative plot")
        return
    
    # Prepare data for plotting
    variables = []
    effect_sizes = []
    p_values = []
    types = []
    
    for r in significant_results:
        variables.append(r['variable'])
        effect_sizes.append(r['effect_size'])
        p_values.append(r['p_value'])
        types.append(r['type'])
    
    # Sort by effect size (largest first)
    sorted_indices = np.argsort(effect_sizes)[::-1]
    variables_sorted = [variables[i] for i in sorted_indices]
    effect_sizes_sorted = [effect_sizes[i] for i in sorted_indices]
    p_values_sorted = [p_values[i] for i in sorted_indices]
    types_sorted = [types[i] for i in sorted_indices]
    
    # Create color map based on variable type and p-value significance
    colors = []
    for i, (var, p, typ) in enumerate(zip(variables_sorted, p_values_sorted, types_sorted)):
        if p < 0.01:
            colors.append('darkred')
        elif p < 0.05:
            colors.append('coral')
        else:
            colors.append('steelblue')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(6, len(variables_sorted) * 0.5)))
    
    bars = ax.barh(variables_sorted, effect_sizes_sorted, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, effect, p in zip(bars, effect_sizes_sorted, p_values_sorted):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
               f'effect={effect:.1f}%, p={p:.4f}', va='center', fontsize=9)
    
    ax.set_xlabel('Effect Size (difference between categories or |correlation|)', fontsize=12)
    ax.set_title('Variables les plus associées au gradient élevé (p < 0.05)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend for p-value categories
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='darkred', alpha=0.7, label='p < 0.01'),
        Patch(facecolor='coral', alpha=0.7, label='0.01 ≤ p < 0.05'),
        Patch(facecolor='steelblue', alpha=0.7, label='p ≥ 0.05 (inclus par erreur)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparative_effect_sizes.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also create a detailed summary table
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


def main():
    print("=" * 80)
    print("ANALYSE DES FACTEURS ASSOCIÉS AU GRADIENT ÉLEVÉ")
    print("Variables analysées: sexe, poids, taille, IMC, approche chirurgicale,")
    print("spécialité, tabac, alcool, comorbidités, hémoglobine, DFG, brassard,")
    print("diamètre artère, nombre ponctions, distance peau-artère, vasopresseurs")
    print("=" * 80)
    
    # Load data
    print("\nLoading gradient data...")
    gradient_df = load_gradient_data()
    print(f"  Loaded {len(gradient_df)} patients from gradient file")
    
    print("\nLoading PROMISES clinical data...")
    clinical_df = load_and_clean_promises_data()
    print(f"  Loaded {len(clinical_df)} patients from PROMISES data")
    
    # Merge
    print("\nMerging data...")
    merged_df = merge_data(gradient_df, clinical_df)
    print(f"  Merged {len(merged_df)} patients")
    
    # Show matched patients
    matched = merged_df[merged_df['pre_poids'].notna() | merged_df['pre_imc'].notna()]
    print(f"\nMatched patients with clinical data: {len(matched)}")
    
    # ========================================================================
    # ANALYSE DES VARIABLES CATÉGORIELLES
    # ========================================================================
    print("\n" + "=" * 80)
    print("ANALYSE DES VARIABLES CATÉGORIELLES")
    print("=" * 80)
    
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
            
            # Create graph
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
        ('taille_brassard', 'Taille du brassard'),
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
            
            # Create graph
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
        
        # Display top 5 most impactful variables
        significant = [r for r in all_results if r['significant']]
        if significant:
            print("\nTop 5 variables avec le plus grand impact sur le gradient:")
            sorted_by_effect = sorted(significant, key=lambda x: x['effect_size'], reverse=True)[:5]
            for i, r in enumerate(sorted_by_effect, 1):
                print(f"  {i}. {r['variable']}: effect size = {r['effect_size']:.1f}% (p={r['p_value']:.4f})")
    
    # ========================================================================
    # PATIENTS À HAUT RISQUE
    # ========================================================================
    print("\n" + "=" * 80)
    print("PATIENTS À HAUT RISQUE (>50% TEMPS EN GRADIENT)")
    print("=" * 80)
    
    high_risk = merged_df[merged_df['percentage_gradient'] > 50].sort_values('percentage_gradient', ascending=False)
    if len(high_risk) > 0:
        for _, row in high_risk.iterrows():
            patient = row['patient']
            pct = row['percentage_gradient']
            print(f"\n  {patient}: {pct:.1f}%")
            
            if 'sexe' in row and pd.notna(row['sexe']):
                sex = 'Masculin' if row['sexe'] == 1 else 'Féminin'
                print(f"    Sexe: {sex}")
            if 'pre_imc' in row and pd.notna(row['pre_imc']):
                print(f"    IMC: {row['pre_imc']:.1f}")
            if 'pre_spe_chir' in row and pd.notna(row['pre_spe_chir']):
                spe = SPECIALITE_MAP.get(row['pre_spe_chir'], row['pre_spe_chir'])
                print(f"    Spécialité: {spe}")
            if 'approche_chirurgicale' in row and pd.notna(row['approche_chirurgicale']):
                approche = APPROCHE_MAP.get(row['approche_chirurgicale'], row['approche_chirurgicale'])
                print(f"    Approche: {approche}")
            if 'atcd_maladie_coro' in row and row['atcd_maladie_coro'] == 1:
                print(f"    Coronaropathie: Oui")
    
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
    
    if not significant_cat and not significant_cont:
        print("\nAucune variable significativement associée (p < 0.05)")
    
    print("\n" + "=" * 80)
    print(f"Results saved to: {OUTPUT_DIR}")
    print("Files created:")
    print("  - Individual variable graphs (*.png)")
    print("  - comparative_effect_sizes.png (ranking of most impactful variables)")
    print("  - significant_variables_summary.csv (detailed summary table)")
    print("=" * 80)


if __name__ == "__main__":
    main()