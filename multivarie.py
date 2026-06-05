"""
ANALYSE MULTIVARIEE - TOUS LES PATIENTS (n=50)
Version corrigee avec visualisations fonctionnelles
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
import re
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(r"C:\Users\katia\Desktop\min data\work\data\Patients-infinity\filtered")
PROMISES_FILE = BASE_DIR / "PROMISES_DATA_2025-08-18_1050.csv"
GRADIENT_FILE = BASE_DIR / "gradient" / "gradient_statistics.xlsx"
OUTPUT_DIR = BASE_DIR / "gradient" / "multivariate_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Charger et préparer les données - TOUS LES 50 PATIENTS"""
    
    print("Loading gradient data...")
    df_grad = pd.read_excel(GRADIENT_FILE, sheet_name='Patients')
    
    def extract_patient_num(name):
        if pd.isna(name):
            return None
        match = re.search(r'(?:Patient|patient|PROMISES)\s+(\d+)', str(name))
        return match.group(1) if match else None
    
    df_grad['patient_num'] = df_grad['patient'].apply(extract_patient_num)
    df_grad = df_grad.dropna(subset=['patient_num'])
    df_grad = df_grad[['patient', 'patient_num', 'percentage_gradient']]
    
    print(f"  Gradient data: {len(df_grad)} patients")
    
    print("Loading clinical data...")
    df_clin = pd.read_csv(PROMISES_FILE, encoding='utf-8-sig')
    df_clin.columns = df_clin.columns.str.replace('ï»¿', '')
    df_clin = df_clin.dropna(subset=['record_id'])
    df_clin['patient_num'] = df_clin['record_id'].astype(int).astype(str)
    
    if 'sexe' in df_clin.columns:
        df_clin['sexe'] = pd.to_numeric(df_clin['sexe'], errors='coerce')
    
    columns_to_keep = [
        'patient_num', 'sexe',
        'pre_taille',
        'approche_chirurgicale',
        'atcd_maladie_coro'
    ]
    
    existing_cols = [col for col in columns_to_keep if col in df_clin.columns]
    df_clin = df_clin[existing_cols]
    
    if 'pre_taille' in df_clin.columns:
        df_clin['pre_taille'] = pd.to_numeric(df_clin['pre_taille'], errors='coerce')
    
    df_merged = df_grad.merge(df_clin, on='patient_num', how='left')
    
    df_merged['sexe_masculin'] = (df_merged['sexe'] == 1).astype(float)
    df_merged['approche_laparotomie'] = (df_merged['approche_chirurgicale'] == 1).astype(float)
    df_merged['coronaropathie'] = df_merged['atcd_maladie_coro'].astype(float)
    
    print(f"  Merged: {len(df_merged)} patients")
    
    return df_merged


def impute_missing_data(df):
    """Imputer TOUTES les données manquantes"""
    
    df = df.copy()
    
    print("\n" + "=" * 80)
    print("DONNEES MANQUANTES AVANT IMPUTATION")
    print("=" * 80)
    
    for col in ['sexe_masculin', 'approche_laparotomie', 'coronaropathie', 'pre_taille']:
        if col in df.columns:
            missing = df[col].isna().sum()
            print(f"  {col}: {missing} missing ({missing/len(df)*100:.1f}%)")
    
    if 'pre_taille' in df.columns:
        median_val = df['pre_taille'].median()
        df['pre_taille'].fillna(median_val, inplace=True)
        print(f"\n  pre_taille: imputed with median={median_val:.1f} cm")
    
    for col in ['sexe_masculin', 'approche_laparotomie', 'coronaropathie']:
        if col in df.columns:
            df[col].fillna(0, inplace=True)
            print(f"  {col}: imputed with 0")
    
    if 'pre_taille' in df.columns:
        mean_taille = df['pre_taille'].mean()
        std_taille = df['pre_taille'].std()
        df['pre_taille_std'] = (df['pre_taille'] - mean_taille) / std_taille
        print(f"\n  Taille standardisee: mean={mean_taille:.1f} cm, std={std_taille:.1f} cm")
    
    return df


def univariate_analysis(df, target='percentage_gradient'):
    """Analyses univariées"""
    
    results = []
    
    categorical_vars = ['sexe_masculin', 'approche_laparotomie', 'coronaropathie']
    
    for var in categorical_vars:
        if var in df.columns:
            group0 = df[df[var] == 0][target].dropna()
            group1 = df[df[var] == 1][target].dropna()
            if len(group0) > 0 and len(group1) > 0:
                t_stat, p_val = stats.ttest_ind(group0, group1)
                results.append({
                    'Variable': var,
                    'Type': 'Categorical',
                    'Mean_without': round(group0.mean(), 1),
                    'Mean_with': round(group1.mean(), 1),
                    'Difference': round(group1.mean() - group0.mean(), 1),
                    'P_value': p_val,
                    'Significant': p_val < 0.05
                })
    
    if 'pre_taille_std' in df.columns:
        clean_df = df[['pre_taille_std', target]].dropna()
        if len(clean_df) > 2:
            corr, p_val = stats.pearsonr(clean_df['pre_taille_std'], clean_df[target])
            results.append({
                'Variable': 'pre_taille',
                'Type': 'Continuous',
                'Correlation': round(corr, 3),
                'P_value': p_val,
                'Significant': p_val < 0.05
            })
    
    return pd.DataFrame(results)


def multivariate_regression_simple(df, target='percentage_gradient'):
    """Régression linéaire multiple"""
    
    predictors = ['sexe_masculin', 'approche_laparotomie', 'coronaropathie', 'pre_taille_std']
    
    print(f"\nVariables incluses dans le modele multivarie:")
    for v in predictors:
        print(f"  - {v}")
    
    X = df[predictors].copy()
    y = df[target].copy()
    
    # Supprimer les lignes avec des NaN
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]
    
    print(f"Nombre de patients dans le modele: {len(X)}")
    
    X_with_const = np.column_stack([np.ones(len(X)), X.values])
    
    # Moindres carrés
    beta, residuals, rank, s = np.linalg.lstsq(X_with_const, y, rcond=None)
    
    y_pred = np.dot(X_with_const, beta)
    residuals = y - y_pred
    n = len(y)
    k = len(predictors)
    
    mse = np.sum(residuals**2) / (n - k - 1)
    
    XtX = np.dot(X_with_const.T, X_with_const)
    XtX_inv = np.linalg.pinv(XtX)
    var_beta = np.diag(XtX_inv) * mse
    se_beta = np.sqrt(var_beta)
    
    t_stats = beta / se_beta
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k - 1))
    
    ci_low = beta - 1.96 * se_beta
    ci_high = beta + 1.96 * se_beta
    
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot)
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    
    results = []
    for i, var in enumerate(['Intercept'] + predictors):
        results.append({
            'Variable': var,
            'Coefficient': round(beta[i], 2),
            'IC_95_inf': round(ci_low[i], 2),
            'IC_95_sup': round(ci_high[i], 2),
            'P_value': p_values[i],
            'Significant': p_values[i] < 0.05
        })
    
    model_summary = {
        'R_squared': r2,
        'R_squared_adjusted': r2_adj,
        'n_observations': n,
        'n_predictors': k
    }
    
    return results, model_summary, (y, y_pred)


def create_forest_plot(results, n_patients, output_dir):
    """Créer un forest plot des coefficients"""
    
    results_no_intercept = [r for r in results if r['Variable'] != 'Intercept']
    
    if len(results_no_intercept) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    var_names = {
        'sexe_masculin': 'Sexe masculin',
        'approche_laparotomie': 'Laparotomie',
        'coronaropathie': 'Maladie coronaire',
        'pre_taille_std': 'Taille (par ecart-type)'
    }
    
    variables = []
    coefficients = []
    ci_low = []
    ci_high = []
    p_values = []
    
    for r in results_no_intercept:
        variables.append(var_names.get(r['Variable'], r['Variable']))
        coefficients.append(r['Coefficient'])
        ci_low.append(r['IC_95_inf'])
        ci_high.append(r['IC_95_sup'])
        p_values.append(r['P_value'])
    
    y_pos = np.arange(len(variables))
    
    # Plot horizontal bars for confidence intervals
    for i, (coef, low, high, p) in enumerate(zip(coefficients, ci_low, ci_high, p_values)):
        color = 'darkred' if p < 0.05 else 'steelblue'
        ax.hlines(y=i, xmin=low, xmax=high, colors=color, linewidth=2)
        ax.plot(coef, i, 'o', color=color, markersize=8)
    
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(variables)
    ax.set_xlabel('Coefficient (changement en % du gradient)', fontsize=12)
    ax.set_title(f'Analyse multivariee: Facteurs associes au gradient eleve\n(n = {n_patients} patients)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add p-value labels
    for i, (coef, p) in enumerate(zip(coefficients, p_values)):
        if p < 0.05:
            ax.text(coef + 2, i, f'p={p:.4f}', va='center', fontsize=9, fontweight='bold')
        else:
            ax.text(coef + 2, i, f'p={p:.4f}', va='center', fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'multivariate_forest_plot.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_predicted_plot(y, y_pred, output_dir):
    """Créer un graphique des valeurs prédites vs observées"""
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(y_pred, y, alpha=0.6, color='steelblue', s=50)
    
    z = np.polyfit(y_pred, y, 1)
    p = np.poly1d(z)
    ax.plot([0, 100], [p(0), p(100)], 'r--', linewidth=2)
    
    ax.plot([0, 100], [0, 100], 'k-', linewidth=1, alpha=0.5, label='Identite')
    
    ax.set_xlabel('Gradient predit (%)', fontsize=12)
    ax.set_ylabel('Gradient observe (%)', fontsize=12)
    ax.set_title('Valeurs predites vs observees', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot)
    ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'multivariate_predicted_vs_observed.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_univariate_plot(df, output_dir):
    """Créer des boxplots pour les variables univariées significatives"""
    
    # Boxplot par sexe
    fig, ax = plt.subplots(figsize=(8, 6))
    data_m = df[df['sexe_masculin'] == 1]['percentage_gradient'].dropna()
    data_f = df[df['sexe_masculin'] == 0]['percentage_gradient'].dropna()
    bp = ax.boxplot([data_m, data_f], labels=['Masculin', 'Feminin'])
    ax.set_ylabel('Temps en gradient (%)')
    ax.set_title('Gradient par sexe (univarie)')
    ax.grid(True, alpha=0.3)
    if len(data_m) > 0 and len(data_f) > 0:
        t_stat, p_val = stats.ttest_ind(data_m, data_f)
        ax.text(0.5, 0.95, f'p = {p_val:.4f}', transform=ax.transAxes, ha='center')
    plt.tight_layout()
    plt.savefig(output_dir / 'univariate_gradient_by_sex.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Boxplot par approche chirurgicale
    fig, ax = plt.subplots(figsize=(8, 6))
    data_laparo = df[df['approche_laparotomie'] == 1]['percentage_gradient'].dropna()
    data_laparo_scopy = df[df['approche_laparotomie'] == 0]['percentage_gradient'].dropna()
    bp = ax.boxplot([data_laparo_scopy, data_laparo], labels=['Laparoscopie', 'Laparotomie'])
    ax.set_ylabel('Temps en gradient (%)')
    ax.set_title('Gradient par approche chirurgicale (univarie)')
    ax.grid(True, alpha=0.3)
    if len(data_laparo) > 0 and len(data_laparo_scopy) > 0:
        t_stat, p_val = stats.ttest_ind(data_laparo_scopy, data_laparo)
        ax.text(0.5, 0.95, f'p = {p_val:.4f}', transform=ax.transAxes, ha='center')
    plt.tight_layout()
    plt.savefig(output_dir / 'univariate_gradient_by_approach.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Boxplot par coronaropathie
    fig, ax = plt.subplots(figsize=(8, 6))
    data_cad = df[df['coronaropathie'] == 1]['percentage_gradient'].dropna()
    data_no_cad = df[df['coronaropathie'] == 0]['percentage_gradient'].dropna()
    bp = ax.boxplot([data_no_cad, data_cad], labels=['Sans coronaropathie', 'Avec coronaropathie'])
    ax.set_ylabel('Temps en gradient (%)')
    ax.set_title('Gradient par maladie coronaire (univarie)')
    ax.grid(True, alpha=0.3)
    if len(data_cad) > 0 and len(data_no_cad) > 0:
        t_stat, p_val = stats.ttest_ind(data_no_cad, data_cad)
        ax.text(0.5, 0.95, f'p = {p_val:.4f}', transform=ax.transAxes, ha='center')
    plt.tight_layout()
    plt.savefig(output_dir / 'univariate_gradient_by_cad.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 80)
    print("ANALYSE MULTIVARIEE - TOUS LES PATIENTS (n=50)")
    print("=" * 80)
    
    # Charger les données
    print("\nLoading data...")
    df = load_data()
    print(f"\nTOTAL PATIENTS: {len(df)}")
    
    # Imputer les données manquantes
    df = impute_missing_data(df)
    
    # Statistiques descriptives du gradient
    print("\n" + "=" * 80)
    print("STATISTIQUES DESCRIPTIVES")
    print("=" * 80)
    print(f"  Mean: {df['percentage_gradient'].mean():.1f}%")
    print(f"  Median: {df['percentage_gradient'].median():.1f}%")
    print(f"  Std: {df['percentage_gradient'].std():.1f}%")
    print(f"  Min: {df['percentage_gradient'].min():.1f}%")
    print(f"  Max: {df['percentage_gradient'].max():.1f}%")
    
    # Analyse univariée
    print("\n" + "=" * 80)
    print("ANALYSE UNIVARIEE")
    print("=" * 80)
    univ_results = univariate_analysis(df)
    print(univ_results.to_string(index=False))
    univ_results.to_csv(OUTPUT_DIR / 'univariate_results.csv', index=False)
    
    # Créer les graphiques univariés
    create_univariate_plot(df, OUTPUT_DIR)
    
    # Analyse multivariée
    print("\n" + "=" * 80)
    print("ANALYSE MULTIVARIEE (REGRESSION LINEAIRE MULTIPLE)")
    print("=" * 80)
    
    results, model_summary, model_data = multivariate_regression_simple(df)
    
    print(f"\nR-squared: {model_summary['R_squared']:.3f}")
    print(f"Adjusted R-squared: {model_summary['R_squared_adjusted']:.3f}")
    print(f"Number of observations: {model_summary['n_observations']}")
    
    print("\nResultats detailles:")
    print("-" * 50)
    for r in results:
        if r['Variable'] == 'Intercept':
            print(f"  Intercept: {r['Coefficient']:.2f} (p={r['P_value']:.4f})")
        else:
            sig = "✓" if r['Significant'] else " "
            print(f"  {sig} {r['Variable']}: {r['Coefficient']:.2f} "
                  f"[IC95: {r['IC_95_inf']:.2f} to {r['IC_95_sup']:.2f}] "
                  f"(p={r['P_value']:.4f})")
    
    # Visualisations
    print("\nCreating visualizations...")
    create_forest_plot(results, model_summary['n_observations'], OUTPUT_DIR)
    create_predicted_plot(model_data[0], model_data[1], OUTPUT_DIR)
    
    # Sauvegarder les résultats
    df_results = pd.DataFrame([r for r in results if r['Variable'] != 'Intercept'])
    df_results.to_csv(OUTPUT_DIR / 'multivariate_results.csv', index=False)
    
    with open(OUTPUT_DIR / 'model_summary.txt', 'w') as f:
        f.write("MODEL SUMMARY\n")
        f.write("=" * 50 + "\n")
        f.write(f"R-squared: {model_summary['R_squared']:.3f}\n")
        f.write(f"Adjusted R-squared: {model_summary['R_squared_adjusted']:.3f}\n")
        f.write(f"Number of observations: {model_summary['n_observations']}\n")
        f.write(f"Number of predictors: {model_summary['n_predictors']}\n\n")
        f.write("COEFFICIENTS\n")
        f.write("-" * 50 + "\n")
        for r in results:
            f.write(f"{r['Variable']}: {r['Coefficient']:.2f} (p={r['P_value']:.4f})\n")
    
    print("\n" + "=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)
    
    significant = [r for r in results if r['Variable'] != 'Intercept' and r['Significant']]
    if significant:
        print("\n✓ Facteurs independamment associes au gradient eleve (p < 0.05):")
        for r in significant:
            coef = r['Coefficient']
            if coef > 0:
                print(f"  - {r['Variable']}: +{coef:.1f}% (IC95: {r['IC_95_inf']:.1f} to {r['IC_95_sup']:.1f}, p={r['P_value']:.4f})")
            else:
                print(f"  - {r['Variable']}: {coef:.1f}% (IC95: {r['IC_95_inf']:.1f} to {r['IC_95_sup']:.1f}, p={r['P_value']:.4f})")
    else:
        print("\n✗ Aucun facteur significatif apres ajustement multivarie (p > 0.05)")
        print("\n  Tendances (p < 0.10):")
        for r in results:
            if r['Variable'] != 'Intercept' and r['P_value'] < 0.10:
                print(f"  - {r['Variable']}: {r['Coefficient']:.1f}% (p={r['P_value']:.4f})")
    
    print(f"\nLe modele explique {model_summary['R_squared']*100:.1f}% de la variance du gradient")
    print(f"Nombre de patients dans l'analyse: {model_summary['n_observations']}")
    
    print("\n" + "=" * 80)
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()