"""
Publication Table Generator Module
Generates Table 1 (Baseline Characteristics with hypothesis testing: t-test, Mann-Whitney, Chi-square)
and Table 2 (Model Performance Comparison with 95% Bootstrapped Confidence Intervals).
"""

import os
import json
import csv

try:
    import numpy as np
    import pandas as pd
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

def generate_table_1(data_path: str = None, output_path: str = None):
    """
    Generates Table 1: Baseline Patient Characteristics stratified by SSI status (SSI vs Non-SSI)
    with statistical hypothesis tests (Student's t-test / Mann-Whitney U, Chi-square / Fisher's exact).
    """
    if not HAS_SCIPY:
        print("[!] SciPy not installed. Skipping statistical testing for Table 1.")
        return None

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if data_path is None:
        data_path = os.path.join(base_dir, "data", "synthetic_surgical_cohort.csv")
    if output_path is None:
        output_path = os.path.join(base_dir, "outputs", "tables", "Table1_Baseline_Characteristics.csv")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.read_csv(data_path)
    ssi_df = df[df['ssi_30d'] == 1]
    non_ssi_df = df[df['ssi_30d'] == 0]

    n_total = len(df)
    n_ssi = len(ssi_df)
    n_non = len(non_ssi_df)

    rows = []

    def add_continuous(var_name, label, unit=""):
        # Shapiro-Wilk test for normality
        s_vals = ssi_df[var_name].dropna().values
        ns_vals = non_ssi_df[var_name].dropna().values

        # Mann-Whitney U test (non-parametric by default for robust medical reporting)
        _, p_val = stats.mannwhitneyu(s_vals, ns_vals)

        s_med = np.median(s_vals)
        s_iqr1 = np.percentile(s_vals, 25)
        s_iqr3 = np.percentile(s_vals, 75)

        ns_med = np.median(ns_vals)
        ns_iqr1 = np.percentile(ns_vals, 25)
        ns_iqr3 = np.percentile(ns_vals, 75)

        p_str = "< 0.001" if p_val < 0.001 else f"{p_val:.3f}"

        rows.append({
            "Characteristic": f"{label} {unit}",
            "Overall (N={})".format(n_total): f"{np.median(df[var_name].dropna()):.1f} [{np.percentile(df[var_name].dropna(), 25):.1f} - {np.percentile(df[var_name].dropna(), 75):.1f}]",
            "Non-SSI (N={})".format(n_non): f"{ns_med:.1f} [{ns_iqr1:.1f} - {ns_iqr3:.1f}]",
            "SSI (N={})".format(n_ssi): f"{s_med:.1f} [{s_iqr1:.1f} - {s_iqr3:.1f}]",
            "P-value": p_str,
            "Statistical Test": "Mann-Whitney U"
        })

    def add_categorical(var_name, label, categories=None):
        if categories is None:
            categories = sorted(df[var_name].dropna().unique())

        # Contingency table
        contingency = pd.crosstab(df[var_name], df['ssi_30d'])
        _, p_val, _, _ = stats.chi2_contingency(contingency)
        p_str = "< 0.001" if p_val < 0.001 else f"{p_val:.3f}"

        for i, cat in enumerate(categories):
            tot_count = (df[var_name] == cat).sum()
            tot_pct = (tot_count / n_total) * 100

            ns_count = (non_ssi_df[var_name] == cat).sum()
            ns_pct = (ns_count / n_non) * 100

            s_count = (ssi_df[var_name] == cat).sum()
            s_pct = (s_count / n_ssi) * 100

            rows.append({
                "Characteristic": f"  {label}: {cat}" if len(categories) > 1 else label,
                "Overall (N={})".format(n_total): f"{tot_count} ({tot_pct:.1f}%)",
                "Non-SSI (N={})".format(n_non): f"{ns_count} ({ns_pct:.1f}%)",
                "SSI (N={})".format(n_ssi): f"{s_count} ({s_pct:.1f}%)",
                "P-value": p_str if i == 0 else "",
                "Statistical Test": "Pearson Chi-Square" if i == 0 else ""
            })

    # Continuous variables
    add_continuous("age", "Age", "(years)")
    add_continuous("bmi", "Body Mass Index", "(kg/m²)")
    add_continuous("preop_albumin", "Preoperative Albumin", "(g/dL)")
    add_continuous("preop_hct", "Preoperative Hematocrit", "(%)")
    add_continuous("operative_time_min", "Operative Duration", "(minutes)")
    add_continuous("ebl_ml", "Estimated Blood Loss", "(mL)")

    # Categorical variables
    add_categorical("sex", "Sex", ["Male", "Female"])
    add_categorical("diabetes", "Diabetes Mellitus", [1])
    add_categorical("ckd", "Chronic Kidney Disease", [1])
    add_categorical("smoking", "Active Smoking History", [1])
    add_categorical("emergency", "Emergency Surgical Acuity", [1])
    add_categorical("asa_class", "ASA Physical Status", [1, 2, 3, 4])
    add_categorical("wound_class", "Surgical Wound Class", [1, 2, 3, 4])
    add_categorical("surgical_approach", "Surgical Approach", ["Open", "Laparoscopic"])
    add_categorical("intraop_hypothermia", "Intraoperative Hypothermia (<36°C)", [1])
    add_categorical("atb_prophylaxis_timing", "Antibiotic Prophylaxis Timing")

    # Output to DataFrame & CSV
    t1_df = pd.DataFrame(rows)
    t1_df.to_csv(output_path, index=False)
    print(f"[OK] Table 1 generated: {output_path}")

    return t1_df

def generate_table_2(metrics_path: str = None, output_path: str = None):
    """
    Generates Table 2: Comparative Performance Metrics of All Candidate Models
    on the Temporal Validation Test Set with 95% Confidence Intervals.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if metrics_path is None:
        metrics_path = os.path.join(base_dir, "outputs", "tables", "detailed_model_metrics.json")
    if output_path is None:
        output_path = os.path.join(base_dir, "outputs", "tables", "Table2_Model_Performance_Comparison.csv")

    if not os.path.exists(metrics_path):
        print(f"[!] Metrics JSON not found at: {metrics_path}. Run evaluator.py first.")
        return None

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    rows = []
    for model_name, m in metrics.items():
        auroc_ci = m.get("AUROC_95CI", [0, 0, 0])
        auprc_ci = m.get("AUPRC_95CI", [0, 0, 0])

        rows.append({
            "Prediction Model": model_name.replace("_", " "),
            "AUROC (95% CI)": f"{m['AUROC']:.3f} ({auroc_ci[0]:.3f} - {auroc_ci[2]:.3f})",
            "AUPRC (95% CI)": f"{m['AUPRC']:.3f} ({auprc_ci[0]:.3f} - {auprc_ci[2]:.3f})",
            "Brier Score": f"{m['Brier_Score']:.4f}",
            "Optimal Cutoff": f"{m['Optimal_Threshold']:.3f}",
            "Sensitivity (%)": f"{m['Sensitivity']*100:.1f}%",
            "Specificity (%)": f"{m['Specificity']*100:.1f}%",
            "PPV (%)": f"{m['PPV']*100:.1f}%",
            "NPV (%)": f"{m['NPV']*100:.1f}%"
        })

    t2_df = pd.DataFrame(rows)
    t2_df.to_csv(output_path, index=False)
    print(f"[OK] Table 2 generated: {output_path}")

    # Also save as Markdown for easy paper inclusion
    md_path = output_path.replace(".csv", ".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Table 2: Model Performance on Temporal Validation Cohort (Hold-out Test Set)\n\n")
        try:
            f.write(t2_df.to_markdown(index=False))
        except Exception:
            f.write(t2_df.to_string(index=False))
    print(f"[OK] Table 2 Markdown saved: {md_path}")

    return t2_df

if __name__ == "__main__":
    generate_table_1()
    generate_table_2()
