"""
Clinical Prediction Triad Evaluation Module
Evaluates Discrimination (AUROC, AUPRC), Calibration (Brier score, slope/intercept, plot),
and Clinical Utility via Decision Curve Analysis (DCA). Computes 95% Bootstrap CIs.
"""

import os
import json
import pickle

try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score,
        brier_score_loss, confusion_matrix
    )
    from sklearn.calibration import calibration_curve
    from sklearn.linear_model import LogisticRegression
    HAS_EVAL_DEPS = True
except ImportError:
    HAS_EVAL_DEPS = False

def calculate_optimal_threshold(y_true, y_prob):
    """Calculates optimal cutoff using Youden's J index (Sensitivity + Specificity - 1)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thresh = thresholds[best_idx]
    return float(best_thresh)

def bootstrap_ci(y_true, y_prob, n_bootstraps: int = 500, seed: int = 42):
    """Computes 95% Confidence Intervals for AUROC, AUPRC, and Brier Score via bootstrapping."""
    np.random.seed(seed)
    n = len(y_true)
    aurocs, auprcs, briers = [], [], []

    for _ in range(n_bootstraps):
        indices = np.random.choice(n, size=n, replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue
        y_t = y_true[indices]
        y_p = y_prob[indices]

        aurocs.append(auc(*roc_curve(y_t, y_p)[:2]))
        auprcs.append(average_precision_score(y_t, y_p))
        briers.append(brier_score_loss(y_t, y_p))

    def get_ci(arr):
        return (
            float(np.percentile(arr, 2.5)),
            float(np.percentile(arr, 50.0)),
            float(np.percentile(arr, 97.5))
        )

    return {
        "AUROC_95CI": get_ci(aurocs),
        "AUPRC_95CI": get_ci(auprcs),
        "Brier_95CI": get_ci(briers)
    }

def calculate_dca(y_true, y_prob, thresholds=None):
    """Calculates Decision Curve Analysis (Net Benefit) across threshold probabilities."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.35, 35)

    n = len(y_true)
    prevalence = np.mean(y_true)

    net_benefits_model = []
    net_benefits_all = []

    for pt in thresholds:
        # Model Net Benefit
        pred_pos = (y_prob >= pt).astype(int)
        tp = np.sum((pred_pos == 1) & (y_true == 1))
        fp = np.sum((pred_pos == 1) & (y_true == 0))
        nb = (tp / n) - (fp / n) * (pt / (1.0 - pt))
        net_benefits_model.append(nb)

        # Treat All Net Benefit
        nb_all = prevalence - (1.0 - prevalence) * (pt / (1.0 - pt))
        net_benefits_all.append(nb_all)

    return thresholds, np.array(net_benefits_model), np.array(net_benefits_all)

def run_evaluation(models=None, X_test=None, y_test=None, output_dir=None):
    """
    Runs full triad evaluation on all models, plots publication figures,
    and saves detailed metrics with 95% Confidence Intervals.
    """
    if not HAS_EVAL_DEPS:
        print("[!] Evaluation dependencies missing. Please install requirements.txt.")
        return

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if output_dir is None:
        output_dir = os.path.join(base_dir, "outputs")
    
    figures_dir = os.path.join(output_dir, "figures")
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    # If models not passed, load from disk
    models_dir = os.path.join(base_dir, "models")
    data_dir = os.path.join(base_dir, "data")

    if X_test is None or y_test is None:
        from src.model_trainer import encode_features
        test_df = pd.read_csv(os.path.join(data_dir, "test_cohort.csv"))
        X_test, y_test, _ = encode_features(test_df)
        with open(os.path.join(models_dir, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        cont_cols = ["age", "bmi", "preop_albumin", "preop_hct", "preop_wbc", "operative_time_min", "ebl_ml"]
        X_test[cont_cols] = scaler.transform(X_test[cont_cols])

    if models is None:
        models = {}
        for m_name in ["Logistic_Regression", "Random_Forest", "LightGBM", "XGBoost"]:
            p = os.path.join(models_dir, f"{m_name}.pkl")
            if os.path.exists(p):
                with open(p, "rb") as f:
                    models[m_name] = pickle.load(f)

    # Check if Calibrated Champion model exists and evaluate it
    calib_p = os.path.join(models_dir, "Calibrated_Champion_Model.pkl")
    if os.path.exists(calib_p) and "Calibrated_Champion" not in models:
        try:
            with open(calib_p, "rb") as f:
                models["Calibrated_Champion"] = pickle.load(f)
        except Exception:
            pass

    # 1. Plot ROC and PR Curves (Figure 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = {
        "Logistic_Regression": "#64748B",
        "Random_Forest": "#0EA5E9",
        "LightGBM": "#10B981",
        "XGBoost": "#8B5CF6",
        "CatBoost": "#F59E0B",
        "Calibrated_Champion": "#EC4899"
    }
    full_metrics = {}

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auroc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, label=f"{name} (AUROC = {auroc:.3f})", color=colors.get(name, '#6366F1'), lw=2)

        # Precision-Recall Curve
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        auprc = average_precision_score(y_test, y_prob)
        ax2.plot(rec, prec, label=f"{name} (AUPRC = {auprc:.3f})", color=colors.get(name, '#6366F1'), lw=2)

        # Cutoff & Classification
        opt_thresh = calculate_optimal_threshold(y_test, y_prob)
        y_pred = (y_prob >= opt_thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        sens = tp / max(1, (tp + fn))
        spec = tn / max(1, (tn + fp))
        ppv = tp / max(1, (tp + fp))
        npv = tn / max(1, (tn + fn))

        # Bootstrap CI
        cis = bootstrap_ci(y_test, y_prob, n_bootstraps=200)

        full_metrics[name] = {
            "AUROC": round(auroc, 4),
            "AUROC_95CI": [round(x, 4) for x in cis["AUROC_95CI"]],
            "AUPRC": round(auprc, 4),
            "AUPRC_95CI": [round(x, 4) for x in cis["AUPRC_95CI"]],
            "Brier_Score": round(brier_score_loss(y_test, y_prob), 4),
            "Optimal_Threshold": round(opt_thresh, 4),
            "Sensitivity": round(sens, 4),
            "Specificity": round(spec, 4),
            "PPV": round(ppv, 4),
            "NPV": round(npv, 4)
        }

    # Reference lines
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Chance (0.50)')
    ax1.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11, fontweight='bold')
    ax1.set_title('A: Receiver Operating Characteristic (ROC) Curves', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', frameon=True)
    ax1.grid(True, alpha=0.2)

    base_prev = np.mean(y_test)
    ax2.axhline(base_prev, color='r', linestyle='--', alpha=0.5, label=f'Baseline Rate ({base_prev:.1%})')
    ax2.set_xlabel('Recall (Sensitivity)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Precision (Positive Predictive Value)', fontsize=11, fontweight='bold')
    ax2.set_title('B: Precision-Recall (PR) Curves (Imbalance Aware)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', frameon=True)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    fig1_path = os.path.join(figures_dir, "Figure1_Discrimination_Curves.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"[OK] Figure 1 saved: {fig1_path}")

    # 2. Plot Calibration & Decision Curve Analysis (Figure 2)
    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))

    # Calibration Plot
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Perfect Calibration (Slope=1, Intercept=0)')
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_prob, n_bins=8, strategy='uniform')
        style = "o--" if "Calibrated" in name else "s-"
        lw = 2.5 if "Calibrated" in name else 1.8
        ax3.plot(mean_predicted_value, fraction_of_positives, style, label=f"{name}", color=colors.get(name, '#6366F1'), lw=lw)

    ax3.set_xlabel('Mean Predicted SSI Risk', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Observed SSI Proportion', fontsize=11, fontweight='bold')
    ax3.set_title('A: Calibration Curves (Reliability Diagram)', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left', frameon=True)
    ax3.grid(True, alpha=0.2)

    # DCA Plot
    thresholds, _, net_benefits_all = calculate_dca(y_test, list(models.values())[0].predict_proba(X_test)[:, 1])
    ax4.plot(thresholds * 100, np.zeros_like(thresholds), 'k-', alpha=0.7, label='Treat None (Net Benefit = 0)')
    ax4.plot(thresholds * 100, net_benefits_all, 'r--', alpha=0.6, label='Treat All')

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        _, nb_model, _ = calculate_dca(y_test, y_prob, thresholds=thresholds)
        ax4.plot(thresholds * 100, nb_model, label=f"{name}", color=colors.get(name, '#6366F1'), lw=2)

    ax4.set_xlim([0, 30])
    ax4.set_ylim([-0.02, max(base_prev * 1.2, 0.08)])
    ax4.set_xlabel('Threshold Probability pt (%)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Net Benefit', fontsize=11, fontweight='bold')
    ax4.set_title('B: Decision Curve Analysis (Clinical Utility)', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', frameon=True)
    ax4.grid(True, alpha=0.2)

    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, "Figure2_Calibration_and_DCA.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"[OK] Figure 2 saved: {fig2_path}")

    # Save detailed JSON metrics
    with open(os.path.join(tables_dir, "detailed_model_metrics.json"), "w") as f:
        json.dump(full_metrics, f, indent=2)

    print("[OK] Detailed model evaluation completed with Bootstrapped 95% CIs.")

    # 3. Run Subgroup & Sensitivity Analysis (Table 3)
    best_eval_model = models.get("Calibrated_Champion") or list(models.values())[0]
    run_subgroup_analysis(champion_model=best_eval_model, X_test=X_test, y_test=y_test, output_dir=tables_dir)

    return full_metrics

def run_subgroup_analysis(champion_model=None, X_test=None, y_test=None, output_dir=None):
    """
    Evaluates champion model across surgical subgroups (Specialty, Approach, Acuity, Wound Class)
    and saves Table 3 in CSV and Markdown formats.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if output_dir is None:
        output_dir = os.path.join(base_dir, "outputs", "tables")
    os.makedirs(output_dir, exist_ok=True)

    models_dir = os.path.join(base_dir, "models")
    if champion_model is None:
        for m_file in ["Calibrated_Champion_Model.pkl", "LightGBM.pkl", "XGBoost.pkl", "Random_Forest.pkl"]:
            p = os.path.join(models_dir, m_file)
            if os.path.exists(p):
                with open(p, "rb") as f:
                    champion_model = pickle.load(f)
                    break

    subgroup_definitions = [
        ("Overall Cohort", "All Test Patients", np.ones(len(y_test), dtype=bool)),
        ("Surgical Specialty", "Colorectal Surgery", X_test["specialty_Colorectal"] == 1),
        ("Surgical Specialty", "Non-Colorectal Surgery", X_test["specialty_Colorectal"] == 0),
        ("Surgical Approach", "Open Laparotomy", X_test["approach_open"] == 1),
        ("Surgical Approach", "Laparoscopic / Minimally Invasive", X_test["approach_open"] == 0),
        ("Surgical Acuity", "Emergency Surgery", X_test["emergency"] == 1),
        ("Surgical Acuity", "Elective Surgery", X_test["emergency"] == 0),
        ("Wound Contamination", "Clean / Clean-Contaminated (Class I-II)", X_test["wound_class"] <= 2),
        ("Wound Contamination", "Contaminated / Dirty (Class III-IV)", X_test["wound_class"] >= 3),
        ("Patient Sex", "Male", X_test["sex_male"] == 1),
        ("Patient Sex", "Female", X_test["sex_male"] == 0),
    ]

    records = []
    for category, subgroup_name, mask in subgroup_definitions:
        sub_X = X_test[mask]
        sub_y = y_test[mask]
        n_total = len(sub_y)
        n_ssi = int(np.sum(sub_y))
        ssi_rate = (n_ssi / n_total) * 100 if n_total > 0 else 0.0

        if n_ssi > 1 and (n_total - n_ssi) > 1:
            y_prob = champion_model.predict_proba(sub_X)[:, 1]
            auroc = roc_auc_score(sub_y, y_prob)
            auprc = average_precision_score(sub_y, y_prob)
            brier = brier_score_loss(sub_y, y_prob)
            opt_thresh = calculate_optimal_threshold(sub_y, y_prob)
            y_pred = (y_prob >= opt_thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(sub_y, y_pred).ravel()
            sens = (tp / max(1, (tp + fn))) * 100
            spec = (tn / max(1, (tn + fp))) * 100
        else:
            auroc, auprc, brier, sens, spec = np.nan, np.nan, np.nan, np.nan, np.nan

        records.append({
            "Subgroup Category": category,
            "Subgroup": subgroup_name,
            "N Total": n_total,
            "SSI Cases (N, %)": f"{n_ssi} ({ssi_rate:.1f}%)",
            "AUROC": f"{auroc:.3f}" if not np.isnan(auroc) else "N/A",
            "AUPRC": f"{auprc:.3f}" if not np.isnan(auprc) else "N/A",
            "Brier Score": f"{brier:.4f}" if not np.isnan(brier) else "N/A",
            "Sensitivity (%)": f"{sens:.1f}%" if not np.isnan(sens) else "N/A",
            "Specificity (%)": f"{spec:.1f}%" if not np.isnan(spec) else "N/A"
        })

    subgroup_df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "Table3_Subgroup_Analysis.csv")
    subgroup_df.to_csv(csv_path, index=False)
    print(f"[OK] Table 3 CSV saved: {csv_path}")

    # Generate Markdown Table
    md_lines = [
        "# Table 3: Subgroup and Sensitivity Analysis on Temporal Validation Cohort",
        "",
        "| Category | Subgroup | N Total | SSI Rate | AUROC | AUPRC | Brier Score | Sensitivity | Specificity |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    for r in records:
        md_lines.append(
            f"| {r['Subgroup Category']} | {r['Subgroup']} | {r['N Total']} | {r['SSI Cases (N, %)']} | {r['AUROC']} | {r['AUPRC']} | {r['Brier Score']} | {r['Sensitivity (%)']} | {r['Specificity (%)']} |"
        )

    md_path = os.path.join(output_dir, "Table3_Subgroup_Analysis.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"[OK] Table 3 Markdown saved: {md_path}")

    return subgroup_df

if __name__ == "__main__":
    run_evaluation()
