"""
Explainable AI (XAI) Module using TreeSHAP
Generates Global Feature Importance (Beeswarm) and Local Patient-Level Waterfall Plots.
"""

import os
import pickle

try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

def run_shap_analysis(model=None, X_test=None, output_dir=None):
    """
    Computes TreeSHAP values for the champion model, generates Beeswarm Summary Plot,
    and a local Case-study Waterfall plot.
    """
    if not HAS_SHAP:
        print("[!] Warning: SHAP package is not installed. Please install shap via requirements.txt.")
        return

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if output_dir is None:
        output_dir = os.path.join(base_dir, "outputs", "figures")
    os.makedirs(output_dir, exist_ok=True)

    models_dir = os.path.join(base_dir, "models")
    data_dir = os.path.join(base_dir, "data")

    # Load data if not provided
    if X_test is None:
        from src.model_trainer import encode_features
        test_df = pd.read_csv(os.path.join(data_dir, "test_cohort.csv"))
        X_test, _, feature_names = encode_features(test_df)
        with open(os.path.join(models_dir, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        cont_cols = ["age", "bmi", "preop_albumin", "preop_hct", "preop_wbc", "operative_time_min", "ebl_ml"]
        X_test[cont_cols] = scaler.transform(X_test[cont_cols])

    # Select champion model using champion_info.json or fallback
    if model is None:
        champ_info_p = os.path.join(models_dir, "champion_info.json")
        model_name = None
        if os.path.exists(champ_info_p):
            try:
                import json
                with open(champ_info_p, "r") as f:
                    info = json.load(f)
                    base_file = info.get("base_model_file")
                    if base_file and os.path.exists(os.path.join(models_dir, base_file)):
                        with open(os.path.join(models_dir, base_file), "rb") as f:
                            model = pickle.load(f)
                            model_name = info.get("champion_name", "Champion")
            except Exception:
                model = None

        if model is None:
            for m_name in ["LightGBM", "XGBoost", "Random_Forest"]:
                p = os.path.join(models_dir, f"{m_name}.pkl")
                if os.path.exists(p):
                    with open(p, "rb") as f:
                        model = pickle.load(f)
                        model_name = m_name
                        break

    print(f"[*] Calculating SHAP values for model: {model_name}...")
    if "logistic" in str(type(model)).lower():
        explainer = shap.LinearExplainer(model, X_test)
    else:
        explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # In binary classification, check dimensions
    if len(shap_values.shape) == 3:
        # Some models output shape (samples, features, 2)
        shap_values_to_plot = shap_values[:, :, 1]
    else:
        shap_values_to_plot = shap_values

    # 1. Global Beeswarm Summary Plot (Figure 3)
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(shap_values_to_plot, max_display=18, show=False)
    plt.title(f"Figure 3: Global SHAP Feature Importance Summary ({model_name})", fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig3_path = os.path.join(output_dir, "Figure3_SHAP_Global_Beeswarm.png")
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Figure 3 saved: {fig3_path}")

    # 2. Local Patient-Level Waterfall Plot (Figure 4 - High Risk Case Example)
    # Find a high-risk case in test set
    pred_probs = model.predict_proba(X_test)[:, 1]
    high_risk_idx = int(np.argmax(pred_probs))

    plt.figure(figsize=(9, 6))
    shap.plots.waterfall(shap_values_to_plot[high_risk_idx], max_display=12, show=False)
    plt.title(f"Figure 4: Local SHAP Waterfall Plot (Case #{high_risk_idx} - High SSI Risk)", fontsize=11, fontweight='bold')
    plt.tight_layout()
    fig4_path = os.path.join(output_dir, "Figure4_SHAP_Waterfall_Case.png")
    plt.savefig(fig4_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Figure 4 saved: {fig4_path}")

    # Save explainer artifact
    with open(os.path.join(models_dir, "shap_explainer.pkl"), "wb") as f:
        pickle.dump(explainer, f)

    return shap_values

if __name__ == "__main__":
    run_shap_analysis()
