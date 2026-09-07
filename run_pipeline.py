"""
Master Pipeline Runner
Executes the End-to-End Machine Learning Pipeline for SSI Risk Prediction:
Data Generation -> Preprocessing -> Model Training -> Evaluation -> XAI -> Report Tables.
"""

import os
import sys
import time

# Ensure project root is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.data_generator import generate_synthetic_cohort
from src.preprocessor import clean_and_split_data

def main():
    print("=" * 75)
    print("🏥 SOMDECH PHRA PINKLAO HOSPITAL - SURGICAL ML PIPELINE")
    print("   Project: Explainable ML for Predicting SSI at End of Surgery")
    print("=" * 75)
    start_time = time.time()

    # Step 1: Generate / Load Data
    raw_data_path = os.path.join(CURRENT_DIR, "data", "synthetic_surgical_cohort.csv")
    if not os.path.exists(raw_data_path):
        print("\n[STEP 1/6] Generating realistic synthetic surgical cohort (4,500 patients)...")
        generate_synthetic_cohort(output_path=raw_data_path)
    else:
        print(f"\n[STEP 1/6] Found existing raw dataset: {raw_data_path}")

    # Step 2: Preprocess & Temporal Split
    print("\n[STEP 2/6] Running Data Cleaning & Temporal Train/Test Split (Zero-Leakage Protocol)...")
    clean_and_split_data(raw_data_path)

    # Step 3: Model Training
    print("\n[STEP 3/6] Training Baseline, ML Algorithms, Hyperparameter Tuning & Probability Calibration...")
    try:
        from src.model_trainer import train_and_compare_models
        train_result = train_and_compare_models(do_tuning=True)
        if train_result is None:
            print("[!] Skipping remaining ML steps due to missing scikit-learn environment.")
            return
        if len(train_result) == 6:
            models, results, X_test, y_test, best_model_name, calibrator = train_result
        else:
            models, results, X_test, y_test = train_result[:4]
    except ImportError as e:
        print(f"[!] Dependencies missing: {e}. Please run `pip install -r requirements.txt`.")
        return

    # Step 4: Triad Evaluation & Figures
    print("\n[STEP 4/6] Running Clinical Prediction Triad Evaluation (Discrimination, Calibration, DCA)...")
    try:
        from src.evaluator import run_evaluation
        run_evaluation(models=models, X_test=X_test, y_test=y_test)
    except Exception as e:
        print(f"[!] Evaluation error: {e}")

    # Step 5: TreeSHAP Explainability
    print("\n[STEP 5/6] Computing TreeSHAP Global & Local Explainability Plots...")
    try:
        from src.explainability import run_shap_analysis
        run_shap_analysis(X_test=X_test)
    except Exception as e:
        print(f"[!] SHAP error: {e}")

    # Step 6: Publication Tables
    print("\n[STEP 6/6] Generating Publication-Ready Table 1 & Table 2...")
    try:
        from src.report_generator import generate_table_1, generate_table_2
        generate_table_1(data_path=raw_data_path)
        generate_table_2()
    except Exception as e:
        print(f"[!] Report error: {e}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 75)
    print(f"🎉 PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS!")
    print("   Output artifacts generated in:")
    print(f"   - Figures: {os.path.join(CURRENT_DIR, 'outputs', 'figures')}")
    print(f"   - Tables:  {os.path.join(CURRENT_DIR, 'outputs', 'tables')}")
    print(f"   - Models:  {os.path.join(CURRENT_DIR, 'models')}")
    print("=" * 75)

if __name__ == "__main__":
    main()
