"""
Model Training & Optimization Module
Trains Multivariable Logistic Regression (Baseline), Random Forest,
LightGBM, XGBoost, and CatBoost with:
1. Class Imbalance handling (scale_pos_weight / class_weight)
2. Automated Hyperparameter Tuning (5-fold Stratified CV optimizing AUPRC)
3. Probability Calibration (CalibratedClassifierCV with Isotonic / Platt scaling)
"""

import os
import csv
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

try:
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

def encode_features(df):
    """
    Transforms raw columns into a numeric feature matrix X and target vector y.
    """
    df = df.copy()

    # Binary mappings
    df['sex_male'] = (df['sex'] == 'Male').astype(int)
    df['approach_open'] = (df['surgical_approach'] == 'Open').astype(int)

    # One-hot encode surgical specialty
    specialties = ["Colorectal", "Hepatobiliary", "Upper_GI", "Breast_Endocrine", "Vascular", "General"]
    for s in specialties:
        df[f'specialty_{s}'] = (df['surgical_specialty'] == s).astype(int)

    # One-hot encode antibiotic timing
    timings = ["Compliant_Within_60min", "Late_After_Incision", "None_Given"]
    for t in timings:
        df[f'atb_{t}'] = (df['atb_prophylaxis_timing'] == t).astype(int)

    feature_cols = [
        "age", "bmi", "sex_male", "diabetes", "ckd", "cirrhosis",
        "malignancy", "smoking", "steroid_immunosuppressant",
        "asa_class", "emergency", "preop_albumin", "preop_hct", "preop_wbc",
        "wound_class", "approach_open", "operative_time_min", "ebl_ml",
        "intraop_hypothermia", "intraop_hypotension", "intraop_transfusion",
        "atb_redosing_needed", "atb_redosing_given"
    ] + [f'specialty_{s}' for s in specialties] + [f'atb_{t}' for t in timings]

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    X = df[feature_cols].copy()
    y = df['ssi_30d'].astype(int).values if 'ssi_30d' in df.columns else None

    return X, y, feature_cols

def tune_models(X_train, y_train, scale_pos_weight, n_iter: int = 12, seed: int = 42):
    """
    Performs Stratified 5-Fold Cross-Validation hyperparameter search
    optimizing for average_precision (AUPRC) on the imbalanced training set.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    tuned_params = {}

    print("\n" + "-" * 65)
    print("🔍 [Hyperparameter Tuning] Optimizing models via Stratified 5-Fold CV (Metric: AUPRC)")
    print("-" * 65)

    # 1. Tune Random Forest
    print("[*] Tuning Random Forest...")
    rf_param_dist = {
        'n_estimators': [150, 250, 350],
        'max_depth': [5, 7, 9, 12],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    rf_search = RandomizedSearchCV(
        RandomForestClassifier(class_weight='balanced', random_state=seed, n_jobs=-1),
        param_distributions=rf_param_dist,
        n_iter=min(n_iter, 10),
        scoring='average_precision',
        cv=cv,
        random_state=seed,
        n_jobs=-1
    )
    rf_search.fit(X_train, y_train)
    tuned_params['Random_Forest'] = rf_search.best_params_
    print(f"    - Best RF AUPRC (CV): {rf_search.best_score_:.4f}")

    # 2. Tune LightGBM
    if HAS_LGB:
        print("[*] Tuning LightGBM...")
        lgb_param_dist = {
            'n_estimators': [120, 180, 250],
            'learning_rate': [0.02, 0.04, 0.07, 0.1],
            'max_depth': [3, 4, 5, 6],
            'num_leaves': [15, 25, 31, 45],
            'subsample': [0.7, 0.85, 1.0],
            'colsample_bytree': [0.7, 0.85, 1.0],
            'scale_pos_weight': [scale_pos_weight * 0.8, scale_pos_weight, scale_pos_weight * 1.2]
        }
        lgb_search = RandomizedSearchCV(
            lgb.LGBMClassifier(random_state=seed, verbosity=-1),
            param_distributions=lgb_param_dist,
            n_iter=n_iter,
            scoring='average_precision',
            cv=cv,
            random_state=seed,
            n_jobs=-1
        )
        lgb_search.fit(X_train, y_train)
        tuned_params['LightGBM'] = lgb_search.best_params_
        print(f"    - Best LightGBM AUPRC (CV): {lgb_search.best_score_:.4f}")

    # 3. Tune XGBoost
    if HAS_XGB:
        print("[*] Tuning XGBoost...")
        xgb_param_dist = {
            'n_estimators': [120, 180, 250],
            'learning_rate': [0.02, 0.04, 0.07, 0.1],
            'max_depth': [3, 4, 5],
            'subsample': [0.7, 0.85, 1.0],
            'colsample_bytree': [0.7, 0.85, 1.0],
            'scale_pos_weight': [scale_pos_weight * 0.8, scale_pos_weight, scale_pos_weight * 1.2]
        }
        xgb_search = RandomizedSearchCV(
            xgb.XGBClassifier(random_state=seed, eval_metric='logloss'),
            param_distributions=xgb_param_dist,
            n_iter=n_iter,
            scoring='average_precision',
            cv=cv,
            random_state=seed,
            n_jobs=-1
        )
        xgb_search.fit(X_train, y_train)
        tuned_params['XGBoost'] = xgb_search.best_params_
        print(f"    - Best XGBoost AUPRC (CV): {xgb_search.best_score_:.4f}")

    return tuned_params

def train_and_compare_models(data_dir: str = None, models_dir: str = None, do_tuning: bool = True):
    """
    Trains all candidate models, tunes hyperparameters, applies Probability Calibration,
    evaluates on Temporal Test Set, and saves model artifacts.
    """
    if not HAS_SKLEARN:
        print("[!] Warning: scikit-learn is not installed in the active environment.")
        print("    Please activate the virtual environment and install requirements.txt first.")
        return None

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if data_dir is None:
        data_dir = os.path.join(base_dir, "data")
    if models_dir is None:
        models_dir = os.path.join(base_dir, "models")

    os.makedirs(models_dir, exist_ok=True)

    train_path = os.path.join(data_dir, "train_cohort.csv")
    test_path = os.path.join(data_dir, "test_cohort.csv")

    print(f"[*] Loading training data: {train_path}")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train_raw, y_train, feature_names = encode_features(train_df)
    X_test_raw, y_test, _ = encode_features(test_df)

    # Standardize continuous variables
    continuous_cols = ["age", "bmi", "preop_albumin", "preop_hct", "preop_wbc", "operative_time_min", "ebl_ml"]
    scaler = StandardScaler()
    
    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()

    X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
    X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])

    # Save feature names and scaler
    with open(os.path.join(models_dir, "feature_names.json"), "w") as f:
        json.dump(feature_names, f, indent=2)

    with open(os.path.join(models_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # Class weight ratio calculation
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = float(neg_count / max(1, pos_count))
    print(f"[*] Class Imbalance: {pos_count} SSI cases vs {neg_count} Non-SSI in train set")
    print(f"    scale_pos_weight = {scale_pos_weight:.2f}")

    # Hyperparameter Tuning
    tuned_params = {}
    if do_tuning:
        tuned_params = tune_models(X_train, y_train, scale_pos_weight, n_iter=10)
        with open(os.path.join(models_dir, "tuning_summary.json"), "w") as f:
            serializable_params = {
                k: {p: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v)
                    for p, v in params.items()}
                for k, params in tuned_params.items()
            }
            json.dump(serializable_params, f, indent=2)

    models = {}

    # 1. Baseline: Multivariable Logistic Regression
    print("[*] Training Baseline Model: Multivariable Logistic Regression...")
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    models["Logistic_Regression"] = lr

    # 2. Random Forest Classifier
    rf_params = tuned_params.get("Random_Forest", {
        "n_estimators": 250, "max_depth": 8, "min_samples_split": 5
    })
    print(f"[*] Training Random Forest Classifier (tuned: {bool(tuned_params.get('Random_Forest'))})...")
    rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1, **rf_params)
    rf.fit(X_train, y_train)
    models["Random_Forest"] = rf

    # 3. LightGBM Classifier
    if HAS_LGB:
        lgb_params = tuned_params.get("LightGBM", {
            "n_estimators": 180, "learning_rate": 0.04, "max_depth": 5, "scale_pos_weight": scale_pos_weight
        })
        print(f"[*] Training LightGBM Classifier (tuned: {bool(tuned_params.get('LightGBM'))})...")
        lgbm = lgb.LGBMClassifier(random_state=42, verbosity=-1, **lgb_params)
        lgbm.fit(X_train, y_train)
        models["LightGBM"] = lgbm

    # 4. XGBoost Classifier
    if HAS_XGB:
        xgb_params = tuned_params.get("XGBoost", {
            "n_estimators": 180, "learning_rate": 0.04, "max_depth": 4, "scale_pos_weight": scale_pos_weight
        })
        print(f"[*] Training XGBoost Classifier (tuned: {bool(tuned_params.get('XGBoost'))})...")
        xg_clf = xgb.XGBClassifier(random_state=42, eval_metric='logloss', **xgb_params)
        xg_clf.fit(X_train, y_train)
        models["XGBoost"] = xg_clf

    # 5. CatBoost Classifier (if available)
    if HAS_CATBOOST:
        print("[*] Training CatBoost Classifier...")
        cb_clf = cb.CatBoostClassifier(
            iterations=200,
            learning_rate=0.04,
            depth=5,
            scale_pos_weight=scale_pos_weight,
            random_seed=42,
            verbose=0
        )
        cb_clf.fit(X_train, y_train)
        models["CatBoost"] = cb_clf

    # Evaluate predictions on Temporal Test Set
    results = {}
    print("\n" + "="*65)
    print(f"{'Model Name':<25} | {'AUROC':<8} | {'AUPRC':<8} | {'Brier':<8}")
    print("="*65)

    best_model_name = None
    best_auprc = -1

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        auroc = roc_auc_score(y_test, y_prob)
        auprc = average_precision_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)

        results[name] = {
            "AUROC": round(auroc, 4),
            "AUPRC": round(auprc, 4),
            "Brier_Score": round(brier, 4)
        }

        print(f"{name:<25} | {auroc:<8.4f} | {auprc:<8.4f} | {brier:<8.4f}")

        # Save individual candidate model
        with open(os.path.join(models_dir, f"{name}.pkl"), "wb") as f:
            pickle.dump(model, f)

        if auprc > best_auprc:
            best_auprc = auprc
            best_model_name = name

    print("="*65)
    print(f"[OK] Champion Model selected by AUPRC: {best_model_name} (AUPRC={best_auprc:.4f})")

    # -------------------------------------------------------------
    # Probability Calibration Step (CalibratedClassifierCV)
    # -------------------------------------------------------------
    print("\n" + "-" * 65)
    print(f"🎯 [Probability Calibration] Calibrating champion model ({best_model_name})")
    print("   Method: Isotonic Regression with 5-fold internal CV")
    print("-" * 65)

    champion_model = models[best_model_name]
    calibrator = CalibratedClassifierCV(estimator=champion_model, method='isotonic', cv=5)
    calibrator.fit(X_train, y_train)

    calib_prob = calibrator.predict_proba(X_test)[:, 1]
    calib_auroc = roc_auc_score(y_test, calib_prob)
    calib_auprc = average_precision_score(y_test, calib_prob)
    calib_brier = brier_score_loss(y_test, calib_prob)

    print(f"[*] Raw Champion Model   -> Brier Score: {results[best_model_name]['Brier_Score']:.4f}")
    print(f"[*] Calibrated Champion   -> Brier Score: {calib_brier:.4f} (Lower = Better Probability Match)")
    print(f"[*] Calibrated Discrim.   -> AUROC: {calib_auroc:.4f} | AUPRC: {calib_auprc:.4f}")

    results["Calibrated_Champion"] = {
        "Base_Model": best_model_name,
        "AUROC": round(calib_auroc, 4),
        "AUPRC": round(calib_auprc, 4),
        "Brier_Score": round(calib_brier, 4)
    }

    # Save calibrated champion artifact
    calib_path = os.path.join(models_dir, "Calibrated_Champion_Model.pkl")
    with open(calib_path, "wb") as f:
        pickle.dump(calibrator, f)
    print(f"[OK] Calibrated model artifact saved to: {calib_path}")

    # Save champion metadata
    with open(os.path.join(models_dir, "champion_info.json"), "w") as f:
        json.dump({
            "champion_name": best_model_name,
            "has_calibration": True,
            "calibrated_model_file": "Calibrated_Champion_Model.pkl",
            "base_model_file": f"{best_model_name}.pkl"
        }, f, indent=2)

    # Save summary metrics
    with open(os.path.join(models_dir, "metrics_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    return models, results, X_test, y_test, best_model_name, calibrator

if __name__ == "__main__":
    train_and_compare_models()
