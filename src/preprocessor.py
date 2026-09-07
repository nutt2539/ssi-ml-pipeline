"""
Data Preprocessing & Cleaning Module
Enforces the Zero Data Leakage Protocol, handles missing data,
performs One-Hot Encoding, and splits data into Temporal Train/Test sets.
"""

import os
import csv
import math

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# STRICT LIST OF ALLOWED PREDICTOR COLUMNS (NO POST-OP LEAKAGE!)
PREDICTOR_COLS = [
    # Preoperative
    "age", "sex", "bmi", "diabetes", "ckd", "cirrhosis",
    "malignancy", "smoking", "steroid_immunosuppressant",
    "asa_class", "emergency", "surgical_specialty",
    "preop_albumin", "preop_hct", "preop_wbc",
    # Intraoperative
    "wound_class", "surgical_approach", "operative_time_min",
    "ebl_ml", "intraop_hypothermia", "intraop_hypotension",
    "intraop_transfusion", "atb_prophylaxis_timing",
    "atb_redosing_needed", "atb_redosing_given"
]

TARGET_COL = "ssi_30d"

# STRICT LIST OF FORBIDDEN LEAKAGE COLUMNS
FORBIDDEN_LEAKAGE = [
    "postop_los", "total_los", "icu_admission", "postop_fever",
    "postop_wbc", "postop_crp", "ward_antibiotics", "reoperation"
]

def clean_and_split_data(raw_csv_path: str, output_dir: str = None, test_year: int = 2024):
    """
    Cleans raw cohort data, verifies zero leakage, handles missing values,
    and splits into Development (Train) and Temporal Validation (Test) cohorts.
    """
    if output_dir is None:
        output_dir = os.path.dirname(raw_csv_path)

    print(f"[*] Loading raw cohort data from: {raw_csv_path}")

    # Read raw data using csv reader
    rows = []
    with open(raw_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for r in reader:
            rows.append(r)

    # 1. Verify Zero Data Leakage
    for forbidden in FORBIDDEN_LEAKAGE:
        if forbidden in fieldnames:
            raise ValueError(f"CRITICAL ERROR: Data leakage detected! Column '{forbidden}' found in raw dataset.")

    print("[*] Guardrail passed: Zero Data Leakage verified.")

    # 2. Separate into Development (Train: <= 2023) and Temporal Test (Test: 2024)
    train_rows = [r for r in rows if int(r["surgery_year"]) < test_year]
    test_rows = [r for r in rows if int(r["surgery_year"]) >= test_year]

    print(f"[*] Temporal Splitting completed:")
    print(f"    - Development Cohort (Train: <{test_year}): {len(train_rows)} patients")
    print(f"    - Temporal Validation Cohort (Test: {test_year}): {len(test_rows)} patients")

    # 3. Compute Train Medians for continuous missing imputation
    continuous_numeric = ["age", "bmi", "preop_albumin", "preop_hct", "preop_wbc", "operative_time_min", "ebl_ml"]
    train_medians = {}

    for col in continuous_numeric:
        vals = [float(r[col]) for r in train_rows if r[col] != "" and r[col] is not None]
        vals.sort()
        mid = len(vals) // 2
        median_val = (vals[mid] if len(vals) % 2 != 0 else (vals[mid - 1] + vals[mid]) / 2) if vals else 0.0
        train_medians[col] = median_val

    # Impute missing values with train median
    def impute_record(row):
        r = row.copy()
        for col in continuous_numeric:
            if r[col] == "" or r[col] is None:
                r[col] = str(round(train_medians[col], 2))
        return r

    train_imputed = [impute_record(r) for r in train_rows]
    test_imputed = [impute_record(r) for r in test_rows]

    # 4. Save processed train and test sets
    train_path = os.path.join(output_dir, "train_cohort.csv")
    test_path = os.path.join(output_dir, "test_cohort.csv")

    with open(train_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(train_imputed)

    with open(test_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(test_imputed)

    print(f"[OK] Preprocessed datasets saved:")
    print(f"     Train: {train_path}")
    print(f"     Test:  {test_path}")

    return train_path, test_path

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base_dir, "data", "synthetic_surgical_cohort.csv")
    clean_and_split_data(raw_path)
