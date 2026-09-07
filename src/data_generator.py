"""
Synthetic Surgical Cohort Generator
Designed for Somdech Phra Pinklao Hospital (Naval Medical Department)
Generates realistic preoperative and intraoperative EMR data for SSI prediction.
"""

import os
import csv
import math
import random
from datetime import datetime, timedelta

def generate_synthetic_cohort(
    num_patients: int = 4500,
    start_year: int = 2020,
    end_year: int = 2024,
    random_seed: int = 42,
    output_path: str = None
):
    random.seed(random_seed)
    
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    days_range = (end_date - start_date).days

    specialties = ["Colorectal", "Hepatobiliary", "Upper_GI", "Breast_Endocrine", "Vascular", "General"]
    specialty_weights = [0.28, 0.22, 0.18, 0.12, 0.08, 0.12]

    records = []

    for i in range(1, num_patients + 1):
        # Patient ID
        pid = f"SPK-{i:05d}"
        
        # Surgery Date
        random_days = random.randint(0, days_range)
        surgery_date = start_date + timedelta(days=random_days)
        surgery_year = surgery_date.year

        # Demographics
        age = int(random.gauss(58.5, 13.5))
        age = max(18, min(92, age))
        sex = "Male" if random.random() < 0.52 else "Female"

        bmi = round(random.gauss(24.5, 4.2), 1)
        bmi = max(15.0, min(44.0, bmi))

        # Comorbidities
        diabetes = 1 if (random.random() < (0.24 if age > 50 else 0.12)) else 0
        ckd = 1 if (random.random() < (0.16 if age > 60 else 0.07)) else 0
        cirrhosis = 1 if (random.random() < 0.04) else 0
        malignancy = 1 if (random.random() < (0.32 if age > 55 else 0.15)) else 0
        smoking = 1 if (sex == "Male" and random.random() < 0.28) or (sex == "Female" and random.random() < 0.06) else 0
        steroid = 1 if (random.random() < 0.06) else 0

        # ASA Class (I, II, III, IV)
        if age < 40 and not diabetes and not ckd:
            asa = random.choices([1, 2, 3], weights=[0.60, 0.35, 0.05])[0]
        elif age > 65 or ckd or (diabetes and malignancy):
            asa = random.choices([2, 3, 4], weights=[0.25, 0.60, 0.15])[0]
        else:
            asa = random.choices([1, 2, 3, 4], weights=[0.15, 0.60, 0.22, 0.03])[0]

        # Emergency surgery
        emergency = 1 if (random.random() < 0.22) else 0

        # Specialty
        specialty = random.choices(specialties, weights=specialty_weights)[0]

        # Wound Class
        if specialty == "Breast_Endocrine":
            wound_class = random.choices([1, 2], weights=[0.90, 0.10])[0]
        elif specialty == "Colorectal":
            wound_class = random.choices([2, 3, 4], weights=[0.75, 0.18, 0.07])[0]
        elif specialty == "Upper_GI":
            wound_class = random.choices([1, 2, 3, 4], weights=[0.15, 0.65, 0.15, 0.05])[0]
        else:
            wound_class = random.choices([1, 2, 3, 4], weights=[0.40, 0.45, 0.10, 0.05])[0]

        # Approach (Open vs Laparoscopic)
        approach = "Laparoscopic" if (random.random() < (0.50 if specialty in ["Colorectal", "Upper_GI"] else 0.25)) else "Open"

        # Preop Labs with realistic missingness
        # Albumin (g/dL)
        if random.random() < 0.08: # 8% missing
            preop_albumin = None
            albumin_val = 3.6 # default for risk calculation
        else:
            alb = round(random.gauss(3.7 - (0.5 if malignancy else 0) - (0.3 if age > 70 else 0), 0.55), 2)
            preop_albumin = max(1.8, min(5.2, alb))
            albumin_val = preop_albumin

        # Hematocrit (%)
        if random.random() < 0.04: # 4% missing
            preop_hct = None
            hct_val = 38.0
        else:
            hct = round(random.gauss(39.0 - (3.5 if malignancy or ckd else 0), 4.5), 1)
            preop_hct = max(20.0, min(52.0, hct))
            hct_val = preop_hct

        # WBC count (x10^3/uL)
        if random.random() < 0.05:
            preop_wbc = None
        else:
            wbc = round(random.gauss(7.8 + (3.2 if emergency else 0), 2.8), 2)
            preop_wbc = max(2.5, min(28.0, wbc))

        # Operative Time (minutes)
        base_time = 140 if approach == "Open" else 165
        if specialty == "Colorectal": base_time += 45
        if specialty == "Hepatobiliary": base_time += 60
        if specialty == "Breast_Endocrine": base_time -= 50
        op_time = int(random.gauss(base_time, 50))
        operative_time_min = max(35, min(480, op_time))

        # Estimated Blood Loss (EBL, mL)
        base_ebl = 250 if approach == "Open" else 120
        if specialty == "Hepatobiliary": base_ebl += 180
        ebl = int(random.gauss(base_ebl, 160))
        ebl_ml = max(20, min(2500, ebl))

        # Intraoperative Vitals & Physiology
        hypothermia = 1 if (random.random() < (0.35 if operative_time_min > 180 else 0.15)) else 0
        hypotension = 1 if (random.random() < (0.25 if ebl_ml > 500 or emergency else 0.10)) else 0
        transfusion = 1 if (ebl_ml > 600 or (preop_hct and preop_hct < 28)) else 0

        # Antibiotic Prophylaxis Timing
        atb_timing = random.choices(
            ["Compliant_Within_60min", "Late_After_Incision", "None_Given"],
            weights=[0.84, 0.12, 0.04]
        )[0]

        # Antibiotic Redosing
        redosing_indicated = 1 if (operative_time_min >= 180 or ebl_ml >= 1500) else 0
        if redosing_indicated:
            redosing_given = 1 if (random.random() < 0.68) else 0
        else:
            redosing_given = 0

        # CALCULATE TRUE RISK OF SSI (Logistic latent model based on clinical literature)
        # Base log-odds for ~5.5% SSI: -4.4
        logit = -4.4

        # Wound class effect
        if wound_class == 1: logit -= 0.6
        elif wound_class == 2: logit += 0.3
        elif wound_class == 3: logit += 1.2
        elif wound_class == 4: logit += 1.9

        # Operative time effect (>180 min adds risk)
        logit += ((operative_time_min - 120) / 60.0) * 0.45

        # EBL effect (>500 mL adds risk)
        if ebl_ml > 500: logit += 0.55
        if ebl_ml > 1000: logit += 0.40

        # Patient factors
        if diabetes: logit += 0.50
        if bmi >= 30.0: logit += 0.45
        if albumin_val < 3.0: logit += 0.65
        if asa >= 3: logit += 0.40
        if emergency: logit += 0.35
        if smoking: logit += 0.30

        # Intraoperative factors
        if hypothermia: logit += 0.40
        if hypotension: logit += 0.30
        if approach == "Laparoscopic": logit -= 0.45

        # Prophylaxis factors
        if atb_timing == "Late_After_Incision": logit += 0.60
        elif atb_timing == "None_Given": logit += 1.10

        if redosing_indicated and not redosing_given:
            logit += 0.55

        # Probability
        prob_ssi = 1.0 / (1.0 + math.exp(-logit))

        # Primary Outcome
        ssi_30d = 1 if (random.random() < prob_ssi) else 0

        # SSI sub-classification
        if ssi_30d == 1:
            if wound_class in [3, 4] or specialty in ["Colorectal", "Hepatobiliary"]:
                ssi_type = random.choices(["Superficial", "Deep_Incisional", "Organ_Space"], weights=[0.40, 0.35, 0.25])[0]
            else:
                ssi_type = random.choices(["Superficial", "Deep_Incisional", "Organ_Space"], weights=[0.65, 0.25, 0.10])[0]
        else:
            ssi_type = "None"

        record = {
            "patient_id": pid,
            "surgery_date": surgery_date.strftime("%Y-%m-%d"),
            "surgery_year": surgery_year,
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "diabetes": diabetes,
            "ckd": ckd,
            "cirrhosis": cirrhosis,
            "malignancy": malignancy,
            "smoking": smoking,
            "steroid_immunosuppressant": steroid,
            "asa_class": asa,
            "emergency": emergency,
            "surgical_specialty": specialty,
            "wound_class": wound_class,
            "surgical_approach": approach,
            "preop_albumin": preop_albumin if preop_albumin is not None else "",
            "preop_hct": preop_hct if preop_hct is not None else "",
            "preop_wbc": preop_wbc if preop_wbc is not None else "",
            "operative_time_min": operative_time_min,
            "ebl_ml": ebl_ml,
            "intraop_hypothermia": hypothermia,
            "intraop_hypotension": hypotension,
            "intraop_transfusion": transfusion,
            "atb_prophylaxis_timing": atb_timing,
            "atb_redosing_needed": redosing_indicated,
            "atb_redosing_given": redosing_given,
            "ssi_30d": ssi_30d,
            "ssi_type": ssi_type
        }
        records.append(record)

    # Save to CSV
    if output_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_path = os.path.join(base_dir, "data", "synthetic_surgical_cohort.csv")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    total_ssi = sum(r["ssi_30d"] for r in records)
    ssi_rate = (total_ssi / num_patients) * 100

    print(f"[OK] Synthetic cohort successfully generated: {num_patients} patients")
    print(f"     Saved to: {output_path}")
    print(f"     Total SSI cases: {total_ssi} ({ssi_rate:.2f}%)")
    print(f"     Years covered: {start_year} - {end_year}")

    return output_path

if __name__ == "__main__":
    generate_synthetic_cohort()
