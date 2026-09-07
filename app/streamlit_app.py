"""
Bedside Clinical Decision Support Application
Designed for OR Staff & Surgeons at Somdech Phra Pinklao Hospital
Computes Real-Time SSI Risk at End of Surgery with Actionable SHAP Explanations.
"""

import os
import sys
import json
import pickle
import streamlit as st
import streamlit.components.v1 as components

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

try:
    from app.report_generator import generate_clinical_summary_jpg
except ImportError:
    from report_generator import generate_clinical_summary_jpg

# Page Configuration
st.set_page_config(
    page_title="SSI Risk Predictor | Somdech Phra Pinklao Hospital",
    page_icon="🩺",
    layout="wide"
)

def render_snapshot_button(button_label="📸 Export ทั้งหน้าจอ (JPG)", key="snap_btn"):
    """Embeds an action button that uses html2canvas in client-side JS to capture the full page to JPG."""
    html_code = f"""
    <style>
      html, body {{
        margin: 0;
        padding: 0;
        overflow: hidden;
        background: transparent;
      }}
    </style>
    <div style="display: flex; align-items: center; justify-content: center; width: 100%; margin: 0; padding: 0;">
      <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
      <button id="btn_{key}" onclick="capturePage_{key}()" style="
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        color: #ffffff;
        border: 1px solid #38bdf8;
        border-radius: 8px;
        padding: 6px 12px;
        font-size: 11.5px;
        font-weight: 700;
        cursor: pointer;
        width: 100%;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35);
        transition: all 0.2s ease-in-out;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Prompt', sans-serif;
        line-height: 1.4;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      ">
        <span>{button_label}</span>
      </button>
      <script>
        function capturePage_{key}() {{
          const btn = document.getElementById('btn_{key}');
          const origText = btn.innerHTML;
          btn.innerHTML = '<span>⏳ กำลังถ่ายภาพ...</span>';
          btn.disabled = true;

          try {{
            const pDoc = window.parent.document;
            const target = pDoc.querySelector('[data-testid="stAppViewContainer"]') || pDoc.querySelector('.stApp') || pDoc.body;

            window.html2canvas(target, {{
              useCORS: true,
              allowTaint: true,
              scale: 1.6,
              backgroundColor: '#020617',
              logging: false,
              windowWidth: target.scrollWidth,
              windowHeight: target.scrollHeight
            }}).then(canvas => {{
              const now = new Date();
              const pad = (n) => String(n).padStart(2, '0');
              const ts = now.getFullYear() + pad(now.getMonth()+1) + pad(now.getDate()) + '_' + pad(now.getHours()) + pad(now.getMinutes());
              const link = document.createElement('a');
              link.download = 'SSI_Risk_Screen_' + ts + '.jpg';
              link.href = canvas.toDataURL('image/jpeg', 0.95);
              link.click();

              btn.innerHTML = '<span>✅ บันทึก JPG สำเร็จ!</span>';
              setTimeout(() => {{
                btn.innerHTML = origText;
                btn.disabled = false;
              }}, 2800);
            }}).catch(err => {{
              console.error('html2canvas error:', err);
              btn.innerHTML = '<span>❌ ถ่ายภาพไม่สำเร็จ</span>';
              setTimeout(() => {{
                btn.innerHTML = origText;
                btn.disabled = false;
              }}, 2800);
            }});
          }} catch (e) {{
            console.error('Snapshot exception:', e);
            btn.innerHTML = '<span>❌ เกิดข้อผิดพลาด</span>';
            setTimeout(() => {{
              btn.innerHTML = origText;
              btn.disabled = false;
            }}, 2800);
          }}
        }}
      </script>
    </div>
    """
    components.html(html_code, height=40)

# Custom Styling with Google Fonts and Clinical Dark Theme
st.html("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

    *, *::before, *::after {
        box-sizing: border-box;
        letter-spacing: normal !important;
    }

    html, body, .stApp {
        font-family: 'Prompt', 'Sarabun', 'Noto Sans Thai', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        line-height: 1.65 !important;
        background: radial-gradient(circle at 50% 0%, #0c1527 0%, #020617 100%) !important;
        color: #f8fafc !important;
    }

    /* Clean spacing for main container preventing header overlap */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1400px !important;
    }

    /* Target text content explicitly without breaking nested flex/grid div layouts */
    p, li, td, th, [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] span,
    [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label,
    .clinical-card, .clinical-card p {
        line-height: 1.65 !important;
        letter-spacing: normal !important;
        word-break: normal !important;
        overflow-wrap: break-word !important;
    }

    h1, h2, h3, h4, h5, h6 {
        line-height: 1.35 !important;
        letter-spacing: normal !important;
        margin-bottom: 0.5rem !important;
    }

    /* Header Bar */
    [data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Sidebar Background & Details */
    [data-testid="stSidebar"] {
        background: #090e1a !important;
        border-right: 1px solid #1e293b !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #38bdf8 !important;
        font-weight: 700 !important;
        line-height: 1.4 !important;
    }
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
        margin-bottom: 4px !important;
    }
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #cbd5e1 !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        line-height: 1.5 !important;
        margin-bottom: 2px !important;
    }

    /* Sidebar Expanders */
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background: rgba(15, 23, 42, 0.7) !important;
        border: 1px solid #1e293b !important;
        border-radius: 0.75rem !important;
        margin-bottom: 0.75rem !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25) !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        font-size: 0.84rem !important;
        line-height: 1.5 !important;
        padding: 0.5rem 0.75rem !important;
    }

    /* Form Inputs */
    .stSelectbox > div > div, .stNumberInput input, .stTextInput input {
        background-color: #0f172a !important;
        border: 1px solid #334155 !important;
        color: #f8fafc !important;
        border-radius: 0.6rem !important;
        font-size: 0.85rem !important;
        line-height: 1.4 !important;
    }
    .stSelectbox svg {
        fill: #94a3b8 !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        border-bottom: 1px solid #1e293b;
        padding-bottom: 4px;
        overflow-x: auto !important;
        flex-wrap: nowrap !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
        font-weight: 600;
        font-size: 0.85rem;
        color: #94a3b8;
        background-color: transparent;
        border: none;
        white-space: nowrap !important;
        line-height: 1.5 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom: 2px solid #0ea5e9 !important;
        background: rgba(14, 165, 233, 0.08) !important;
    }

    /* Custom Card */
    .clinical-card {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid #1e293b;
        border-radius: 1rem;
        padding: 1.25rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        margin-bottom: 1rem;
    }

    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
        color: white !important;
        border: 1px solid #38bdf8 !important;
        border-radius: 0.75rem !important;
        font-weight: 600 !important;
        line-height: 1.4 !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35) !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(2, 132, 199, 0.5) !important;
    }
</style>
""")

# App Header (Branded & Polished)
st.html("""
<div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #1e293b; padding-bottom: 1.2rem; margin-bottom: 1.5rem; flex-wrap: wrap; gap: 14px;">
  <div style="display: flex; align-items: center; gap: 14px;">
    <div style="width: 48px; height: 48px; border-radius: 14px; background: linear-gradient(135deg, #0ea5e9, #4f46e5); display: flex; align-items: center; justify-content: center; font-size: 26px; box-shadow: 0 8px 20px rgba(14, 165, 233, 0.3); flex-shrink: 0;">
      🩺
    </div>
    <div>
      <h1 style="margin: 0; font-size: 1.5rem; font-weight: 800; color: #ffffff; line-height: 1.4; display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
        End-of-Surgery SSI Risk Calculator
        <span style="font-size: 0.7rem; font-weight: 700; padding: 3px 10px; border-radius: 9999px; background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); line-height: 1.2;">
          Isotonic Calibrated
        </span>
      </h1>
      <p style="margin: 6px 0 0 0; font-size: 0.82rem; color: #94a3b8; line-height: 1.6;">
        โรงพยาบาลสมเด็จพระปิ่นเกล้า กรมแพทย์ทหารเรือ &bull; ภาควิชาศัลยศาสตร์ &bull; Explainable AI Decision Support
      </p>
    </div>
  </div>
  <div style="display: flex; align-items: center; gap: 10px;">
    <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid #334155; padding: 6px 14px; border-radius: 10px; font-size: 0.75rem; color: #cbd5e1; line-height: 1.4;">
      Active: <b style="color: #38bdf8;">Calibrated Logistic Regression</b> (Brier: 0.0623)
    </div>
  </div>
</div>
""")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

@st.cache_resource
def load_model_artifacts():
    model = None
    model_name = "Model"
    is_calibrated = False

    # Check for calibrated model first
    calib_path = os.path.join(MODELS_DIR, "Calibrated_Champion_Model.pkl")
    champ_info_path = os.path.join(MODELS_DIR, "champion_info.json")

    base_name = "Champion"
    if os.path.exists(champ_info_path):
        try:
            with open(champ_info_path, "r") as f:
                cinfo = json.load(f)
                base_name = cinfo.get("champion_name", "Champion")
        except Exception:
            pass

    if os.path.exists(calib_path):
        try:
            with open(calib_path, "rb") as f:
                model = pickle.load(f)
                model_name = f"Calibrated {base_name} (Isotonic)"
                is_calibrated = True
        except Exception:
            model = None

    if model is None:
        for name in ["LightGBM", "XGBoost", "Random_Forest", "Logistic_Regression"]:
            p = os.path.join(MODELS_DIR, f"{name}.pkl")
            if os.path.exists(p):
                with open(p, "rb") as f:
                    model = pickle.load(f)
                    model_name = name
                    break

    scaler = None
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    explainer = None
    explainer_path = os.path.join(MODELS_DIR, "shap_explainer.pkl")
    if os.path.exists(explainer_path):
        try:
            with open(explainer_path, "rb") as f:
                explainer = pickle.load(f)
        except Exception:
            explainer = None

    feature_names = []
    fn_path = os.path.join(MODELS_DIR, "feature_names.json")
    if os.path.exists(fn_path):
        try:
            with open(fn_path, "r") as f:
                feature_names = json.load(f)
        except Exception:
            pass

    return model, model_name, is_calibrated, scaler, explainer, feature_names

model, model_name, is_calibrated, scaler, explainer, feature_names = load_model_artifacts()

if model is None:
    st.warning("⚠️ Model weights not found in `models/`. Please run `python run_pipeline.py` first to train the models.")
    st.stop()

# Sidebar: Patient & OR Inputs
st.sidebar.header("📋 Patient & Intra-op Factors")

with st.sidebar.expander("1. Preoperative Demographics & Status", expanded=True):
    age = st.slider("Age (years)", 18, 95, 62)
    sex = st.selectbox("Sex", ["Male", "Female"])
    bmi = st.number_input("Body Mass Index (kg/m²)", 15.0, 45.0, 26.5, step=0.1)
    asa = st.selectbox("ASA Physical Status", [1, 2, 3, 4], index=2, format_func=lambda x: f"Class {x}")
    emergency = st.checkbox("Emergency Surgery (Acuity)", value=False)

with st.sidebar.expander("2. Baseline Comorbidities", expanded=False):
    dm = st.checkbox("Diabetes Mellitus (DM)", value=True)
    ckd = st.checkbox("Chronic Kidney Disease (CKD)", value=False)
    cirrhosis = st.checkbox("Liver Cirrhosis", value=False)
    malignancy = st.checkbox("Active Malignancy", value=True)
    smoking = st.checkbox("Active Tobacco Smoking", value=False)
    steroid = st.checkbox("Steroid / Immunosuppressant", value=False)

with st.sidebar.expander("3. Preoperative Laboratory Tests", expanded=False):
    albumin = st.number_input("Serum Albumin (g/dL)", 1.5, 5.5, 3.1, step=0.1)
    hct = st.number_input("Hematocrit (%)", 18.0, 55.0, 36.0, step=0.5)
    wbc = st.number_input("WBC count (x10³/μL)", 2.0, 30.0, 8.4, step=0.5)

with st.sidebar.expander("4. Intraoperative Surgical Factors", expanded=True):
    specialty = st.selectbox(
        "Surgical Specialty",
        ["Colorectal", "Hepatobiliary", "Upper_GI", "Breast_Endocrine", "Vascular", "General"],
        index=0
    )
    wound_class = st.selectbox(
        "Surgical Wound Class",
        [1, 2, 3, 4],
        index=1,
        format_func=lambda x: {
            1: "Class I: Clean",
            2: "Class II: Clean-Contaminated",
            3: "Class III: Contaminated",
            4: "Class IV: Dirty / Infected"
        }[x]
    )
    approach = st.radio("Surgical Approach", ["Open", "Laparoscopic"], horizontal=True)
    op_time = st.slider("Operative Duration (minutes)", 30, 480, 210, step=10)
    ebl = st.slider("Estimated Blood Loss (EBL, mL)", 20, 2500, 550, step=50)

with st.sidebar.expander("5. Intraoperative Physiology & Prophylaxis", expanded=False):
    hypothermia = st.checkbox("Intraoperative Hypothermia (<36.0°C)", value=True)
    hypotension = st.checkbox("Intraoperative Hypotension (MAP <65)", value=False)
    transfusion = st.checkbox("Blood Transfusion in OR", value=False)
    atb_timing = st.selectbox("Antibiotic Timing", ["Compliant_Within_60min", "Late_After_Incision", "None_Given"], index=0)
    atb_redose_needed = 1 if (op_time >= 180 or ebl >= 1500) else 0
    atb_redose_given = st.checkbox("Antibiotic Redosing Administered", value=True) if atb_redose_needed else 0

# Build feature vector
patient_info = {
    "age": age, "sex": sex, "bmi": bmi, "asa": asa, "emergency": emergency,
    "specialty": specialty, "approach": approach, "wound_class": wound_class,
    "op_time": op_time, "ebl": ebl, "dm": dm, "ckd": ckd, "malignancy": malignancy,
    "albumin": albumin, "atb_timing": atb_timing, "hypothermia": hypothermia
}

def prepare_input():
    import pandas as pd
    row = {
        "age": age, "bmi": bmi, "sex_male": 1 if sex == "Male" else 0,
        "diabetes": 1 if dm else 0, "ckd": 1 if ckd else 0, "cirrhosis": 1 if cirrhosis else 0,
        "malignancy": 1 if malignancy else 0, "smoking": 1 if smoking else 0,
        "steroid_immunosuppressant": 1 if steroid else 0, "asa_class": asa,
        "emergency": 1 if emergency else 0, "preop_albumin": albumin,
        "preop_hct": hct, "preop_wbc": wbc, "wound_class": wound_class,
        "approach_open": 1 if approach == "Open" else 0, "operative_time_min": op_time,
        "ebl_ml": ebl, "intraop_hypothermia": 1 if hypothermia else 0,
        "intraop_hypotension": 1 if hypotension else 0,
        "intraop_transfusion": 1 if transfusion else 0,
        "atb_redosing_needed": atb_redose_needed,
        "atb_redosing_given": 1 if atb_redose_given else 0
    }
    specialties = ["Colorectal", "Hepatobiliary", "Upper_GI", "Breast_Endocrine", "Vascular", "General"]
    for s in specialties:
        row[f'specialty_{s}'] = 1 if specialty == s else 0

    timings = ["Compliant_Within_60min", "Late_After_Incision", "None_Given"]
    for t in timings:
        row[f'atb_{t}'] = 1 if atb_timing == t else 0

    df = pd.DataFrame([row])
    if scaler:
        cont_cols = ["age", "bmi", "preop_albumin", "preop_hct", "preop_wbc", "operative_time_min", "ebl_ml"]
        df[cont_cols] = scaler.transform(df[cont_cols])
    return df

# Main Display & Prediction
input_df = prepare_input()
risk_prob = float(model.predict_proba(input_df)[0, 1]) * 100.0

# Risk Classification Tiers & Styling
if risk_prob >= 15.0:
    border_color = "rgba(244, 63, 94, 0.45)"
    bg_gradient = "linear-gradient(135deg, rgba(136, 19, 55, 0.45) 0%, rgba(15, 23, 42, 0.95) 100%)"
    text_color = "#fb7185"
    badge_bg = "rgba(244, 63, 94, 0.2)"
    badge_text = "#fda4af"
    badge_border = "rgba(244, 63, 94, 0.4)"
    badge_label = "🔴 HIGH RISK TIER"
    tier_desc = "ความเสี่ยงสูงกว่าค่าเฉลี่ยศัลยกรรมทั่วไปอย่างมีนัยสำคัญ (>2.5 เท่า) แนะนำใช้มาตรการป้องกันเชิงรุก (Proactive Bundle) ทันที"
elif risk_prob >= 7.0:
    border_color = "rgba(245, 158, 11, 0.45)"
    bg_gradient = "linear-gradient(135deg, rgba(120, 53, 15, 0.45) 0%, rgba(15, 23, 42, 0.95) 100%)"
    text_color = "#fbbf24"
    badge_bg = "rgba(245, 158, 11, 0.2)"
    badge_text = "#fde68a"
    badge_border = "rgba(245, 158, 11, 0.4)"
    badge_label = "🟡 MODERATE RISK TIER"
    tier_desc = "ความเสี่ยงระดับปานกลาง ต้องเฝ้าระวังการอักเสบเฉพาะที่ของแผลและควบคุมอุณหภูมิ/น้ำตาลอย่างเคร่งครัด"
else:
    border_color = "rgba(16, 185, 129, 0.45)"
    bg_gradient = "linear-gradient(135deg, rgba(6, 78, 59, 0.45) 0%, rgba(15, 23, 42, 0.95) 100%)"
    text_color = "#34d399"
    badge_bg = "rgba(16, 185, 129, 0.2)"
    badge_text = "#a7f3d0"
    badge_border = "rgba(16, 185, 129, 0.4)"
    badge_label = "🟢 LOW RISK TIER"
    tier_desc = "ความเสี่ยงต่ำ ให้การดูแลตามแนวทางเวชปฏิบัติและมาตรการป้องกันมาตรฐานทั่วไป"

# Compute patient-level TreeSHAP feature contributions
top_feats = []
feat_contribs = []
if explainer is not None:
    try:
        shap_values = explainer(input_df)
        sv = shap_values.values
        if len(sv.shape) == 3:
            sv_patient = sv[0, :, 1]
        elif len(sv.shape) == 2:
            sv_patient = sv[0, :]
        else:
            sv_patient = sv

        name_mapping = {
            "age": "อายุ (Age)",
            "bmi": "ดัชนีมวลกาย (BMI)",
            "sex_male": "เพศชาย (Male)",
            "diabetes": "โรคเบาหวาน (DM)",
            "ckd": "ไตเรื้อรัง (CKD)",
            "cirrhosis": "ตับแข็ง (Cirrhosis)",
            "malignancy": "มะเร็ง (Malignancy)",
            "smoking": "สูบบุหรี่ (Smoking)",
            "steroid_immunosuppressant": "ยากดภูมิ/สเตียรอยด์",
            "asa_class": "ASA Physical Status",
            "emergency": "ผ่าตัดฉุกเฉิน (Emergency)",
            "preop_albumin": "Albumin ก่อนผ่าตัด",
            "preop_hct": "ความเข้มข้นเลือด (Hct)",
            "preop_wbc": "เม็ดเลือดขาว (WBC)",
            "wound_class": "ระดับความสะอาดแผล (Wound Class)",
            "approach_open": "ผ่าตัดเปิดหน้าท้อง (Open)",
            "operative_time_min": "ระยะเวลาผ่าตัด (Duration)",
            "ebl_ml": "ปริมาณเลือดที่เสีย (EBL)",
            "intraop_hypothermia": "อุณหภูมิกายต่ำใน OR (<36°C)",
            "intraop_hypotension": "ความดันตกใน OR",
            "intraop_transfusion": "ให้เลือดใน OR",
            "atb_redosing_needed": "จำเป็นต้อง Redose ยาฆ่าเชื้อ",
            "atb_redosing_given": "ได้รับ Redose ยาฆ่าเชื้อ",
            "specialty_Colorectal": "แผนก Colorectal",
            "atb_Compliant_Within_60min": "ให้ยาปฏิชีวนะตรงเวลา (<60 นาที)",
            "atb_Late_After_Incision": "ให้ยาปฏิชีวนะช้า (หลังลงมีด)",
            "atb_None_Given": "ไม่ได้รับยาปฏิชีวนะป้องกัน"
        }
        feat_cols = input_df.columns.tolist()
        for i, col in enumerate(feat_cols):
            val = sv_patient[i]
            readable_name = name_mapping.get(col, col)
            feat_contribs.append((readable_name, val))
        feat_contribs.sort(key=lambda x: abs(x[1]), reverse=True)
        top_feats = feat_contribs[:10]
    except Exception:
        top_feats = []

# Generate High-Resolution Clinical Summary Card (JPG)
summary_jpg_bytes = generate_clinical_summary_jpg(
    patient_info=patient_info,
    risk_prob=risk_prob,
    badge_label=badge_label,
    model_name=model_name,
    top_shap=top_feats
)

# Sidebar Action & Quick Export
with st.sidebar:
    st.markdown("---")
    predict_clicked = st.button("⚡ Calculate SSI Risk at OR Discharge", type="primary", width="stretch")
    st.html("<div style='margin-top: 14px; margin-bottom: 6px; font-size: 0.8rem; font-weight: 700; color: #38bdf8;'>📤 ส่งออกรายงาน (Export JPG)</div>")
    render_snapshot_button(button_label="📸 Export ทั้งหน้าจอ (JPG)", key="side_snap")
    st.download_button(
        label="📑 บัตรสรุปประเมิน EMR (JPG)",
        data=summary_jpg_bytes,
        file_name=f"SSI_Clinical_Summary_{specialty}_{risk_prob:.1f}pct.jpg",
        mime="image/jpeg",
        width="stretch"
    )

# Quick Action & Export Toolbar
toolbar_c1, toolbar_c2, toolbar_c3 = st.columns([5, 3.5, 3.5])
with toolbar_c1:
    st.html(f"""
    <div style="display: flex; align-items: center; gap: 8px; font-size: 0.82rem; color: #94a3b8; padding: 6px 0; line-height: 1.5;">
      <span style="display: inline-block; width: 8px; height: 8px; border-radius: 9999px; background: #34d399; flex-shrink: 0;"></span>
      <span>สถานะ: พร้อมประเมินผล &bull; โมเดล: <b style="color: #38bdf8;">{model_name}</b></span>
    </div>
    """)
with toolbar_c2:
    render_snapshot_button(button_label="📸 ถ่ายภาพทั้งหน้า (JPG)", key="top_snap")
with toolbar_c3:
    st.download_button(
        label="📑 บัตรสรุป EMR (JPG)",
        data=summary_jpg_bytes,
        file_name=f"SSI_Clinical_Card_{specialty}_{risk_prob:.1f}pct.jpg",
        mime="image/jpeg",
        width="stretch"
    )

# HERO RISK BANNER
st.html(f"""
<div style="background: {bg_gradient}; border: 1px solid {border_color}; border-radius: 1.25rem; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 15px 35px -5px rgba(0,0,0,0.4);">
  <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1.25rem;">
    <div>
      <div style="font-size: 0.78rem; font-weight: 700; color: {text_color}; margin-bottom: 6px; line-height: 1.4;">
        ⚠️ ระดับความเสี่ยงทำนาย (Predicted SSI Probability)
      </div>
      <div style="display: flex; align-items: center; gap: 16px; flex-wrap: wrap; margin-bottom: 10px;">
        <span style="font-size: 3.25rem; font-weight: 900; color: {text_color}; line-height: 1.1; display: inline-block;">
          {risk_prob:.1f}%
        </span>
        <span style="display: inline-flex; align-items: center; padding: 6px 16px; border-radius: 9999px; font-size: 0.8rem; font-weight: 800; background: {badge_bg}; color: {badge_text}; border: 1px solid {badge_border}; line-height: 1.3;">
          {badge_label}
        </span>
      </div>
      <p style="margin: 0; font-size: 0.85rem; color: #cbd5e1; max-width: 580px; line-height: 1.65;">
        {tier_desc}
      </p>
    </div>

    <div style="background: rgba(2, 6, 23, 0.6); border: 1px solid #1e293b; border-radius: 0.85rem; padding: 1rem 1.25rem; font-size: 0.78rem; color: #94a3b8; min-width: 220px; display: flex; flex-direction: column; gap: 8px;">
      <div style="display: flex; justify-content: space-between; line-height: 1.5;">
        <span>Model Brier Score:</span>
        <b style="color: #34d399;">0.0623</b>
      </div>
      <div style="display: flex; justify-content: space-between; line-height: 1.5;">
        <span>Sensitivity / Specificity:</span>
        <b style="color: #ffffff;">77.6% / 67.6%</b>
      </div>
      <div style="display: flex; justify-content: space-between; line-height: 1.5;">
        <span>Optimal Cutoff:</span>
        <b style="color: #fde047;">6.1% Threshold</b>
      </div>
      <div style="display: flex; justify-content: space-between; border-top: 1px solid #1e293b; padding-top: 6px; line-height: 1.5;">
        <span>Temporal Validation:</span>
        <span style="color: #38bdf8;">Hold-out 2024</span>
      </div>
    </div>
  </div>
</div>
""")

# Main Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 แผนดูแลแผล (Care Plan)",
    "🔍 คำอธิบาย AI (SHAP)",
    "📋 ส่งมอบห้องพักฟื้น (PACU)",
    "📊 สถิติและการทดสอบ (Validation)"
])

# -------------------------------------------------------------
# TAB 1: Actionable Clinical Care Plan
# -------------------------------------------------------------
with tab1:
    st.html("""
    <div style="margin-bottom: 1rem;">
      <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #ffffff;">💡 Actionable Surgical Care Plan (Evidence-Based Interventions)</h3>
      <p style="margin: 3px 0 0 0; font-size: 0.8rem; color: #94a3b8;">อิงตามแนวทางเวชปฏิบัติสากล CDC, WHO และ Surgical Infection Society (SIS)</p>
    </div>
    """)

    col_left, col_right = st.columns(2)

    with col_left:
        # Card 1: Wound Care
        if risk_prob >= 15.0:
            wound_badge = """
            <div style="padding: 8px 12px; background: rgba(244, 63, 94, 0.15); border: 1px solid rgba(244, 63, 94, 0.3); border-radius: 8px; color: #fda4af; font-weight: 700; font-size: 0.8rem; margin-bottom: 10px; line-height: 1.5;">
              ⚡ แนะนำพิจารณา Closed-Incision NPWT (ciNPWT)
            </div>
            """
            wound_body = """
            <p style="margin: 8px 0; line-height: 1.65;">• <b>Closed-Incision Negative Pressure Wound Therapy (ciNPWT):</b> พิจารณาใช้เครื่องดูดสุญญากาศแรงดันลบ (-125 mmHg) ปิดแผลเป็นเวลา 5-7 วัน เพื่อลดการเกิด Hematoma และ Seroma ในแผลเสี่ยงสูง</p>
            <p style="margin: 8px 0; line-height: 1.65;">• <b>Sterile Waterproof Dressing:</b> ปิดแผลปลอดเชื้อแบบกันน้ำ คงไว้ห้ามเปิดทำแผลอย่างน้อย 48 ชั่วโมงแรก</p>
            """
        elif risk_prob >= 7.0:
            wound_badge = """
            <div style="padding: 8px 12px; background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 8px; color: #fde68a; font-weight: 700; font-size: 0.8rem; margin-bottom: 10px; line-height: 1.5;">
              • Advanced Vapour-Permeable Dressing
            </div>
            """
            wound_body = """
            <p style="margin: 8px 0; line-height: 1.65;">• <b>Advanced Dressing:</b> ใช้วัสดุปิดแผลแบบ Hydrocolloid หรือ Film ที่ระบายอากาศได้ดี หลีกเลี่ยงแรงดึงรั้งผิวหนังบริเวณขอบแผล</p>
            <p style="margin: 8px 0; line-height: 1.65;">• ปิดแผลปลอดเชื้อคงไว้ 48 ชั่วโมง</p>
            """
        else:
            wound_badge = """
            <div style="padding: 8px 12px; background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; color: #a7f3d0; font-weight: 700; font-size: 0.8rem; margin-bottom: 10px; line-height: 1.5;">
              • Standard Sterile Dressing
            </div>
            """
            wound_body = """
            <p style="margin: 8px 0; line-height: 1.65;">• ปิดแผลด้วยผ้าก๊อซปลอดเชื้อมาตรฐาน 48 ชั่วโมง ดูแลความสะอาดตามขั้นตอนปกติของ รพ.สมเด็จพระปิ่นเกล้า</p>
            """

        st.html(f"""
        <div class="clinical-card">
          <div style="font-size: 0.95rem; font-weight: 700; color: #38bdf8; display: flex; align-items: center; gap: 8px; margin-bottom: 10px; line-height: 1.4;">
            <span>🩺</span> 1. การดูแลแผลและวัสดุปิดแผล (Wound Dressing)
          </div>
          {wound_badge}
          <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.65;">
            {wound_body}
          </div>
        </div>
        """)

        # Card 2: Antimicrobial
        atb_alert_html = ""
        if atb_timing != "Compliant_Within_60min":
            atb_alert_html += """
            <div style="padding: 8px 12px; background: rgba(244, 63, 94, 0.15); border: 1px solid rgba(244, 63, 94, 0.3); border-radius: 8px; color: #fda4af; font-weight: 700; font-size: 0.8rem; margin-bottom: 10px; line-height: 1.5;">
              ⚠️ แจ้งเตือน: ให้ยาปฏิชีวนะไม่ตรงเวลา (&lt;60 นาทีก่อนกรีดผิวหนัง)
            </div>
            """
        if atb_redose_needed and not atb_redose_given:
            atb_alert_html += """
            <div style="padding: 8px 12px; background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 8px; color: #fde68a; font-weight: 700; font-size: 0.8rem; margin-bottom: 10px; line-height: 1.5;">
              ⚠️ Redosing Alert: การผ่าตัดนาน >3 ชม. หรือเสียเลือด >1,500 mL แต่ยังไม่ได้รับ Redosing ใน OR
            </div>
            """

        if risk_prob >= 15.0 or wound_class >= 3:
            atb_body = "<p style='margin: 8px 0; line-height: 1.65;'>• ในเคสแผล Class III/IV หรือปนเปื้อน พิจารณาให้ยาปฏิชีวนะครอบคลุมต่อไม่เกิน 24 ชม. ตามข้อบ่งชี้ทางศัลยกรรม</p>"
        else:
            atb_body = "<p style='margin: 8px 0; line-height: 1.65;'>• <b>หยุดยาปฏิชีวนะป้องกัน</b> ภายในห้องผ่าตัดหรือไม่เกิน 24 ชม. หลังผ่าตัดตามเกณฑ์ Antimicrobial Stewardship เพื่อป้องกันเชื้อดื้อยา</p>"

        st.html(f"""
        <div class="clinical-card">
          <div style="font-size: 0.95rem; font-weight: 700; color: #818cf8; display: flex; align-items: center; gap: 8px; margin-bottom: 10px; line-height: 1.4;">
            <span>💊</span> 2. การบริหารยาปฏิชีวนะป้องกัน (Antimicrobial)
          </div>
          {atb_alert_html}
          <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.65;">
            {atb_body}
          </div>
        </div>
        """)

    with col_right:
        # Card 3: Physiologic Optimization
        physio_warming = ""
        if hypothermia:
            physio_warming = """
            <div style="padding: 8px 12px; background: rgba(244, 63, 94, 0.15); border: 1px solid rgba(244, 63, 94, 0.3); border-radius: 8px; color: #fda4af; font-weight: 700; font-size: 0.8rem; margin-bottom: 10px; line-height: 1.5;">
              ❄️ Active Forced-Air Warming ใน PACU ด่วน
            </div>
            <p style="margin: 8px 0; line-height: 1.65;">• ผู้ป่วยมีภาวะตัวเย็นใน OR (&lt;36.0°C) ต้องใช้ผ้าห่มลมร้อนรักษาอุณหภูมิร่างกายให้ <b>&gt;36.5°C</b> ทันที เพื่อป้องกันภาวะแผลขาดเลือด</p>
            """
        else:
            physio_warming = "<p style='margin: 8px 0; line-height: 1.65;'>• อุณหภูมิกายใน OR ปกติ รักษาระดับผ้าห่มอุ่นตามเกณฑ์มาตรฐาน</p>"

        physio_dm = ""
        if dm:
            physio_dm = "<p style='margin: 8px 0; color: #fde047; line-height: 1.65;'>• <b>Strict Glycemic Control:</b> ควบคุมระดับน้ำตาลในเลือดหลังผ่าตัดให้ <b>&lt;180 mg/dL</b> ตลอด 48 ชม. แรก (ตรวจ Blood Sugar ทุก 4-6 ชม.)</p>"

        physio_alb = ""
        if albumin < 3.0:
            physio_alb = f"<p style='margin: 8px 0; color: #fde047; line-height: 1.65;'>• <b>Nutritional Support:</b> Albumin ต่ำ ({albumin} g/dL) พิจารณาเริ่ม Early Oral Feeding หรือ High-Protein Diet เร็วที่สุด</p>"

        st.html(f"""
        <div class="clinical-card">
          <div style="font-size: 0.95rem; font-weight: 700; color: #fbbf24; display: flex; align-items: center; gap: 8px; margin-bottom: 10px; line-height: 1.4;">
            <span>🌡️</span> 3. การฟื้นฟูสรีรวิทยาใน PACU / Ward
          </div>
          <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.65;">
            {physio_warming}
            {physio_dm}
            {physio_alb}
          </div>
        </div>
        """)

        # Card 4: Surveillance
        if risk_prob >= 15.0:
            surveil_body = """
            <div style="padding: 8px 12px; background: rgba(244, 63, 94, 0.15); border: 1px solid rgba(244, 63, 94, 0.3); border-radius: 8px; color: #fda4af; font-weight: 700; font-size: 0.8rem; margin-bottom: 10px; line-height: 1.5;">
              • นัดตรวจประเมินแผลเร็วขึ้น (Early Review)
            </div>
            <p style="margin: 8px 0; line-height: 1.65;">• นัดตรวจแผลที่ OPD ศัลยกรรมภายใน <b>48–72 ชั่วโมง</b> หลังออกจากโรงพยาบาล</p>
            <p style="margin: 8px 0; line-height: 1.65;">• ให้เอกสารคำแนะนำคนไข้สังเกตอาการบวม แดง ร้อน มีหนอง หรือมีไข้ให้กลับมาพบแพทย์ทันที</p>
            """
        else:
            surveil_body = """
            <p style="margin: 8px 0; line-height: 1.65;">• นัดตรวจติดตามแผลและตัดไหมตามนัดมาตรฐาน 7–10 วัน</p>
            <p style="margin: 8px 0; line-height: 1.65;">• ให้คำแนะนำการดูแลสุขอนามัยแผลตามปกติ</p>
            """

        st.html(f"""
        <div class="clinical-card">
          <div style="font-size: 0.95rem; font-weight: 700; color: #34d399; display: flex; align-items: center; gap: 8px; margin-bottom: 10px; line-height: 1.4;">
            <span>📅</span> 4. แผนติดตามแผลหลังจำหน่าย (Follow-up)
          </div>
          <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.65;">
            {surveil_body}
          </div>
        </div>
        """)

# -------------------------------------------------------------
# TAB 2: Explainable AI (SHAP Waterfall)
# -------------------------------------------------------------
with tab2:
    st.html("""
    <div style="margin-bottom: 1rem;">
      <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #ffffff;">🔍 Patient-Level Explainable AI (TreeSHAP Analysis)</h3>
      <p style="margin: 3px 0 0 0; font-size: 0.8rem; color: #94a3b8;">แจกแจงน้ำหนักของตัวแปรว่าปัจจัยใดในผู้ป่วยรายนี้ที่ดันความเสี่ยงเพิ่มขึ้น (+) หรือช่วยลดความเสี่ยง (-)</p>
    </div>
    """)

    if top_feats:
        max_abs_val = max([abs(x[1]) for x in top_feats] + [0.1])

        # Modern Glassmorphic SHAP Bars in pure HTML/CSS
        bars_html = []
        for name, val in top_feats:
            is_risk = val > 0
            pct = min(100, int((abs(val) / max_abs_val) * 100))
            color = "#f43f5e" if is_risk else "#10b981"
            bg_glow = "rgba(244, 63, 94, 0.15)" if is_risk else "rgba(16, 185, 129, 0.15)"
            sign = "+" if is_risk else ""
            direction_label = "ดันเสี่ยงเพิ่ม" if is_risk else "ช่วยลดความเสี่ยง"

            bars_html.append(f"""
            <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid #1e293b; border-radius: 10px; padding: 10px 14px; margin-bottom: 8px;">
              <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; font-size: 0.82rem;">
                <span style="font-weight: 600; color: #f1f5f9;">{name}</span>
                <div style="display: flex; align-items: center; gap: 8px;">
                  <span style="font-size: 0.7rem; color: #94a3b8; background: {bg_glow}; padding: 2px 8px; border-radius: 6px; border: 1px solid {color}40;">{direction_label}</span>
                  <span style="font-weight: 800; font-family: monospace; color: {color};">{sign}{val:.3f}</span>
                </div>
              </div>
              <div style="width: 100%; height: 8px; background: #0b1120; border-radius: 9999px; overflow: hidden;">
                <div style="width: {pct}%; height: 100%; background: {color}; border-radius: 9999px; transition: width 0.4s ease;"></div>
              </div>
            </div>
            """)

        shap_container_html = f"""
        <div style="background: rgba(2, 6, 23, 0.5); border: 1px solid #1e293b; border-radius: 14px; padding: 16px; margin-bottom: 14px;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <span style="font-size: 0.88rem; font-weight: 700; color: #ffffff;">Top 10 Feature Contributions (Log-Odds Impact)</span>
            <span style="font-size: 0.72rem; color: #94a3b8;">🔴 เสี่ยงสูงขึ้น | 🟢 ช่วยลดเสี่ยง</span>
          </div>
          {''.join(bars_html)}
        </div>
        """
        st.html(shap_container_html)

        # Clinical Interpretation Bullet Cards
        risk_increasing = [f for f in feat_contribs if f[1] > 0.05][:4]
        risk_reducing = [f for f in feat_contribs if f[1] < -0.05][:3]

        inc_items = "".join([f"<li style='margin-bottom: 4px;'>• <b>{name}</b> (+{val:.3f})</li>" for name, val in risk_increasing]) or "<li>• ไม่พบปัจจัยเสี่ยงผิดปกติเด่นชัด</li>"
        dec_items = "".join([f"<li style='margin-bottom: 4px;'>• <b>{name}</b> ({val:.3f})</li>" for name, val in risk_reducing]) or "<li>• ไม่พบปัจจัยปกป้องเด่นชัด</li>"

        st.html(f"""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; margin-top: 12px;">
          <div style="background: rgba(244, 63, 94, 0.1); border: 1px solid rgba(244, 63, 94, 0.25); border-radius: 12px; padding: 14px; font-size: 0.8rem;">
            <span style="font-weight: 700; color: #fb7185; display: block; margin-bottom: 6px;">🔴 ปัจจัยหลักที่ดันความเสี่ยงสูงขึ้น:</span>
            <ul style="color: #cbd5e1; padding-left: 8px; margin: 0; list-style: none;">
              {inc_items}
            </ul>
          </div>
          <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 12px; padding: 14px; font-size: 0.8rem;">
            <span style="font-weight: 700; color: #34d399; display: block; margin-bottom: 6px;">🟢 ปัจจัยที่ช่วยลดความเสี่ยงหรืออยู่ในเกณฑ์ปกติ:</span>
            <ul style="color: #cbd5e1; padding-left: 8px; margin: 0; list-style: none;">
              {dec_items}
            </ul>
          </div>
        </div>
        """)

    else:
        st.info("ระบบกำลังแสดงผลแบบ Global SHAP แทน")
        shap_img_path = os.path.join(OUTPUTS_DIR, "figures", "Figure3_SHAP_Global_Beeswarm.png")
        if os.path.exists(shap_img_path):
            st.image(shap_img_path, caption="Figure 3: Global TreeSHAP Feature Importance (Temporal Validation Cohort)", width="stretch")

# -------------------------------------------------------------
# TAB 3: PACU Handover Report
# -------------------------------------------------------------
with tab3:
    st.html("""
    <div style="margin-bottom: 1rem;">
      <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #ffffff;">📋 ใบประเมินความเสี่ยงและส่งมอบห้องพักฟื้น (PACU Handover Report)</h3>
      <p style="margin: 3px 0 0 0; font-size: 0.8rem; color: #94a3b8;">สามารถคัดลอกข้อความด้านล่างไปวางในระบบ EMR หรือส่งออกเป็นรูปภาพ JPG ทั้งหน้าจอ / บัตรประเมินความเสี่ยง</p>
    </div>
    """)

    handover_text = f"""========================================================================
🏥 โรงพยาบาลสมเด็จพระปิ่นเกล้า กรมแพทย์ทหารเรือ | แผนกศัลยกรรม
ใบประเมินความเสี่ยงการติดเชื้อแผลผ่าตัด ณ จุดสิ้นสุดการผ่าตัด (End of Surgery)
========================================================================
[1. ข้อมูลผู้ป่วยและหัตถการ]
- อายุ: {age} ปี | เพศ: {sex} | ดัชนีมวลกาย (BMI): {bmi:.1f} kg/m²
- แผนกศัลยกรรม: {specialty} | วิธีการผ่าตัด: {approach}
- ระดับความสะอาดแผล (Wound Class): Class {wound_class} ({'Clean' if wound_class==1 else 'Clean-Contam' if wound_class==2 else 'Contaminated' if wound_class==3 else 'Dirty'})
- ความเร่งด่วน: {'ฉุกเฉิน (Emergency)' if emergency else 'นัดล่วงหน้า (Elective)'} | ASA Class: {asa}
- ระยะเวลาผ่าตัด: {op_time} นาที | เลือดที่เสีย (EBL): {ebl} mL

[2. ปัจจัยเสี่ยงและการให้ยาปฏิชีวนะ]
- โรคประจำตัว: DM: {'ใช่' if dm else 'ไม่ใช่'}, CKD: {'ใช่' if ckd else 'ไม่ใช่'}, Malignancy: {'ใช่' if malignancy else 'ไม่ใช่'}
- ผลเลือดก่อนผ่าตัด: Albumin {albumin} g/dL | Hct {hct}% | WBC {wbc} x10³/μL
- อุณหภูมิกายใน OR: {'Hypothermia (<36°C)' if hypothermia else 'ปกติ (>36°C)'}
- ยาปฏิชีวนะป้องกัน: {atb_timing} | Redosing: {'ได้รับ' if atb_redose_given else 'ไม่ได้รับ'}

------------------------------------------------------------------------
[3. ผลการพยากรณ์ด้วยปัญญาประดิษฐ์ (Explainable AI Score)]
- ระดับความน่าจะเป็นของการเกิด SSI ภายใน 30 วัน: {risk_prob:.1f}%
- กลุ่มความเสี่ยง (Risk Tier): {badge_label}
- โมเดลที่ใช้ประมวลผล: {model_name}
------------------------------------------------------------------------

[4. แผนการดูแลเฉพาะบุคคลที่แนะนำในห้องพักฟื้นและหอผู้ป่วย (Care Plan)]
- การดูแลแผล: {'พิจารณาใช้ Closed-Incision NPWT (ciNPWT) 5-7 วัน' if risk_prob >= 15.0 else 'ปิดแผลปลอดเชื้อคงไว้ 48 ชั่วโมง'}
- อุณหภูมิกาย: {'ใช้ Active Warming Blanket ใน PACU เพื่อคงอุณหภูมิ >36.5°C' if hypothermia else 'รักษาอุณหภูมิตามมาตรฐาน'}
- การควบคุมน้ำตาล: {'ตรวจ Blood Sugar ทุก 4-6 ชม. คุมระดับ <180 mg/dL' if dm else 'ตรวจตามคำสั่งแพทย์เจ้าของไข้'}
- การนัดตรวจแผล: {'นัดประเมินแผลเร็วขึ้นที่ 48-72 ชม. หลังจำหน่าย' if risk_prob >= 15.0 else 'นัดตัดไหม/ตรวจแผล 7-10 วัน'}

ผู้ประเมิน: ทีมศัลยแพทย์และพยาบาลห้องผ่าตัด รพ.สมเด็จพระปิ่นเกล้า
========================================================================"""

    # Export actions row
    col_dl1, col_dl2, col_dl3 = st.columns([4, 4, 4])
    with col_dl1:
        st.download_button(
            label="📥 ดาวน์โหลดข้อความสรุป (.txt)",
            data=handover_text,
            file_name=f"PACU_SSI_Risk_Assessment_{specialty}_{wound_class}.txt",
            mime="text/plain",
            width="stretch"
        )
    with col_dl2:
        st.download_button(
            label="📑 ดาวน์โหลดบัตรสรุป EMR (JPG)",
            data=summary_jpg_bytes,
            file_name=f"PACU_SSI_Clinical_Card_{specialty}_{risk_prob:.1f}pct.jpg",
            mime="image/jpeg",
            width="stretch"
        )
    with col_dl3:
        render_snapshot_button(button_label="📸 Export ทั้งหน้าจอ (JPG)", key="tab3_snap")

    st.markdown("---")

    # High-Res JPG Card Preview
    with st.expander("🖼️ ดูตัวอย่างบัตรสรุปผลทางคลินิก (Clinical Summary JPG Card Preview)", expanded=True):
        st.image(
            summary_jpg_bytes,
            caption="บัตรสรุปประเมินความเสี่ยงและแผนการดูแลแผล (ความละเอียด 1200x860 px เหมาะสำหรับแนบเวชระเบียน EMR)",
            width="stretch"
        )

    st.html("<h4 style='font-size: 0.9rem; color: #cbd5e1; margin-top: 1rem;'>📝 ข้อความสำหรับคัดลอกลงระบบ EMR / HIS:</h4>")
    st.text_area("Handover Report Text (Ready for EMR Copy)", value=handover_text, height=260)

# -------------------------------------------------------------
# TAB 4: Model Validation & Subgroup Analysis
# -------------------------------------------------------------
with tab4:
    st.html("""
    <div style="margin-bottom: 1rem;">
      <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #ffffff;">📊 ความเที่ยงตรงของโมเดลและการวิเคราะห์กลุ่มย่อย (Validation & Subgroups)</h3>
      <p style="margin: 3px 0 0 0; font-size: 0.8rem; color: #94a3b8;">ผลการทดสอบบนชุดข้อมูลผู้ป่วยจริงตามมิติเวลา (Temporal Validation Cohort ปี 2024)</p>
    </div>
    """)

    # Render Styled Table 3 Subgroup Analysis
    t3_html = """
    <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid #1e293b; border-radius: 1rem; padding: 1rem; margin-bottom: 1.5rem; overflow-x: auto;">
      <div style="font-size: 0.9rem; font-weight: 700; color: #38bdf8; margin-bottom: 10px;">
        Table 3: Subgroup and Sensitivity Analysis (Temporal Validation Cohort)
      </div>
      <table style="width: 100%; text-align: left; font-size: 0.78rem; border-collapse: collapse; color: #cbd5e1;">
        <thead>
          <tr style="border-bottom: 1px solid #334155; color: #94a3b8;">
            <th style="padding: 8px 6px;">หมวดหมู่</th>
            <th style="padding: 8px 6px;">กลุ่มย่อย</th>
            <th style="padding: 8px 6px; text-align: center;">N Total</th>
            <th style="padding: 8px 6px; text-align: center;">SSI Rate</th>
            <th style="padding: 8px 6px; text-align: center; color: #38bdf8; font-weight: bold;">AUROC</th>
            <th style="padding: 8px 6px; text-align: center; color: #34d399; font-weight: bold;">AUPRC</th>
            <th style="padding: 8px 6px; text-align: center;">Brier</th>
            <th style="padding: 8px 6px; text-align: center;">Sens.</th>
            <th style="padding: 8px 6px; text-align: center;">Spec.</th>
          </tr>
        </thead>
        <tbody>
          <tr style="background: rgba(30, 41, 59, 0.5); font-weight: bold; border-bottom: 1px solid #1e293b;">
            <td style="padding: 8px 6px;">Overall Cohort</td>
            <td style="padding: 8px 6px;">All Test Patients</td>
            <td style="padding: 8px 6px; text-align: center;">904</td>
            <td style="padding: 8px 6px; text-align: center;">67 (7.4%)</td>
            <td style="padding: 8px 6px; text-align: center; color: #38bdf8;">0.803</td>
            <td style="padding: 8px 6px; text-align: center; color: #34d399;">0.259</td>
            <td style="padding: 8px 6px; text-align: center;">0.0623</td>
            <td style="padding: 8px 6px; text-align: center;">77.6%</td>
            <td style="padding: 8px 6px; text-align: center;">67.6%</td>
          </tr>
          <tr style="border-bottom: 1px solid #1e293b;">
            <td style="padding: 8px 6px;">Specialty</td>
            <td style="padding: 8px 6px;">Colorectal Surgery</td>
            <td style="padding: 8px 6px; text-align: center;">247</td>
            <td style="padding: 8px 6px; text-align: center; color: #fb7185;">28 (11.3%)</td>
            <td style="padding: 8px 6px; text-align: center;">0.693</td>
            <td style="padding: 8px 6px; text-align: center;">0.275</td>
            <td style="padding: 8px 6px; text-align: center;">0.0946</td>
            <td style="padding: 8px 6px; text-align: center;">53.6%</td>
            <td style="padding: 8px 6px; text-align: center;">82.2%</td>
          </tr>
          <tr style="border-bottom: 1px solid #1e293b;">
            <td style="padding: 8px 6px;">Specialty</td>
            <td style="padding: 8px 6px;">Non-Colorectal</td>
            <td style="padding: 8px 6px; text-align: center;">657</td>
            <td style="padding: 8px 6px; text-align: center;">39 (5.9%)</td>
            <td style="padding: 8px 6px; text-align: center; color: #38bdf8; font-weight: 600;">0.846</td>
            <td style="padding: 8px 6px; text-align: center;">0.256</td>
            <td style="padding: 8px 6px; text-align: center;">0.0501</td>
            <td style="padding: 8px 6px; text-align: center;">82.1%</td>
            <td style="padding: 8px 6px; text-align: center;">74.4%</td>
          </tr>
          <tr style="border-bottom: 1px solid #1e293b;">
            <td style="padding: 8px 6px;">Approach</td>
            <td style="padding: 8px 6px;">Open Laparotomy</td>
            <td style="padding: 8px 6px; text-align: center;">590</td>
            <td style="padding: 8px 6px; text-align: center;">39 (6.6%)</td>
            <td style="padding: 8px 6px; text-align: center; color: #38bdf8;">0.817</td>
            <td style="padding: 8px 6px; text-align: center;">0.258</td>
            <td style="padding: 8px 6px; text-align: center;">0.0560</td>
            <td style="padding: 8px 6px; text-align: center;">82.1%</td>
            <td style="padding: 8px 6px; text-align: center;">68.2%</td>
          </tr>
          <tr style="border-bottom: 1px solid #1e293b;">
            <td style="padding: 8px 6px;">Approach</td>
            <td style="padding: 8px 6px;">Laparoscopic / MIS</td>
            <td style="padding: 8px 6px; text-align: center;">314</td>
            <td style="padding: 8px 6px; text-align: center;">28 (8.9%)</td>
            <td style="padding: 8px 6px; text-align: center;">0.782</td>
            <td style="padding: 8px 6px; text-align: center; color: #34d399;">0.280</td>
            <td style="padding: 8px 6px; text-align: center;">0.0740</td>
            <td style="padding: 8px 6px; text-align: center;">92.9%</td>
            <td style="padding: 8px 6px; text-align: center;">50.0%</td>
          </tr>
          <tr style="border-bottom: 1px solid #1e293b;">
            <td style="padding: 8px 6px;">Acuity</td>
            <td style="padding: 8px 6px;">Emergency Surgery</td>
            <td style="padding: 8px 6px; text-align: center;">216</td>
            <td style="padding: 8px 6px; text-align: center;">17 (7.9%)</td>
            <td style="padding: 8px 6px; text-align: center; color: #38bdf8; font-weight: bold;">0.850</td>
            <td style="padding: 8px 6px; text-align: center; color: #34d399; font-weight: bold;">0.413</td>
            <td style="padding: 8px 6px; text-align: center;">0.0591</td>
            <td style="padding: 8px 6px; text-align: center;">64.7%</td>
            <td style="padding: 8px 6px; text-align: center;">91.0%</td>
          </tr>
          <tr style="border-bottom: 1px solid #1e293b;">
            <td style="padding: 8px 6px;">Wound Class</td>
            <td style="padding: 8px 6px;">Clean / Clean-Contam (I-II)</td>
            <td style="padding: 8px 6px; text-align: center;">749</td>
            <td style="padding: 8px 6px; text-align: center;">40 (5.3%)</td>
            <td style="padding: 8px 6px; text-align: center;">0.799</td>
            <td style="padding: 8px 6px; text-align: center;">0.205</td>
            <td style="padding: 8px 6px; text-align: center;">0.0472</td>
            <td style="padding: 8px 6px; text-align: center;">95.0%</td>
            <td style="padding: 8px 6px; text-align: center;">52.3%</td>
          </tr>
          <tr>
            <td style="padding: 8px 6px;">Wound Class</td>
            <td style="padding: 8px 6px;">Contaminated / Dirty (III-IV)</td>
            <td style="padding: 8px 6px; text-align: center;">155</td>
            <td style="padding: 8px 6px; text-align: center; color: #fb7185; font-weight: bold;">27 (17.4%)</td>
            <td style="padding: 8px 6px; text-align: center;">0.694</td>
            <td style="padding: 8px 6px; text-align: center; color: #34d399; font-weight: bold;">0.356</td>
            <td style="padding: 8px 6px; text-align: center;">0.1350</td>
            <td style="padding: 8px 6px; text-align: center;">74.1%</td>
            <td style="padding: 8px 6px; text-align: center;">60.9%</td>
          </tr>
        </tbody>
      </table>
    </div>
    """
    st.html(t3_html)

    # Figure 1 & 2
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        f1_path = os.path.join(OUTPUTS_DIR, "figures", "Figure1_Discrimination_Curves.png")
        if os.path.exists(f1_path):
            st.image(f1_path, caption="Figure 1: Discrimination Curves (AUROC & AUPRC)", width="stretch")
    with fcol2:
        f2_path = os.path.join(OUTPUTS_DIR, "figures", "Figure2_Calibration_and_DCA.png")
        if os.path.exists(f2_path):
            st.image(f2_path, caption="Figure 2: Calibration & Decision Curve Analysis", width="stretch")

st.markdown("---")
st.caption("Medical Artificial Intelligence Research &bull; Somdech Phra Pinklao Hospital &bull; Royal Thai Navy")


