"""
Medical Report & Image Export Generator for Somdech Phra Pinklao Hospital
Generates high-resolution clinical summary cards in JPG format.
"""

import os
import io
import datetime
from PIL import Image, ImageDraw, ImageFont

def get_best_font(size=14, bold=False):
    """Attempt to load a high-quality font that supports Thai and English."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    bundled_bold = os.path.join(base_dir, 'fonts', 'Sarabun-Bold.ttf')
    bundled_regular = os.path.join(base_dir, 'fonts', 'Sarabun-Regular.ttf')

    primary = bundled_bold if bold else bundled_regular
    secondary = bundled_regular if bold else bundled_bold

    font_candidates = [
        primary,
        secondary,
        '/System/Library/Fonts/Supplemental/Thonburi.ttc',
        '/System/Library/Fonts/ThonburiUI.ttc',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/freefont/FreeSans.ttf',
        '/Library/Fonts/Arial Unicode.ttf',
        '/System/Library/Fonts/Supplemental/Arial.ttf',
        '/System/Library/Fonts/Helvetica.ttc'
    ]
    for p in font_candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                continue
    return ImageFont.load_default()

def generate_clinical_summary_jpg(
    patient_info,
    risk_prob,
    badge_label,
    model_name="Calibrated Logistic Regression",
    top_shap=None,
    care_plans=None
):
    """
    Renders an A4-proportioned, 1200x860px medical-grade infographic summary card in JPG format.
    """
    W, H = 1200, 860
    img = Image.new('RGB', (W, H), color='#020617')
    draw = ImageDraw.Draw(img)

    f_title = get_best_font(23, bold=True)
    f_sub = get_best_font(13)
    f_head = get_best_font(16, bold=True)
    f_body = get_best_font(13)
    f_bold = get_best_font(13, bold=True)
    f_huge = get_best_font(44, bold=True)
    f_badge = get_best_font(15, bold=True)

    # Clean badge label from emoji chars
    clean_badge = badge_label.replace("🔴", "").replace("🟡", "").replace("🟢", "").strip()

    # 1. Top Hospital Header Card
    draw.rectangle([25, 20, W-25, 105], fill='#0f172a', outline='#1e293b', width=1)
    draw.text((45, 32), 'โรงพยาบาลสมเด็จพระปิ่นเกล้า กรมแพทย์ทหารเรือ | SOMDECH PHRA PINKLAO HOSPITAL', fill='#ffffff', font=f_title)
    draw.text((45, 72), 'Surgical Site Infection (SSI) Risk Assessment Report | End-of-Surgery Bedside Decision Support', fill='#94a3b8', font=f_sub)
    now_str = datetime.datetime.now().strftime('%d/%m/%Y %H:%M น.')
    draw.text((W-240, 72), f'เวลาประเมิน: {now_str}', fill='#38bdf8', font=f_sub)

    # 2. Left Panel: Patient & Surgical Factors
    draw.rectangle([25, 120, 600, 475], fill='#090e1a', outline='#1e293b', width=1)
    draw.text((45, 135), 'ข้อมูลผู้ป่วยและปัจจัยการผ่าตัด (Patient & OR Factors)', fill='#38bdf8', font=f_head)
    draw.line([45, 162, 580, 162], fill='#1e293b', width=1)

    age = patient_info.get("age", "-")
    sex = patient_info.get("sex", "-")
    bmi = patient_info.get("bmi", 0.0)
    asa = patient_info.get("asa", "-")
    emergency = patient_info.get("emergency", False)
    specialty = patient_info.get("specialty", "-")
    approach = patient_info.get("approach", "-")
    wound_class = patient_info.get("wound_class", "-")
    op_time = patient_info.get("op_time", 0)
    ebl = patient_info.get("ebl", 0)
    dm = patient_info.get("dm", False)
    ckd = patient_info.get("ckd", False)
    malignancy = patient_info.get("malignancy", False)
    albumin = patient_info.get("albumin", "-")
    atb_timing = patient_info.get("atb_timing", "-")
    hypothermia = patient_info.get("hypothermia", False)

    wc_map = {1: "Clean (I)", 2: "Clean-Contaminated (II)", 3: "Contaminated (III)", 4: "Dirty (IV)"}
    wc_label = wc_map.get(wound_class, f"Class {wound_class}")

    details = [
        f'• ผู้ป่วย: อายุ {age} ปี | เพศ: {sex} | ดัชนีมวลกาย (BMI): {bmi:.1f} kg/m²',
        f'• ระดับความเร่งด่วน: {"ผ่าตัดฉุกเฉิน (Emergency)" if emergency else "นัดล่วงหน้า (Elective)"} | ASA: Class {asa}',
        f'• แผนกศัลยศาสตร์: {specialty} | วิธีผ่าตัด: {approach}',
        f'• ระดับความสะอาดแผล: {wc_label}',
        f'• ระยะเวลาผ่าตัด: {op_time} นาที | ปริมาณเลือดที่เสีย (EBL): {ebl} mL',
        f'• โรคประจำตัว: เบาหวาน: {"มี" if dm else "ไม่มี"} | ไตเรื้อรัง: {"มี" if ckd else "ไม่มี"} | มะเร็ง: {"มี" if malignancy else "ไม่มี"}',
        f'• ค่าทางห้องปฏิบัติการ: Albumin {albumin} g/dL | อุณหภูมิ OR: {"<36°C (Hypothermia)" if hypothermia else "ปกติ"}',
        f'• การให้ยาปฏิชีวนะ: {atb_timing}'
    ]
    y_text = 175
    for d in details:
        draw.text((45, y_text), d, fill='#cbd5e1', font=f_body)
        y_text += 34

    # 3. Right Panel: Risk Score Banner & TreeSHAP
    risk_color = '#f43f5e' if risk_prob >= 15.0 else '#f59e0b' if risk_prob >= 7.0 else '#10b981'
    risk_bg = '#1e0a12' if risk_prob >= 15.0 else '#1e1405' if risk_prob >= 7.0 else '#041f17'

    draw.rectangle([620, 120, W-25, 475], fill=risk_bg, outline=risk_color, width=2)
    draw.text((640, 135), 'ผลการประเมินความเสี่ยงการติดเชื้อ (Predicted SSI Risk)', fill=risk_color, font=f_head)
    draw.line([640, 162, W-45, 162], fill=risk_color, width=1)

    draw.text((640, 180), f'{risk_prob:.1f}%', fill=risk_color, font=f_huge)
    
    # Badge Box with color indicator circle
    draw.rectangle([815, 192, W-45, 235], fill='#0f172a', outline=risk_color, width=1)
    draw.ellipse([830, 208, 842, 220], fill=risk_color)
    draw.text((852, 202), clean_badge, fill=risk_color, font=f_badge)

    draw.text((640, 255), 'ปัจจัยสำคัญที่สุดเฉพาะราย (Patient-Level TreeSHAP Impact):', fill='#ffffff', font=f_bold)
    y_shap = 285
    if top_shap:
        for name, val in top_shap[:5]:
            sign = '+' if val > 0 else ''
            c = '#f43f5e' if val > 0 else '#10b981'
            draw.text((645, y_shap), f'• {name}', fill='#cbd5e1', font=f_body)
            draw.text((W-110, y_shap), f'{sign}{val:.3f}', fill=c, font=f_bold)
            y_shap += 32
    else:
        draw.text((645, y_shap), '• ไม่พบปัจจัยเสี่ยงผิดปกติเด่นชัด', fill='#94a3b8', font=f_body)

    # 4. Bottom Panel: Clinical Action Plan
    draw.rectangle([25, 490, W-25, 785], fill='#090e1a', outline='#1e293b', width=1)
    draw.text((45, 505), 'มาตรการทางการแพทย์เฉพาะบุคคลที่แนะนำ (Clinical Care Plan: CDC / SIS Guidelines)', fill='#38bdf8', font=f_head)
    draw.line([45, 532, W-45, 532], fill='#1e293b', width=1)

    if not care_plans:
        care_plans = [
            f'1. การดูแลแผล: {"พิจารณาใช้ Closed-Incision NPWT (ciNPWT) 5-7 วัน" if risk_prob >= 15.0 else "ปิดแผลปลอดเชื้อคงไว้ 48 ชั่วโมง"}',
            '2. ยาปฏิชีวนะ: ให้ยาตามแนวทาง CPG โรงพยาบาล และยุติภายใน 24 ชม. หลังผ่าตัด',
            f'3. อุณหภูมิกาย: {"ใช้ Active Warming Blanket ใน PACU เพื่อคงอุณหภูมิ >36.5°C" if hypothermia else "รักษาอุณหภูมิตามเกณฑ์มาตรฐาน"}',
            f'4. การควบคุมน้ำตาล: {"ตรวจ Blood Sugar ทุก 4-6 ชม. คุมระดับ <180 mg/dL" if dm else "ตรวจติดตามตามคำสั่งแพทย์"}',
            f'5. การติดตามผล: {"นัดประเมินแผลเร็วขึ้นที่ 48-72 ชม. หลังจำหน่าย" if risk_prob >= 15.0 else "นัดตัดไหม/ตรวจแผล 7-10 วัน"}'
        ]

    y_plan = 548
    for cp in care_plans:
        draw.text((45, y_plan), f'•  {cp}', fill='#e2e8f0', font=f_body)
        y_plan += 42

    # 5. Footer and Doctor Signature
    draw.text((45, 810), 'TRIPOD+AI & PROBAST Calibrated Clinical Model | Somdech Phra Pinklao Hospital', fill='#64748b', font=f_sub)
    draw.text((W-460, 810), 'ลายมือชื่อศัลยแพทย์/พยาบาล: ________________________', fill='#94a3b8', font=f_sub)

    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=95)
    return buf.getvalue()
