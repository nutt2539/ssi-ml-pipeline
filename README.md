# 🏥 SSI Explainable Machine Learning Pipeline
**Development and Internal Validation of an Explainable Machine Learning Model for Predicting the Risk of Surgical Site Infection at the End of Surgery Using Preoperative and Intraoperative Data in Somdech Phra Pinklao Hospital: A Retrospective Cohort Study**

> ระบบปัญญาประดิษฐ์ต้นแบบสำหรับการพยากรณ์ความเสี่ยงการติดเชื้อแผลผ่าตัด (SSI) ณ จุดสิ้นสุดการผ่าตัด (End of Surgery) พร้อมการอธิบายผลลัพธ์ด้วย Explainable AI (TreeSHAP) ออกแบบตามมาตรฐาน **TRIPOD+AI (2024)** และ **PROBAST** สำหรับงานวิจัยแพทย์ประจำบ้านศัลยกรรม โรงพยาบาลสมเด็จพระปิ่นเกล้า กรมแพทย์ทหารเรือ

---

## 📁 โครงสร้างโฟลเดอร์โปรเจกต์ (Project Structure)

```
ssi_ml_pipeline/
├── data/                                 # โฟลเดอร์เก็บข้อมูลดิบและข้อมูลที่ผ่านการคลีน
│   ├── synthetic_surgical_cohort.csv    # ข้อมูลจำลองผู้ป่วย 4,500 ราย (2020-2024)
│   ├── train_cohort.csv                 # ข้อมูล Development set (ปี 2020-2023: 80%)
│   └── test_cohort.csv                  # ข้อมูล Temporal Validation set (ปี 2024: 20%)
│
├── models/                               # โฟลเดอร์เก็บโมเดลที่เทรนเสร็จแล้ว
│   ├── Logistic_Regression.pkl          # Baseline statistical model
│   ├── Random_Forest.pkl                # Random Forest model
│   ├── LightGBM.pkl                     # LightGBM champion model
│   ├── XGBoost.pkl                      # XGBoost classifier
│   ├── scaler.pkl                       # ตัวปรับสเกลข้อมูลตัวเลขต่อเนื่อง
│   └── metrics_summary.json             # สรุปคะแนนโมเดลเบื้องต้น
│
├── outputs/                              # ผลลัพธ์สำหรับนำไปใส่เล่มวิจัยและตีพิมพ์
│   ├── figures/                         # กราฟความละเอียดสูง (300 DPI)
│   │   ├── Figure1_Discrimination_Curves.png   # กราฟ ROC และ Precision-Recall Curves
│   │   ├── Figure2_Calibration_and_DCA.png     # กราฟ Calibration และ Decision Curve Analysis
│   │   ├── Figure3_SHAP_Global_Beeswarm.png    # กราฟ SHAP สรุป 18 ปัจจัยเสี่ยงสูงสุด
│   │   └── Figure4_SHAP_Waterfall_Case.png     # กราฟ Waterfall จำแนกรายคนไข้ใน OR
│   └── tables/                          # ตารางทางสถิติมาตรฐานวารสาร
│       ├── Table1_Baseline_Characteristics.csv  # ตารางที่ 1 ข้อมูลพื้นฐานพร้อมค่า p-value
│       ├── Table2_Model_Performance_Comparison.csv # ตารางที่ 2 เปรียบเทียบโมเดล + 95% CI
│       └── Table2_Model_Performance_Comparison.md  # ตารางที่ 2 รูปแบบ Markdown
│
├── src/                                  # โค้ดโมดูลหลัก (Core Modules)
│   ├── data_generator.py                # ตัวสร้างข้อมูลจำลองตามบริบท รพ.ปิ่นเกล้า
│   ├── preprocessor.py                  # ตัวคลีนข้อมูล, ตรวจ Zero Leakage, จัดการ Missing
│   ├── model_trainer.py                 # ตัวเทรนโมเดล จัดการ Class Imbalance (scale_pos_weight)
│   ├── evaluator.py                     # ตัวประเมิน 3 มิติ (AUROC, AUPRC, Brier, DCA, Bootstrap CI)
│   ├── explainability.py                # ตัวสกัด TreeSHAP Values (Global & Local)
│   └── report_generator.py              # ตัวสร้างตารางที่ 1 และ 2 ตามเกณฑ์สถิติแพทย์
│
├── app/                                  # เว็บแอปพลิเคชันสำหรับใช้งานหน้างาน
│   └── streamlit_app.py                 # เครื่องมือคำนวณความเสี่ยงท้ายห้องผ่าตัด
│
├── run_pipeline.py                       # สคริปต์หลักสั่งรันทุกขั้นตอนอัตโนมัติใน 1 คำสั่ง
├── setup_env.sh                          # สคริปต์ติดตั้งสภาพแวดล้อมและแพ็กเกจในคลิกเดียว
├── requirements.txt                      # รายการไลบรารี Python ที่ต้องใช้
└── README.md                             # คู่มือการใช้งานนี้
```

---

## ⚡ วิธีการติดตั้งและรันงานวิจัย (Quick Start)

### ขั้นตอนที่ 1: ติดตั้งไลบรารี (ครั้งแรกครั้งเดียว)
เปิด Terminal บนเครื่อง Mac แล้วเข้าไปที่โฟลเดอร์นี้ จากนั้นรันคำสั่ง:
```bash
cd /Users/nuttp./Desktop/Bestbest/ssi_ml_pipeline
./setup_env.sh
```
*หรือหากต้องการติดตั้งด้วยตัวเอง:*
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### ขั้นตอนที่ 2: รันกระบวนการ Machine Learning ทั้งหมด (One-Command Execution)
รันคำสั่งเพียงบรรทัดเดียว ระบบจะทำการเทรนโมเดล, ประเมินผล 3 มิติ, สร้างกราฟ SHAP และส่งออกตาราง Table 1 & 2:
```bash
python run_pipeline.py
```

### ขั้นตอนที่ 3: เปิดใช้งานเว็บแอปพลิเคชันจำลองท้ายห้องผ่าตัด (Bedside Web App)
```bash
streamlit run app/streamlit_app.py
```
เบราว์เซอร์จะเปิดหน้าเว็บขึ้นมาอัตโนมัติ ให้ศัลยแพทย์สามารถเลื่อนปรับค่าตัวแปร เช่น เวลาผ่าตัด, เลือดที่เสีย, Wound Class แล้วดูการคำนวณความเสี่ยงและกราฟแท่ง SHAP แบบ Real-time!

---

## 🔄 เมื่อได้ข้อมูลจริงจากศูนย์สารสนเทศ รพ.สมเด็จพระปิ่นเกล้า ต้องทำอย่างไร?

ระบบถูกออกแบบให้ **Plug-and-Play** เพื่อความง่ายของทีมวิจัย:
1. นำไฟล์ข้อมูลจริงจาก EMR ของโรงพยาบาลที่ De-identified แล้ว มาบันทึกทับไฟล์:  
   `data/synthetic_surgical_cohort.csv`  
   *(โดยตั้งชื่อคอลัมน์ให้ตรงตามตาราง Predictor Matrix)*
2. สั่งรันคำสั่งเดิม:
   ```bash
   python run_pipeline.py
   ```
3. ระบบจะสร้าง **Table 1**, **Table 2**, และกราฟผลลัพธ์ของข้อมูลจริงสำหรับใช้เขียนวิทยานิพนธ์และตีพิมพ์ทันที!

---

## 🛡️ มาตรการทางวิชาการที่สำคัญ (Academic Rigor)
- **Zero Data Leakage:** ป้องกันการนำตัวแปรหลังผ่าตัด (Post-op LOS, ICU, Post-op lab) มาใส่เด็ดขาด
- **Temporal Validation:** แบ่งชุดข้อมูลปี 2020–2023 สำหรับ Development และปี 2024 สำหรับ Validation จริง
- **Clinical Prediction Triad:** วัดผลครบทั้ง Discrimination (AUROC/AUPRC), Calibration (Brier/Slope), และ Clinical Utility (DCA)
- **Bootstrap 95% CIs:** คำนวณช่วงเชื่อมั่นผ่านการสุ่มซ้ำ 500-1,000 รอบ ตามมาตรฐานวารสารสากล
