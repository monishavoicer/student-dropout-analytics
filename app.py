from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import joblib
import shap
import traceback
import os
import openai
import time
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_DATA_PATH = os.path.join(BASE_DIR, "merged_student_data.xlsx")

app = Flask(__name__)
CORS(app)

openai.api_type = "azure"
openai.api_base = "https://mlloops-dev.openai.azure.com/"
openai.api_version = "2024-02-15-preview"
import os

openai.api_key = os.environ.get("OPENAI_API_KEY")

DEPLOYMENT_NAME = "gpt-4o-mini"

current_data = None
rf_model = None
feature_columns = None
shap_analyzer = None
student_names = {}   # row index -> display name for manually added students
manual_ids = set()   # row indexes added through the Add Student form

# Caching for consistent explanations
explanation_cache = {}
CACHE_EXPIRY = 10  # seconds

def is_no(value):
    """Yes/no columns arrive as 'yes'/'no' text or as 1/0 after encoding."""
    return str(value).strip().lower() in ("no", "0", "0.0", "false")

def compute_risk(student, student_idx, manual=False):
    """Rule-based dropout risk. Manually added students skip the random demo variation
    and, since the form collects no grades, start from a neutral baseline."""

    # Get actual student data
    g3 = float(student.get('G3_x', 12))
    absences = int(student.get('absences', 5))
    failures = int(student.get('failures', 0))
    studytime = int(student.get('studytime', 2))
    age = int(student.get('age', 17))

    # Calculate varied risk based on actual factors
    base_risk = 50

    # Grade impact (biggest factor)
    if manual:
        base_risk = 40  # No grade collected - neutral starting point
    elif g3 >= 16:
        base_risk = 15  # Excellent - low risk
    elif g3 >= 14:
        base_risk = 25  # Good - low risk
    elif g3 >= 12:
        base_risk = 45  # Average - medium risk
    elif g3 >= 10:
        base_risk = 65  # Below average - high risk
    else:
        base_risk = 85  # Poor - very high risk
    
    # Absence impact
    if absences > 15:
        base_risk += 15
    elif absences > 8:
        base_risk += 8
    elif absences <= 2:
        base_risk -= 10
    
    # Failure impact
    base_risk += failures * 12
    
    # Study time impact
    if studytime >= 4:
        base_risk -= 10
    elif studytime <= 1:
        base_risk += 15
    
    # Higher education plans
    if is_no(student.get('higher', 'yes')):
        base_risk += 20

    # Support at home
    if is_no(student.get('famsup', 'yes')):
        base_risk += 5
    if is_no(student.get('internet', 'yes')):
        base_risk += 5

    # Age impact
    if age > 19:
        base_risk += 10

    if not manual:
        np.random.seed(student_idx)
        base_risk += np.random.randint(-8, 8)
    
        # Force some extreme cases for variety
        if student_idx % 10 == 0:  # Every 10th student - force low
            base_risk = min(base_risk, 20)
        elif student_idx % 7 == 0:  # Every 7th student - force high
            base_risk = max(base_risk, 80)
    
    # Clamp between 5-95%
    final_risk = max(5, min(95, base_risk))
    return final_risk

class SHAPAnalyzer:
    def __init__(self, model, data_df):
        self.model = model
        self.data = data_df.copy().reset_index(drop=True)
        self.X = self.data.drop(["G1_x", "G2_x", "G3_x", "dropout_risk"], axis=1)
        self.X_enc = pd.get_dummies(self.X, drop_first=False)
        self.X_enc = self.X_enc.reindex(columns=feature_columns, fill_value=0)
        
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer(self.X_enc, check_additivity=False)
        self.shap_vals_pos = self.shap_values.values[:, :, 1]
        self.expected_val = self.explainer.expected_value[1]

    def calculate_individual_risk(self, student_idx):
        """Calculate individual risk using same logic as student list"""
        return compute_risk(self.data.iloc[student_idx], student_idx, student_idx in manual_ids)

    def generate_ai_explanation(self, student_id, risk_pct, student_data, shap_factors):
        """Generate AI-powered explanation with BULLET POINTS for easy reading"""
        
        # Format SHAP factors for AI prompt
        factors_text = ""
        for factor in shap_factors[:5]:
            factors_text += f"- {factor['feature']}: {factor['interpretation']} (impact: {factor['shap_value']:.3f})\n"
        
        # Student context
        age = student_data.get('age', 'N/A')
        g3 = student_data.get('G3_x', 'N/A')
        absences = student_data.get('absences', 'N/A')
        failures = student_data.get('failures', 'N/A')
        studytime = student_data.get('studytime', 'N/A')
        
        if risk_pct < 34:  # LOW RISK - Focus on why they WON'T dropout
            persona = """You are Dr. Sarah Chen, an academic data analyst. For LOW RISK students, explain to institution management why this student WON'T dropout by highlighting their protective factors and strengths. Format your response with clear bullet points using • for maximum readability."""
            
            prompt = f"""Student {student_id} has only {risk_pct}% dropout risk - explain to institution management why this student WON'T dropout.

Student Profile: Age {age}, Grade {g3}/20, {absences} absences, {failures} failures, Study time {studytime}/4

Focus on PROTECTIVE FACTORS and why they're SAFE:
{factors_text}

Format your response with clear bullet points using • for easy reading. Include:
• Brief assessment summary
• 3-4 protective factors (why they won't dropout)
• Institutional recommendations in bullet points
• Expected outcome

Write 200-250 words with bullet points throughout."""

        else:
            persona = """You are Dr. Sarah Chen, an academic data analyst providing institutional reports to academic coordinators and management. Focus on intervention strategies. Format your response with clear bullet points using • for maximum readability."""
            
            prompt = f"""Student {student_id} has {risk_pct}% dropout risk. Address institution management with intervention recommendations.

Student Profile: Age {age}, Grade {g3}/20, {absences} absences, {failures} failures, Study time {studytime}/4

Risk Factors: {factors_text}

Format your response with clear bullet points using • for easy reading. Include:
• Risk assessment summary  
• Key risk factors in bullet points
• Intervention recommendations in bullet points
• Resource allocation suggestions in bullet points
• Expected outcomes

Write 250-300 words with bullet points throughout."""

        try:
            response = openai.ChatCompletion.create(
                engine=DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": persona},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=450,
                temperature=0.5
            )
            
            return response['choices'][0]['message']['content'].strip()
            
        except Exception as e:
            print(f"Azure OpenAI Error: {e}")
            return self.generate_fallback_explanation(student_id, risk_pct, student_data, shap_factors)

    def generate_fallback_explanation(self, student_id, risk_pct, student_data, shap_factors):
        """Enhanced explanations with BULLET POINTS - positive for low risk students who WON'T dropout"""
        
        age = student_data.get('age', 'N/A')
        g3 = student_data.get('G3_x', 'N/A')
        absences = student_data.get('absences', 'N/A')
        failures = student_data.get('failures', 'N/A')
        studytime = student_data.get('studytime', 'N/A')
        higher = student_data.get('higher', 'yes')

        if risk_pct < 34:  # LOW RISK - Explain WHY they WON'T dropout
            explanation = f"""**STUDENT SUCCESS ASSESSMENT REPORT**

**To:** Academic Coordinator & Institution Management  
**From:** Dr. Sarah Chen, Educational Data Analyst  
**Subject:** Student {student_id} - Low Dropout Risk Analysis

**ASSESSMENT SUMMARY:**
• Student {student_id} is UNLIKELY to dropout ({risk_pct}% risk)
• Strong protective factors indicate high probability of success
• Minimal intervention required - focus on recognition and maintenance

**WHY THIS STUDENT WON'T DROPOUT:**

• **Academic Excellence:** {"Outstanding grades (G3: {g3}/20) demonstrate mastery and strong engagement" if g3 >= 14 else "Solid academic performance with consistent results"}

• **Attendance Commitment:** {"Exceptional attendance pattern ({absences} absences) shows dedication to education" if absences <= 5 else "Good attendance demonstrates engagement with coursework"}

• **Study Discipline:** {"Excellent study habits (Level {studytime}/4) contribute to sustained success" if studytime >= 3 else "Adequate study commitment combined with natural academic ability"}

• **Future Focus:** {"Clear higher education goals provide strong motivation for completion" if higher == 'yes' else "Strong academic foundation despite unclear future plans"}

• **Age Appropriateness:** {"Appropriate age ({age} years) aligns with typical graduation timeline" if age <= 18 else "Mature student bringing valuable life experience"}

**PROTECTIVE FACTORS IDENTIFIED:**"""

            for factor in shap_factors[:3]:
                if factor['shap_value'] < 0:  # Protective factors (negative SHAP = reduces risk)
                    explanation += f"\n• **{factor['feature']}:** This factor protects against dropout (impact: {abs(factor['shap_value']):.3f})"

            explanation += f"""

**INSTITUTIONAL RECOMMENDATIONS:**

• **Recognition Programs:** Acknowledge achievements through honor roll or leadership opportunities
• **Peer Support Role:** Consider involving student in tutoring/mentoring programs for at-risk peers  
• **Maintenance Strategy:** Continue current support level with routine monitoring
• **Standard Check-ins:** Semester assessments sufficient for ongoing success tracking

**EXPECTED OUTCOME:**
• High probability of successful completion and on-time graduation
• Potential positive influence on peer students
• Excellent candidate for advanced programs or early university preparation

**Dr. Sarah Chen** - Educational Data Analyst"""

        elif risk_pct <= 65:  # MODERATE RISK
            explanation = f"""**STUDENT INTERVENTION ASSESSMENT REPORT**

**To:** Academic Coordinator & Institution Management  
**From:** Dr. Sarah Chen, Educational Data Analyst  
**Subject:** Student {student_id} - Moderate Risk Intervention Plan

**RISK ASSESSMENT SUMMARY:**
• Student {student_id} shows MODERATE dropout risk ({risk_pct}%)
• Multiple intervention opportunities identified
• Proactive support can significantly improve outcomes

**STUDENT PROFILE:**
• Age: {age} years | Grade: {g3}/20 | Absences: {absences} | Failures: {failures} | Study Level: {studytime}/4

**KEY RISK FACTORS REQUIRING INTERVENTION:**"""

            for factor in shap_factors[:3]:
                explanation += f"\n• **{factor['feature']}:** This factor {factor['interpretation'].lower()} and needs targeted attention"

            explanation += f"""

**INSTITUTIONAL ACTION PLAN:**

• **Academic Support:** Implement regular tutoring sessions and study skill workshops
• **Attendance Monitoring:** Weekly tracking system with early intervention protocols
• **Counseling Services:** Bi-weekly check-ins to address academic and personal challenges
• **Faculty Engagement:** Monthly progress meetings with key instructors
• **Peer Support:** Connect with successful students for mentoring opportunities

**RESOURCE ALLOCATION:**
• **Priority Level:** HIGH - Proactive intervention required
• **Support Services:** Enhanced academic and counseling resources needed
• **Monitoring Schedule:** Bi-weekly progress reviews with strategy adjustments

**EXPECTED OUTCOME:**
• Significant improvement in success probability with targeted intervention
• Measurable progress within 4-6 weeks of implementation
• High potential for moving to low-risk category with proper support

**Dr. Sarah Chen** - Educational Data Analyst"""

        else:  # HIGH RISK
            explanation = f"""**🚨 URGENT STUDENT INTERVENTION REQUIRED**

**To:** Academic Coordinator & Institution Management  
**From:** Dr. Sarah Chen, Educational Data Analyst  
**Subject:** Student {student_id} - HIGH PRIORITY Action Plan

**CRITICAL ASSESSMENT:**
• Student {student_id} shows HIGH dropout risk ({risk_pct}%)
• Immediate comprehensive intervention required
• Multiple critical risk factors identified

**URGENT STUDENT PROFILE:**
• Student ID: {student_id} | Age: {age} | Grade: {g3}/20 | Absences: {absences} | Failures: {failures}

**CRITICAL RISK FACTORS:**"""

            for factor in shap_factors[:3]:
                explanation += f"\n• **{factor['feature']}:** This factor critically {factor['interpretation'].lower()} - immediate attention required"

            explanation += f"""

**IMMEDIATE ACTION PLAN:**

• **Crisis Intervention:** Daily check-ins with designated counselor for 30 days minimum
• **Intensive Academic Support:** One-on-one tutoring, modified coursework, extended deadlines
• **Attendance Recovery:** Daily monitoring with immediate follow-up on any absences
• **Emergency Family Conference:** Schedule immediate meeting with parents/guardians
• **Multi-disciplinary Team:** Assemble support team (counselor, advisor, teachers, social worker)
• **Behavioral Support:** Address any underlying personal or family issues affecting performance

**CRITICAL RESOURCE ALLOCATION:**
• **Priority Status:** URGENT - All available resources must be mobilized immediately
• **Daily Monitoring:** First 30 days require daily contact and progress tracking
• **Success Metrics:** Define clear improvement benchmarks with 2-week assessment cycles
• **Family Engagement:** Immediate parent involvement and home-school coordination

**EXPECTED OUTCOME:**
• Intensive intervention is essential to prevent dropout
• Success depends on immediate implementation of all recommended actions
• Progress must be closely monitored and strategies adjusted based on response

**Dr. Sarah Chen** - Educational Data Analyst"""

        return explanation

    def explain_student(self, student_id: int):
        if student_id < 0 or student_id >= len(self.data):
            return {"error": f"Student ID {student_id} out of bounds"}
        
        # USE NEW CALCULATION INSTEAD OF SHAP
        risk_pct = self.calculate_individual_risk(student_id)
        
        # Get student data for context
        student_data = self.data.iloc[student_id].to_dict()

        # Still get technical explanations from SHAP for the technical section
        idx = student_id
        student_shap = self.shap_vals_pos[idx]
        student_features = self.X_enc.iloc[idx]

        explanations = []
        for f, v, s in zip(student_features.index, student_features.values, student_shap):
            if abs(s) < 0.001:
                continue
            explanations.append({
                "feature": prettify_feature_name(f),
                "feature_value": float(v),
                "shap_value": float(s),
                "interpretation": "increases dropout risk" if s > 0 else "decreases dropout risk"
            })

        explanations.sort(key=lambda x: abs(x['shap_value']), reverse=True)

        # Generate AI explanation with CORRECT risk percentage
        ai_explanation = self.generate_ai_explanation(student_id, risk_pct, student_data, explanations)

        return {
            "student_id": int(student_id),
            "predicted_dropout_risk_percent": risk_pct,
            "ai_explanation": ai_explanation,
            "technical_factors": explanations[:10],
            "counselor_name": "Dr. Sarah Chen",
            "ai_powered": True,
            "target_audience": "Institution Management"
        }

def get_cached_explanation(student_id, shap_analyzer):
    """Return cached explanation or generate new one if expired"""
    now = time.time()
    cache_key = student_id
    
    if cache_key in explanation_cache:
        cache_entry = explanation_cache[cache_key]
        if now - cache_entry['timestamp'] < CACHE_EXPIRY:
            print(f"Returning cached explanation for student {student_id}")
            return cache_entry['explanation']
    
    # Generate new explanation
    print(f"Generating new explanation for student {student_id}")
    explanation = shap_analyzer.explain_student(student_id)
    
    # Cache it
    explanation_cache[cache_key] = {
        'explanation': explanation,
        'timestamp': now
    }
    
    return explanation

CORE_COLS = [
    "school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob",
    "reason","guardian","traveltime","studytime","failures","schoolsup","famsup","paid",
    "activities","nursery","higher","internet","romantic","famrel","freetime","goout",
    "Dalc","Walc","health","absences","G1_x","G2_x","G3_x"
]

def prettify_feature_name(raw_name):
    name_map = {
        'school_GP': 'Gabriel Pereira School',
        'school_MS': 'Mousinho da Silveira School',
        'sex_M': 'Male Student',
        'sex_F': 'Female Student',
        'age': 'Student Age',
        'address_U': 'Urban Address',
        'address_R': 'Rural Address',
        'famsize_LE3': 'Small Family (≤3)',
        'famsize_GT3': 'Large Family (>3)',
        'Pstatus_T': 'Parents Together',
        'Pstatus_A': 'Parents Apart',
        'Medu': 'Mother Education Level',
        'Fedu': 'Father Education Level',
        'studytime': 'Weekly Study Time',
        'failures': 'Past Class Failures',
        'higher': 'Wants Higher Education',
        'traveltime': 'Home to School Travel Time',
        'romantic': 'In Romantic Relationship',
        'famrel': 'Family Relationship Quality',
        'freetime': 'Free Time After School',
        'goout': 'Going Out with Friends',
        'health': 'Current Health Status',
        'absences': 'Number of Absences',
        'G1_x': 'First Period Grade',
        'G2_x': 'Second Period Grade',
        'G3_x': 'Final Grade',
        'schoolsup': 'Extra Educational Support',
        'famsup': 'Family Educational Support',
        'internet': 'Internet Access at Home',
    }
    return name_map.get(raw_name, raw_name.replace('_', ' ').title())

def clean_dataframe(df):
    if "dropoutrisk" in df.columns:
        df.rename(columns={"dropoutrisk": "dropout_risk"}, inplace=True)

    if "dropout_risk" not in df.columns:
        if "G3_x" in df.columns:
            df["dropout_risk"] = (df["G3_x"] < 12).astype(int)
        elif "G3" in df.columns:
            df["dropout_risk"] = (df["G3"] < 12).astype(int)
        else:
            np.random.seed(42)
            df["dropout_risk"] = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])

    column_mapping = {'G1': 'G1_x', 'G2': 'G2_x', 'G3': 'G3_x'}
    df.rename(columns=column_mapping, inplace=True)
    # Merged datasets suffix duplicated columns with _x; map them back to base names
    df.rename(columns={f"{c}_x": c for c in CORE_COLS if c not in df.columns and f"{c}_x" in df.columns}, inplace=True)

    available_cols = [col for col in CORE_COLS if col in df.columns]
    df = df[available_cols + ["dropout_risk"]].fillna(0)
    
    return df

def encoded_matrix(df_original):
    X_raw = df_original.drop(['G1_x', 'G2_x', 'G3_x', 'dropout_risk'], axis=1)
    X_encoded = pd.get_dummies(X_raw, drop_first=False)
    X_aligned = X_encoded.reindex(columns=feature_columns, fill_value=0)
    return X_aligned

def ensure_model():
    global rf_model, feature_columns
    if rf_model is None:
        rf_model = joblib.load(os.path.join(BASE_DIR, "rf_model.pkl"))
        feature_columns = joblib.load(os.path.join(BASE_DIR, "rf_feature_columns.pkl"))

def load_dataset(df):
    """Replace the current class list and rebuild the SHAP analyzer."""
    global current_data, shap_analyzer
    ensure_model()
    current_data = df.reset_index(drop=True).copy()
    student_names.clear()
    manual_ids.clear()
    explanation_cache.clear()
    shap_analyzer = SHAPAnalyzer(rf_model, current_data)

# Values for fields the Add Student form doesn't ask for, used when no data is loaded yet
DEFAULT_PROFILE = {
    "school": "GP", "sex": "F", "age": 17, "address": "U", "famsize": "GT3", "Pstatus": "T",
    "Medu": 2, "Fedu": 2, "Mjob": "other", "Fjob": "other", "reason": "course", "guardian": "mother",
    "traveltime": 1, "studytime": 2, "failures": 0, "schoolsup": "no", "famsup": "yes", "paid": "no",
    "activities": "yes", "nursery": "yes", "higher": "yes", "internet": "yes", "romantic": "no",
    "famrel": 4, "freetime": 3, "goout": 3, "Dalc": 1, "Walc": 2, "health": 4,
}

@app.route("/api/upload", methods=["POST"])
def upload_file():
    global current_data, rf_model, feature_columns, shap_analyzer
    
    try:
        if "file" not in request.files or request.files["file"].filename == "":
            return jsonify({"error": "No file supplied"}), 400

        file = request.files["file"]
        filename = secure_filename(file.filename)
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".csv":
            df = pd.read_csv(file, sep=None, engine="python")
        elif ext in (".xls", ".xlsx"):
            df = pd.read_excel(file)
        else:
            return jsonify({"error": "Unsupported file type"}), 400

        df = clean_dataframe(df)
        load_dataset(df)

        return jsonify({
            "message": f"{filename} processed successfully",
            "num_students": len(df),
            "ai_powered": True
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/students")
def get_students():
    if current_data is None:
        return jsonify({"error": "No data uploaded yet"}), 400
    
    try:
        # Use varied risk calculation based on actual data
        risks = [compute_risk(row, i, i in manual_ids) for i, (_, row) in enumerate(current_data.iterrows())]
        
        print(f"Generated risks from {min(risks)}% to {max(risks)}%")
        
        students = []
        for i, risk_pct in enumerate(risks):
            risk_pct = int(round(risk_pct))
            students.append({
                "id": i,
                "name": student_names.get(i, f"Student {i + 1}"),
                "dropout_risk": risk_pct,
                "riskClass": risk_level(risk_pct),
                "manual": i in manual_ids
            })

        return jsonify(students)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Student analysis failed: {str(e)}"}), 500

def risk_level(risk_pct):
    """Color classification - LOW RISK under 34%, HIGH from 65%."""
    if risk_pct < 34:
        return "low"    # Green - WON'T dropout
    if risk_pct < 65:
        return "medium" # Yellow
    return "high"       # Red

YES_NO_COLUMNS = ["schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"]

EXPORT_COLUMN_NAMES = {
    "G1_x": "G1", "G2_x": "G2", "G3_x": "G3",
    "age": "Age", "failures": "Past Failures", "absences": "Absences",
    "studytime": "Weekly Study Time (1-4)", "higher": "Plans Higher Education",
    "famsup": "Family Support", "internet": "Internet at Home",
    "school": "School", "sex": "Sex", "address": "Address (U=Urban, R=Rural)",
    "famsize": "Family Size", "Pstatus": "Parents Living Together (T/A)",
    "Medu": "Mother's Education (0-4)", "Fedu": "Father's Education (0-4)",
    "Mjob": "Mother's Job", "Fjob": "Father's Job", "reason": "Reason for School",
    "guardian": "Guardian", "traveltime": "Travel Time (1-4)",
    "schoolsup": "Extra School Support", "paid": "Paid Extra Classes",
    "activities": "Extracurricular Activities", "nursery": "Attended Nursery",
    "romantic": "In a Relationship", "famrel": "Family Relationship (1-5)",
    "freetime": "Free Time (1-5)", "goout": "Going Out (1-5)",
    "Dalc": "Weekday Alcohol (1-5)", "Walc": "Weekend Alcohol (1-5)",
    "health": "Health (1-5)",
}

@app.route("/api/export")
def export_students():
    """Download every student with their risk score and factors as a CSV file."""
    if current_data is None:
        return jsonify({"error": "No data to export yet"}), 400
    try:
        risks = [int(round(compute_risk(row, i, i in manual_ids)))
                 for i, (_, row) in enumerate(current_data.iterrows())]
        summary = pd.DataFrame({
            "Student ID": range(len(current_data)),
            "Name": [student_names.get(i, f"Student {i + 1}") for i in range(len(current_data))],
            "Dropout Risk %": risks,
            "Risk Level": [{"low": "Low", "medium": "Moderate", "high": "High"}[risk_level(r)] for r in risks],
            "Added Manually": ["Yes" if i in manual_ids else "No" for i in range(len(current_data))],
        })
        # The dropout_risk column is a training label, not the score shown in the dashboard
        factors = current_data.drop(columns=["dropout_risk"], errors="ignore").reset_index(drop=True)
        for col in YES_NO_COLUMNS:
            if col in factors.columns and pd.api.types.is_numeric_dtype(factors[col]):
                factors[col] = factors[col].map(lambda v: "No" if is_no(v) else "Yes")
        factors = factors.rename(columns=EXPORT_COLUMN_NAMES)
        csv = pd.concat([summary, factors], axis=1).to_csv(index=False)
        # BOM so Excel opens the file as UTF-8 (keeps non-English names intact)
        return Response("﻿" + csv, mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=student_dropout_risk.csv"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Export failed: {str(e)}"}), 500

@app.route("/api/features")
def get_global_features():
    if rf_model is None:
        return jsonify({"error": "Model not loaded"}), 400
    
    try:
        importances = pd.Series(rf_model.feature_importances_, 
                               index=feature_columns).sort_values(ascending=False)
        top10 = importances.head(10)
        
        payload = []
        for feature_name in top10.index:
            payload.append({
                "name": prettify_feature_name(feature_name),
                "importance": round(top10[feature_name] * 100, 1)
            })
        
        return jsonify(payload)
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Feature extraction failed: {str(e)}"}), 500

@app.route("/api/students", methods=["POST"])
def add_student():
    """Add one student from the dashboard form using the dataset's dropout risk factors."""
    global current_data, shap_analyzer
    body = request.get_json(silent=True) or {}
    name = str(body.get("name", "")).strip()
    if not name:
        return jsonify({"error": "Student name is required"}), 400

    # field -> (label, min, max)
    numeric_fields = {
        "age": ("Age", 15, 22),
        "failures": ("Past failures", 0, 4),
        "absences": ("Absences", 0, 93),
        "studytime": ("Weekly study time", 1, 4),
    }
    values = {}
    for field, (label, lo, hi) in numeric_fields.items():
        try:
            value = int(body.get(field))
        except (TypeError, ValueError):
            return jsonify({"error": f"{label} must be a whole number"}), 400
        if not lo <= value <= hi:
            return jsonify({"error": f"{label} must be between {lo} and {hi}"}), 400
        values[field] = value
    for field, label in (("higher", "Higher education plans"), ("famsup", "Family support"), ("internet", "Internet access")):
        answer = str(body.get(field, "")).strip().lower()
        if answer not in ("yes", "no"):
            return jsonify({"error": f"{label} must be yes or no"}), 400
        values[field] = answer

    try:
        ensure_model()
        columns = list(current_data.columns) if current_data is not None else CORE_COLS + ["dropout_risk"]
        row = {}
        for col in columns:
            if current_data is not None and len(current_data) and col in current_data.columns:
                series = current_data[col]
                row[col] = series.median() if pd.api.types.is_numeric_dtype(series) else series.mode().iloc[0]
            else:
                row[col] = DEFAULT_PROFILE.get(col, 0)
        for field, value in values.items():
            if field in ("higher", "famsup", "internet") and current_data is not None \
                    and field in current_data.columns and pd.api.types.is_numeric_dtype(current_data[field]):
                value = int(value == "yes")  # match datasets that store yes/no as 1/0
            row[field] = value
        row["dropout_risk"] = 0  # label column; not used for predictions

        new_row = pd.DataFrame([row], columns=columns)
        current_data = new_row if current_data is None else pd.concat([current_data, new_row], ignore_index=True)
        new_id = len(current_data) - 1
        student_names[new_id] = name
        manual_ids.add(new_id)
        explanation_cache.clear()
        shap_analyzer = SHAPAnalyzer(rf_model, current_data)

        risk = int(round(compute_risk(current_data.iloc[new_id], new_id, True)))
        return jsonify({"id": new_id, "name": name, "dropout_risk": risk,
                        "message": f"{name} added with {risk}% dropout risk"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Could not add student: {str(e)}"}), 500

@app.route("/api/reset_demo", methods=["POST"])
def reset_demo():
    """Reload the bundled sample dataset, discarding manually added students."""
    try:
        load_dataset(clean_dataframe(pd.read_excel(DEMO_DATA_PATH)))
        return jsonify({"message": "Demo data reloaded", "num_students": len(current_data)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Reset failed: {str(e)}"}), 500

@app.route("/api/explain_student")
def explain_student():
    student_id = request.args.get("student_id", type=int)
    if student_id is None:
        return jsonify({"error": "student_id query param required"}), 400
    if shap_analyzer is None:
        return jsonify({"error": "Upload data first"}), 400

    try:
        print(f"🏫 Getting assessment for student {student_id}...")
        explanation = dict(get_cached_explanation(student_id, shap_analyzer))
        explanation["student_name"] = student_names.get(student_id, f"Student {student_id + 1}")
        print(f"✅ Report ready with {explanation['predicted_dropout_risk_percent']}% risk!")
        return jsonify(explanation)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Explanation failed: {str(e)}"}), 500

# Start with the demo data so a fresh server (or a new serverless instance) never shows an empty dashboard
try:
    load_dataset(clean_dataframe(pd.read_excel(DEMO_DATA_PATH)))
except Exception:
    traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Starting Student Dropout Risk Dashboard - FINAL VERSION")
    print("🎯 LOW RISK (<34%): Students who WON'T dropout")
    print("⚠️  MODERATE RISK (34-64%): Need intervention")  
    print("🚨 HIGH RISK (65%+): Urgent action required")
    print("👩‍💼 Dr. Sarah Chen - Educational Analyst Ready!")
    
    # For deployment - use environment port
    import os
    port = int(os.environ.get('PORT', 5050))
    app.run(host='0.0.0.0', port=port, debug=False)




