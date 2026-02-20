#!/usr/bin/env python3
"""
================================================================================
ICT304 — AI System Development | Assignment 1 Prototype
================================================================================
SYSTEM NAME  : Early Academic Risk Prediction Engine
AI SUB-SYSTEM: Quiz Performance Trend Analyser + ML Failure Risk Classifier

DESCRIPTION:
  This prototype implements the AI sub-system of a larger Early Academic
  Warning System for a university trimester environment. It analyses a
  student's quiz scores entered so far, identifies performance trends,
  projects end-of-trimester quiz outcomes, and uses two trained ML models
  to classify risk of unit failure. Staff are alerted if a student is
  predicted to be at risk.

TECHNIQUES COMPARED (Assignment Requirement):
  - Technique 1: Logistic Regression (baseline, linear, interpretable)
  - Technique 2: Random Forest Classifier (ensemble, non-linear)
  Justification: Random Forest handles non-linear patterns and feature
  interactions better; Logistic Regression provides a transparent baseline.
  The system selects the model with the highest F1 score on the at-risk class.

UNIVERSITY ASSESSMENT STRUCTURE:
  Quizzes      : 10 x /20 marks  = 20% of unit
  Assignment 1 : /100 marks      = 15% of unit
  Assignment 2 : /100 marks      = 15% of unit
  Final Exam   : /100 marks      = 50% of unit
  Pass Mark    : 50 / 100

DATASET: Students_Grading_Dataset.csv (5000 students)
  Source: Kaggle — University Student Grading Dataset
  Note  : Dataset acknowledged to contain intentional bias.
          ML models are trained for pattern comparison purposes.
          Primary risk engine uses mathematically derived grade-path
          projection which is dataset-independent.

REFERENCES:
  - Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.
  - Pedregosa et al. (2011). Scikit-learn. JMLR 12, 2825–2830.

TEAM: IZAAN SHUMAIZ, ROHIT KUMAR, MOHAMED SINAN | ICT304 | FEB 2026
================================================================================
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import joblib

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# UNIVERSITY CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
DATASET_FILE    = "Students_Grading_Dataset.csv"

QUIZ_COUNT      = 10        # Total quizzes in trimester
QUIZ_MAX        = 20        # Max marks per quiz
QUIZ_UNIT_MARKS = 20.0      # Quizzes contribute 20 unit marks

A1_MAX          = 100       # Assignment 1 max marks
A1_UNIT_MARKS   = 15.0      # Assignment 1 contributes 15 unit marks

A2_MAX          = 100       # Assignment 2 max marks
A2_UNIT_MARKS   = 15.0      # Assignment 2 contributes 15 unit marks

FINAL_UNIT_MARKS= 50.0      # Final exam contributes 50 unit marks
PASS_MARK       = 50.0      # Unit pass mark out of 100

MIN_QUIZZES     = 3         # Minimum quizzes required for reliable prediction
ML_THRESHOLD    = 0.45      # ML probability threshold for at-risk flag

# Combined risk weights: quiz trend = 60%, assignment performance = 40%
QUIZ_RISK_WEIGHT   = 0.60
ASSIGN_RISK_WEIGHT = 0.40


# ══════════════════════════════════════════════════════════════════════════════
# INPUT VALIDATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def get_int_input(prompt, min_val, max_val):
    """Validated integer input — rejects out-of-range and non-numeric."""
    while True:
        raw = input(prompt).strip()
        if raw == "":
            print(f"   Please enter a number between {min_val} and {max_val}.")
            continue
        try:
            val = int(raw)
            if val < min_val or val > max_val:
                print(f"   Must be between {min_val} and {max_val}. Got {val}.")
            else:
                return val
        except ValueError:
            print(f"   Invalid input '{raw}'. Please enter a whole number.")


def get_float_input(prompt, min_val, max_val):
    """Validated float input — rejects out-of-range and non-numeric."""
    while True:
        raw = input(prompt).strip()
        if raw == "":
            print(f"   Please enter a number between {min_val} and {max_val}.")
            continue
        try:
            val = float(raw)
            if val < min_val or val > max_val:
                print(f"   Must be between {min_val} and {max_val}. Got {val}.")
            else:
                return round(val, 2)
        except ValueError:
            print(f"   Invalid input '{raw}'. Please enter a number.")


def get_optional_float(prompt, min_val, max_val):
    """Optional float input — pressing Enter returns None (skip)."""
    while True:
        raw = input(prompt).strip()
        if raw == "":
            return None
        try:
            val = float(raw)
            if val < min_val or val > max_val:
                print(f"   Must be between {min_val} and {max_val}. Got {val}.")
            else:
                return round(val, 2)
        except ValueError:
            print(f"   Invalid input. Press Enter to skip, or enter a number.")


# ══════════════════════════════════════════════════════════════════════════════
# AI SUB-SYSTEM: QUIZ TREND ANALYSER
# ══════════════════════════════════════════════════════════════════════════════
def analyse_quiz_trend(scores):
    """
    AI Sub-system: Quiz Performance Trend Analyser

    Analyses quiz scores entered so far and projects end-of-trimester
    performance using linear regression on the score sequence.

    Args:
        scores (list): Quiz scores entered so far (validated 0-QUIZ_MAX).

    Returns:
        dict: Trend analysis including slope, projected avg, risk level.
    """
    n   = len(scores)
    arr = np.array(scores, dtype=float)

    current_avg     = float(np.mean(arr))
    current_avg_pct = round(current_avg / QUIZ_MAX * 100.0, 1)

    # Fit linear trend to score sequence
    if n >= 2:
        x                   = np.arange(n, dtype=float)
        slope, intercept    = np.polyfit(x, arr, 1)
    else:
        slope     = 0.0
        intercept = current_avg

    # Project remaining quizzes (clamped to valid score range)
    projected = list(arr)
    for i in range(n, QUIZ_COUNT):
        val = float(intercept) + float(slope) * i
        projected.append(max(0.0, min(float(QUIZ_MAX), val)))

    projected_avg     = float(np.mean(projected))
    projected_avg_pct = round(projected_avg / QUIZ_MAX * 100.0, 1)
    projected_marks   = round(projected_avg / QUIZ_MAX * QUIZ_UNIT_MARKS, 2)
    current_marks     = round(current_avg / QUIZ_MAX * QUIZ_UNIT_MARKS, 2)

    if   slope >  0.5: trend = "Improving"
    elif slope < -0.5: trend = "Declining"
    else:              trend = "Stable"

    # Quiz-only risk score (0.0 = no risk, 1.0 = maximum risk)
    # Based on projected unit marks out of 20
    quiz_risk_score = max(0.0, min(1.0, 1.0 - (projected_marks / QUIZ_UNIT_MARKS)))

    return {
        'n'                   : n,
        'current_avg'         : round(current_avg, 2),
        'current_avg_pct'     : current_avg_pct,
        'current_marks'       : current_marks,
        'slope'               : round(slope, 3),
        'trend'               : trend,
        'projected_avg_pct'   : projected_avg_pct,
        'projected_marks'     : projected_marks,
        'quiz_risk_score'     : round(quiz_risk_score, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# AI SUB-SYSTEM: ASSIGNMENT RISK SCORER
# ══════════════════════════════════════════════════════════════════════════════
def score_assignment_risk(a1, a2):
    """
    Calculates a normalised assignment risk score (0-1) from any
    submitted assignment scores. If none submitted, returns neutral 0.5.
    """
    submitted = [s for s in [a1, a2] if s is not None]
    if not submitted:
        return 0.5  # neutral — no data

    avg_pct    = float(np.mean(submitted))
    # Risk = how far below 65% the student is (65% is a reasonable pass benchmark)
    risk_score = max(0.0, min(1.0, (65.0 - avg_pct) / 65.0))
    return round(risk_score, 4)


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED RISK DECISION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def compute_combined_risk(quiz_risk, assign_risk, a1, a2):
    """
    Combines quiz trend risk and assignment risk into a final risk level.
    If no assignments submitted, quiz risk carries full weight.
    """
    has_assignments = (a1 is not None or a2 is not None)

    if has_assignments:
        combined = (quiz_risk * QUIZ_RISK_WEIGHT) + (assign_risk * ASSIGN_RISK_WEIGHT)
    else:
        combined = quiz_risk  # quiz is only signal

    if   combined >= 0.55: level = "HIGH"
    elif combined >= 0.35: level = "MEDIUM"
    else:                  level = "LOW"

    return round(combined, 4), level


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load Dataset + Train & Compare ML Models
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  ICT304 — Early Academic Risk Prediction Engine  |  Prototype v1.0")
print("=" * 70)
print("  AI Sub-systems:")
print("    1. Quiz Trend Analyser    (linear trend projection)")
print("    2. ML Failure Classifier  (Logistic Regression vs Random Forest)")
print("=" * 70)

if not os.path.exists(DATASET_FILE):
    print(f"\n  ERROR: '{DATASET_FILE}' not found. Place it in the same folder.")
    sys.exit(1)

df        = pd.read_csv(DATASET_FILE)
df.columns = df.columns.str.strip()

# ML training: predict Grade F from Quizzes_Avg + Assignments_Avg
ML_FEAT = ['Quizzes_Avg', 'Assignments_Avg']
df_ml   = df[ML_FEAT + ['Grade']].dropna()
y_ml    = (df_ml['Grade'].str.strip() == 'F').astype(int)
X_ml    = df_ml[ML_FEAT]

Xtr, Xte, ytr, yte = train_test_split(
    X_ml, y_ml, test_size=0.2, random_state=42, stratify=y_ml
)

scaler = StandardScaler()
Xtr_sc = scaler.fit_transform(Xtr)
Xte_sc = scaler.transform(Xte)

# ── Technique 1: Logistic Regression ─────────────────────────────────────────
lr_model = LogisticRegression(
    class_weight='balanced',
    max_iter=2000,
    random_state=42,
    C=1.0
)
lr_model.fit(Xtr_sc, ytr)

# 5-fold cross-validation (more robust evaluation)
cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr_cv_f1 = cross_val_score(lr_model, Xtr_sc, ytr, cv=cv, scoring='f1').mean()

# ── Technique 2: Random Forest ───────────────────────────────────────────────
rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    max_depth=6,
    min_samples_leaf=5,
    random_state=42
)
rf_model.fit(Xtr, ytr)

rf_cv_f1 = cross_val_score(rf_model, Xtr, ytr, cv=cv, scoring='f1').mean()

# ── Evaluate on test set ──────────────────────────────────────────────────────
def compute_metrics(model, Xe, ye, use_sc=False):
    Xi    = scaler.transform(Xe) if use_sc else Xe
    proba = model.predict_proba(Xi)[:, 1]
    pred  = (proba >= ML_THRESHOLD).astype(int)
    cm    = confusion_matrix(ye, pred)
    return {
        'accuracy'  : round(accuracy_score(ye, pred),                   4),
        'precision' : round(precision_score(ye, pred, zero_division=0), 4),
        'recall'    : round(recall_score(ye, pred, zero_division=0),    4),
        'f1'        : round(f1_score(ye, pred, zero_division=0),        4),
        'roc_auc'   : round(roc_auc_score(ye, proba),                   4),
        'cv_f1'     : round(lr_cv_f1 if use_sc else rf_cv_f1,           4),
        'TP'        : int(cm[1, 1]) if cm.shape == (2, 2) else 0,
        'FP'        : int(cm[0, 1]) if cm.shape == (2, 2) else 0,
        'TN'        : int(cm[0, 0]) if cm.shape == (2, 2) else 0,
        'FN'        : int(cm[1, 0]) if cm.shape == (2, 2) else 0,
    }

lr_m = compute_metrics(lr_model, Xte, yte, use_sc=True)
rf_m = compute_metrics(rf_model, Xte, yte, use_sc=False)

# ── Model selection + justification ──────────────────────────────────────────
if rf_m['f1'] >= lr_m['f1']:
    best_model = rf_model
    best_name  = "Random Forest"
    use_sc     = False
    reason     = "higher F1 on at-risk class; better handles non-linear patterns"
else:
    best_model = lr_model
    best_name  = "Logistic Regression"
    use_sc     = True
    reason     = "higher F1 on at-risk class; stable and interpretable baseline"

print(f"\n  Dataset  : {len(df_ml)} students | Training: {len(Xtr)} | Test: {len(Xte)}")
print(f"  ML Label : Grade = F  (unit failure)\n")

print(f"  {'Metric':<14} {'Logistic Regression':>20} {'Random Forest':>16}  {'Better':>8}")
print("  " + "-" * 62)
metrics_to_show = [
    ('accuracy',   'Accuracy'),
    ('precision',  'Precision'),
    ('recall',     'Recall'),
    ('f1',         'F1-Score'),
    ('roc_auc',    'ROC-AUC'),
    ('cv_f1',      '5-Fold CV F1'),
]
for key, label in metrics_to_show:
    better = "LR " if lr_m[key] >= rf_m[key] else "RF "
    print(f"  {label:<14} {lr_m[key]:>20.4f} {rf_m[key]:>16.4f}  {better:>8}")

print(f"\n  Confusion Matrix:")
print(f"    Logistic Regression  — TP:{lr_m['TP']}  FP:{lr_m['FP']}  TN:{lr_m['TN']}  FN:{lr_m['FN']}")
print(f"    Random Forest        — TP:{rf_m['TP']}  FP:{rf_m['FP']}  TN:{rf_m['TN']}  FN:{rf_m['FN']}")

print(f"\n  ► Selected Model : {best_name}")
print(f"    Justification  : {reason}")
print(f"    Note           : ML is a pattern-matching secondary signal.")
print(f"                     Primary prediction uses quiz trend projection.")

joblib.dump({
    'model'     : best_model,
    'scaler'    : scaler,
    'name'      : best_name,
    'use_scaler': use_sc,
    'lr_metrics': lr_m,
    'rf_metrics': rf_m,
    'threshold' : ML_THRESHOLD,
}, "risk_model.joblib")

print(f"\n  Model saved → risk_model.joblib")

# ══════════════════════════════════════════════════════════════════════════════
# STUDENT ASSESSMENT LOOP
# ══════════════════════════════════════════════════════════════════════════════
run_another = 'y'
student_num = 0

while run_another.strip().lower() == 'y':
    student_num += 1

    print(f"\n{'=' * 70}")
    print(f"  Student {student_num} Assessment")
    print(f"{'=' * 70}")

    # ── Quiz input with validation ────────────────────────────────────────────
    num_q = get_int_input(
        f"  Quizzes completed (min {MIN_QUIZZES}, max {QUIZ_COUNT}): ",
        MIN_QUIZZES, QUIZ_COUNT
    )

    quiz_scores = []
    for i in range(num_q):
        s = get_float_input(f"    Quiz {i+1:>2} score (0–{QUIZ_MAX}): ", 0, QUIZ_MAX)
        quiz_scores.append(s)

    # ── Assignment input (optional, press Enter to skip) ─────────────────────
    print(f"\n  Assignment scores — press Enter to skip if not yet submitted/marked:")
    a1 = get_optional_float(f"    Assignment 1 score (0–{A1_MAX}) or Enter to skip: ", 0, A1_MAX)
    a2 = None
    if a1 is not None:
        a2 = get_optional_float(f"    Assignment 2 score (0–{A2_MAX}) or Enter to skip: ", 0, A2_MAX)

    # ── Run AI sub-systems ────────────────────────────────────────────────────
    trend     = analyse_quiz_trend(quiz_scores)
    a_risk    = score_assignment_risk(a1, a2)
    combined_score, risk_level = compute_combined_risk(
        trend['quiz_risk_score'], a_risk, a1, a2
    )

    # ── ML secondary prediction ───────────────────────────────────────────────
    assign_pct_for_ml = float(np.mean(
        [s for s in [a1, a2] if s is not None]
    )) if (a1 is not None or a2 is not None) else trend['current_avg_pct']

    student_row = pd.DataFrame(
        [[trend['current_avg_pct'], assign_pct_for_ml]], columns=ML_FEAT
    )
    lr_prob  = float(lr_model.predict_proba(scaler.transform(student_row))[0, 1])
    rf_prob  = float(rf_model.predict_proba(student_row)[0, 1])
    ml_prob  = lr_prob if use_sc else rf_prob
    ml_flag  = ml_prob >= ML_THRESHOLD

    # ── Compute unit marks from submitted components ──────────────────────────
    quiz_marks  = round(trend['current_avg_pct'] / 100.0 * QUIZ_UNIT_MARKS, 2)
    a1_marks    = round((a1 / A1_MAX) * A1_UNIT_MARKS, 2) if a1 is not None else None
    a2_marks    = round((a2 / A2_MAX) * A2_UNIT_MARKS, 2) if a2 is not None else None
    earned      = quiz_marks + (a1_marks or 0.0) + (a2_marks or 0.0)
    available   = QUIZ_UNIT_MARKS + (A1_UNIT_MARKS if a1 is not None else 0) + \
                  (A2_UNIT_MARKS if a2 is not None else 0)

    # ── Display results ───────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"  PREDICTION RESULTS — Student {student_num}")
    print(f"{'─' * 70}")
    print(f"  Quizzes ({num_q}/{QUIZ_COUNT} done, each out of {QUIZ_MAX})")
    print(f"    Scores entered    : {quiz_scores}")
    print(f"    Current average   : {trend['current_avg_pct']:.1f}%  "
          f"({trend['current_marks']:.1f} unit marks so far)")
    print(f"    Trend             : {trend['trend']}  "
          f"(slope = {trend['slope']:+.2f} per quiz)")
    print(f"    Projected avg     : {trend['projected_avg_pct']:.1f}%  "
          f"→  {trend['projected_marks']:.1f} / {QUIZ_UNIT_MARKS:.0f} projected unit marks")

    if a1 is not None:
        print(f"  Assignment 1 : {a1:.0f}/{A1_MAX}  →  {a1_marks:.1f}/{A1_UNIT_MARKS:.0f} unit marks")
    if a2 is not None:
        print(f"  Assignment 2 : {a2:.0f}/{A2_MAX}  →  {a2_marks:.1f}/{A2_UNIT_MARKS:.0f} unit marks")

    print(f"  {'─' * 66}")
    print(f"  Marks earned (submitted) : {earned:.1f} / {available:.0f} available so far")
    print(f"  Final exam (50 marks)    : not yet sat")
    print(f"")
    print(f"  ── ML Comparison ──────────────────────────────────────────────")
    print(f"  Logistic Regression  : {lr_prob:.1%} risk  "
          f"{' flagged' if lr_prob >= ML_THRESHOLD else ' ok'}")
    print(f"  Random Forest        : {rf_prob:.1%} risk  "
          f"{' flagged' if rf_prob >= ML_THRESHOLD else ' ok'}")
    print(f"  Selected Model ({best_name[:2]}): {ml_prob:.1%} risk")
    print(f"  ───────────────────────────────────────────────────────────────")
    print(f"  Combined Risk Score  : {combined_score:.2f}  "
          f"(quiz trend {QUIZ_RISK_WEIGHT*100:.0f}% + "
          f"{'assignment ' + str(ASSIGN_RISK_WEIGHT*100)[:2] + '%' if (a1 or a2) else 'quiz-only, no assignments yet'})")
    print()

    # ── Status + Staff Alert ──────────────────────────────────────────────────
    if risk_level == "HIGH":
        print(f"  STATUS  :  *** HIGH RISK ***")
        print(f"  MODEL   :  {best_name} selected for this prediction")
        print(f"  ACTION  :  NOTIFY STAFF — Urgent intervention required")
        print(f"\n  ── Staff Alert {'─' * 51}")
        print(f"  Subject : URGENT — Student {student_num} at HIGH academic risk")
        print()
        reasons = []
        if trend['projected_avg_pct'] < 50:
            reasons.append(f"Quiz trend projects only {trend['projected_avg_pct']:.1f}% average "
                           f"({trend['projected_marks']:.1f}/{QUIZ_UNIT_MARKS:.0f} unit marks)")
        if trend['trend'] == "Declining":
            reasons.append(f"Scores are declining (slope = {trend['slope']:+.2f} per quiz)")
        if trend['current_avg_pct'] < 40:
            reasons.append(f"Current quiz average critically low at {trend['current_avg_pct']:.1f}%")
        if a1 is not None and a1 < 50:
            reasons.append(f"Assignment 1 ({a1:.0f}/100) is below pass mark")
        if a2 is not None and a2 < 50:
            reasons.append(f"Assignment 2 ({a2:.0f}/100) is below pass mark")
        if not reasons:
            reasons.append("Combined quiz + assignment risk score is high")
        for r in reasons:
            print(f"  • {r}")
        print(f"  {'─' * 66}")

    elif risk_level == "MEDIUM":
        print(f"  STATUS  :    MEDIUM RISK")
        print(f"  MODEL   :  {best_name} selected for this prediction")
        print(f"  ACTION  :  NOTIFY STAFF — Recommend check-in meeting")
        print(f"\n  ── Staff Alert {'─' * 51}")
        print(f"  Subject : Student {student_num} may need academic support")
        print()
        if trend['trend'] == "Declining":
            print(f"  • Scores are declining (slope = {trend['slope']:+.2f}). Early support recommended.")
        if trend['current_avg_pct'] < 65:
            print(f"  • Quiz average {trend['current_avg_pct']:.1f}% is below the recommended 65%.")
        if a1 is not None and a1 < 65:
            print(f"  • Assignment 1 ({a1:.0f}/100) is below recommended performance level.")
        print(f"  {'─' * 66}")

    else:
        print(f"  STATUS  :    ON TRACK")
        print(f"  MODEL   :  {best_name} selected for this prediction")
        print(f"  ACTION  :  No alert required. Continue monitoring.")
        if trend['trend'] == "Declining":
            print(f"  Note    :  Scores declining (slope = {trend['slope']:+.2f}) — worth watching.")

    # ── Save report ───────────────────────────────────────────────────────────
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "system"              : "ICT304 Early Academic Risk Prediction Engine",
        "generated_at"        : datetime.now().strftime("%Y-%m-%d %H:%M"),
        "student_number"      : student_num,
        "unit_structure"      : {
            "quizzes"         : f"{QUIZ_COUNT} x /{QUIZ_MAX} = {QUIZ_UNIT_MARKS:.0f} unit marks",
            "assignment_1"    : f"/{A1_MAX} = {A1_UNIT_MARKS:.0f} unit marks",
            "assignment_2"    : f"/{A2_MAX} = {A2_UNIT_MARKS:.0f} unit marks",
            "final_exam"      : f"{FINAL_UNIT_MARKS:.0f} unit marks",
            "pass_mark"       : f"{PASS_MARK}/100",
        },
        "student_input"       : {
            "quizzes_completed": num_q,
            "quiz_scores"      : quiz_scores,
            "assignment_1"     : a1 if a1 is not None else "Not submitted",
            "assignment_2"     : a2 if a2 is not None else "Not submitted",
        },
        "quiz_trend_analysis" : {
            "current_avg_pct"      : trend['current_avg_pct'],
            "current_unit_marks"   : trend['current_marks'],
            "trend_direction"      : trend['trend'],
            "trend_slope"          : trend['slope'],
            "projected_avg_pct"    : trend['projected_avg_pct'],
            "projected_unit_marks" : trend['projected_marks'],
            "out_of_unit_marks"    : QUIZ_UNIT_MARKS,
            "quiz_risk_score"      : trend['quiz_risk_score'],
        },
        "assignment_analysis" : {
            "a1_unit_marks"          : a1_marks,
            "a2_unit_marks"          : a2_marks,
            "assignment_risk_score"  : a_risk,
        },
        "marks_summary"       : {
            "earned_so_far"   : round(earned, 2),
            "available_so_far": available,
            "final_exam"      : "not yet sat",
        },
        "ml_comparison"       : {
            "technique_1"     : "Logistic Regression",
            "technique_2"     : "Random Forest",
            "lr_risk_prob"    : round(lr_prob, 4),
            "rf_risk_prob"    : round(rf_prob, 4),
            "selected_model"  : best_name,
            "selected_model_prob": round(ml_prob, 4),
            "ml_threshold"    : ML_THRESHOLD,
            "lr_test_metrics" : lr_m,
            "rf_test_metrics" : rf_m,
            "justification"   : reason,
        },
        "combined_prediction" : {
            "combined_risk_score"   : combined_score,
            "risk_level"            : risk_level,
            "at_risk"               : risk_level in ('HIGH', 'MEDIUM'),
            "staff_action"          : (
                "URGENT INTERVENTION" if risk_level == "HIGH"
                else "Check-in meeting" if risk_level == "MEDIUM"
                else "Monitor"
            ),
        },
    }

    fname = f"Risk_Report_Student{student_num}_{ts}.json"
    with open(fname, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved → {fname}")
    print(f"{'=' * 70}")

    run_another = input("\n  Assess another student? (y/n): ")

print("\n  System closed.\n")
