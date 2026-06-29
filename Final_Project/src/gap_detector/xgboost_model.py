"""
Gap Detector (XGBoost)
=======================
Identifies at-risk students and their weak areas using XGBoost classification.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report, f1_score, confusion_matrix,
                             roc_auc_score, accuracy_score)
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import os


def train_gap_detector(X_train, X_test, y_train, y_test):
    """Train XGBoost-style classifier for at-risk student detection."""
    print("\n--- Training Gap Detector (GradientBoosting) ---")
    
    # Using sklearn's GradientBoostingClassifier (no xgboost dependency needed)
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"    Accuracy:  {accuracy:.4f}")
    print(f"    F1-Score:  {f1:.4f}")
    print(f"    AUC-ROC:   {auc:.4f}")
    
    print(f"\n    Classification Report:")
    report = classification_report(y_test, y_pred, target_names=['Pass', 'At-Risk'])
    for line in report.split('\n'):
        print(f"    {line}")
    
    return model, {'accuracy': accuracy, 'f1': f1, 'auc': auc}


def get_feature_importance(model, feature_cols, top_n=15):
    """Get top feature importances from the model."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    print(f"\n    Top {top_n} Feature Importances:")
    result = []
    for i, idx in enumerate(indices):
        print(f"    {i+1:2d}. {feature_cols[idx]:<25} {importances[idx]:.4f}")
        result.append((feature_cols[idx], importances[idx]))
    
    return result


def identify_weak_areas(model, student_features, feature_cols):
    """
    Identify weak areas for a specific student.
    Returns concepts where the student is likely struggling.
    """
    prediction = model.predict_proba(student_features.reshape(1, -1))[0]
    risk_probability = prediction[1]
    
    # Analyze which features contribute most to risk
    weak_areas = []
    feature_importance = model.feature_importances_
    
    # Check grade-related features
    feature_dict = dict(zip(feature_cols, student_features))
    
    if feature_dict.get('G1', 20) < 10:
        weak_areas.append({'area': 'Period 1 Performance', 'severity': 'high',
                          'detail': f'G1 score: {feature_dict.get("G1", 0):.0f}/20'})
    if feature_dict.get('G2', 20) < 10:
        weak_areas.append({'area': 'Period 2 Performance', 'severity': 'high',
                          'detail': f'G2 score: {feature_dict.get("G2", 0):.0f}/20'})
    if feature_dict.get('grade_momentum', 0) < -2:
        weak_areas.append({'area': 'Declining Performance', 'severity': 'high',
                          'detail': f'Grade dropped by {abs(feature_dict.get("grade_momentum", 0)):.0f} points'})
    if feature_dict.get('failures', 0) > 0:
        weak_areas.append({'area': 'Past Failures', 'severity': 'medium',
                          'detail': f'{feature_dict.get("failures", 0):.0f} past class failures'})
    if feature_dict.get('absences', 0) > 10:
        weak_areas.append({'area': 'High Absences', 'severity': 'medium',
                          'detail': f'{feature_dict.get("absences", 0):.0f} absences recorded'})
    if feature_dict.get('studytime', 4) < 2:
        weak_areas.append({'area': 'Low Study Time', 'severity': 'medium',
                          'detail': 'Less than 2 hours weekly study time'})
    if feature_dict.get('alcohol_score', 0) > 3:
        weak_areas.append({'area': 'High Alcohol Consumption', 'severity': 'low',
                          'detail': 'Above average alcohol consumption'})
    
    if not weak_areas:
        weak_areas.append({'area': 'General Improvement', 'severity': 'low',
                          'detail': 'No specific weak areas identified'})
    
    return {
        'risk_probability': risk_probability,
        'is_at_risk': risk_probability > 0.5,
        'weak_areas': weak_areas
    }


def save_gap_detector(model, path):
    """Save the trained gap detector model."""
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"    [SAVE] Gap detector saved to: {path}")


def load_gap_detector(path):
    """Load a trained gap detector model."""
    with open(path, 'rb') as f:
        return pickle.load(f)
