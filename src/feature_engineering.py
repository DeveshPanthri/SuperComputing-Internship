"""
Feature Engineering Module
==========================
Prepares features for DKT, XGBoost, and Recommender models.
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle

def load_processed_data():
    """Load the processed student performance data."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "processed", "student_performance_processed.csv")
    return pd.read_csv(data_path)

def engineer_features(df):
    """Create all engineered features for model training."""
    
    # --- Numeric features for XGBoost ---
    feature_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
                    'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences',
                    'G1', 'G2']
    
    # Encode categorical columns
    cat_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob',
                'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
                'nursery', 'higher', 'internet', 'romantic']
    
    label_encoders = {}
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            feature_cols.append(f'{col}_encoded')
    
    # --- Derived features ---
    # Grade momentum (is the student improving or declining?)
    df['grade_momentum'] = df['G2'] - df['G1']
    feature_cols.append('grade_momentum')
    
    # Average of G1 and G2
    df['avg_grade_so_far'] = (df['G1'] + df['G2']) / 2
    feature_cols.append('avg_grade_so_far')
    
    # Parent education average
    df['parent_edu_avg'] = (df['Medu'] + df['Fedu']) / 2
    feature_cols.append('parent_edu_avg')
    
    # Social activity score
    df['social_score'] = (df['goout'] + df['freetime']) / 2
    feature_cols.append('social_score')
    
    # Alcohol consumption score
    df['alcohol_score'] = (df['Dalc'] + df['Walc']) / 2
    feature_cols.append('alcohol_score')
    
    # Study-to-social ratio
    df['study_social_ratio'] = df['studytime'] / (df['social_score'] + 0.1)
    feature_cols.append('study_social_ratio')
    
    # Absence rate category
    df['high_absence'] = (df['absences'] > df['absences'].median()).astype(int)
    feature_cols.append('high_absence')
    
    return df, feature_cols, label_encoders

def create_sequences_for_dkt(df):
    """
    Create sequential data for DKT model.
    Simulates concept-level interactions from G1, G2, G3 progression.
    Each student has a sequence of 'concept attempts' derived from their grades.
    """
    sequences = []
    
    # Define subject concepts based on grade ranges
    concepts = ['fundamentals', 'intermediate', 'advanced', 'application', 'mastery']
    n_concepts = len(concepts)
    
    for idx, row in df.iterrows():
        student_seq = []
        
        # Generate concept interactions from G1 period
        g1_mastery = row['G1'] / 20.0  # Normalize to 0-1
        for c_id in range(n_concepts):
            threshold = (c_id + 1) / n_concepts
            correct = 1 if g1_mastery >= threshold else 0
            student_seq.append({
                'student_id': idx,
                'concept_id': c_id,
                'correct': correct,
                'timestamp': 1,
                'grade_period': 'G1'
            })
        
        # Generate concept interactions from G2 period
        g2_mastery = row['G2'] / 20.0
        for c_id in range(n_concepts):
            threshold = (c_id + 1) / n_concepts
            correct = 1 if g2_mastery >= threshold else 0
            student_seq.append({
                'student_id': idx,
                'concept_id': c_id,
                'correct': correct,
                'timestamp': 2,
                'grade_period': 'G2'
            })
        
        # Generate concept interactions from G3 period
        g3_mastery = row['G3'] / 20.0
        for c_id in range(n_concepts):
            threshold = (c_id + 1) / n_concepts
            correct = 1 if g3_mastery >= threshold else 0
            student_seq.append({
                'student_id': idx,
                'concept_id': c_id,
                'correct': correct,
                'timestamp': 3,
                'grade_period': 'G3'
            })
        
        sequences.append(student_seq)
    
    return sequences

def prepare_xgboost_data(df, feature_cols):
    """Prepare train/test splits for XGBoost at-risk classifier."""
    X = df[feature_cols].copy()
    y = df['at_risk'].copy()  # Binary: 1 = at-risk, 0 = pass
    
    # Handle any NaN/inf
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def prepare_recommender_data(df):
    """
    Create student-resource interaction matrix for collaborative filtering.
    We simulate resource interactions based on student profiles.
    """
    resources = {
        0: {'name': 'Basic Math Fundamentals - Khan Academy', 'type': 'video', 'difficulty': 1, 'url': 'https://youtube.com/watch?v=basics'},
        1: {'name': 'Algebra Practice Problems Set', 'type': 'practice', 'difficulty': 2, 'url': 'https://example.com/algebra'},
        2: {'name': 'Statistics for Beginners PDF', 'type': 'pdf', 'difficulty': 1, 'url': 'https://example.com/stats.pdf'},
        3: {'name': 'Advanced Calculus Tutorial', 'type': 'video', 'difficulty': 3, 'url': 'https://youtube.com/watch?v=calculus'},
        4: {'name': 'Linear Algebra Complete Course', 'type': 'video', 'difficulty': 3, 'url': 'https://youtube.com/watch?v=linalg'},
        5: {'name': 'Probability & Combinatorics', 'type': 'practice', 'difficulty': 2, 'url': 'https://example.com/probability'},
        6: {'name': 'Study Skills & Time Management', 'type': 'pdf', 'difficulty': 1, 'url': 'https://example.com/study.pdf'},
        7: {'name': 'Exam Preparation Strategies', 'type': 'video', 'difficulty': 1, 'url': 'https://youtube.com/watch?v=exam'},
        8: {'name': 'Interactive Math Quizzes', 'type': 'practice', 'difficulty': 2, 'url': 'https://example.com/quizzes'},
        9: {'name': 'Critical Thinking Workshop', 'type': 'video', 'difficulty': 3, 'url': 'https://youtube.com/watch?v=critical'},
    }
    
    # Simulate ratings based on student mastery and resource difficulty match
    np.random.seed(42)
    interactions = []
    
    for idx, row in df.iterrows():
        mastery = row.get('mastery_level', 2)
        for res_id, res_info in resources.items():
            # Students more likely to interact with difficulty-appropriate resources
            diff_match = 1.0 - abs(mastery/4.0 - res_info['difficulty']/3.0)
            prob = max(0.1, min(0.9, diff_match + np.random.normal(0, 0.15)))
            
            if np.random.random() < prob:
                rating = max(1, min(5, int(3 + (diff_match * 2) + np.random.normal(0, 0.5))))
                interactions.append({
                    'student_id': idx,
                    'resource_id': res_id,
                    'rating': rating
                })
    
    interaction_df = pd.DataFrame(interactions)
    return interaction_df, resources


def run_feature_engineering():
    """Main function to run all feature engineering."""
    print("=" * 70)
    print("PHASE 3: FEATURE ENGINEERING")
    print("=" * 70)
    
    # Load data
    df = load_processed_data()
    print(f"\n[DATA] Loaded processed data: {df.shape}")
    
    # Engineer features
    df, feature_cols, label_encoders = engineer_features(df)
    print(f"[OK] Engineered {len(feature_cols)} features")
    print(f"     Features: {feature_cols[:5]}... +{len(feature_cols)-5} more")
    
    # Create DKT sequences
    sequences = create_sequences_for_dkt(df)
    print(f"[OK] Created {len(sequences)} student sequences for DKT")
    print(f"     Each sequence has {len(sequences[0])} interactions (5 concepts x 3 periods)")
    
    # Prepare XGBoost data
    X_train, X_test, y_train, y_test = prepare_xgboost_data(df, feature_cols)
    print(f"[OK] XGBoost data split: Train={len(X_train)}, Test={len(X_test)}")
    print(f"     At-risk ratio - Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}")
    
    # Prepare recommender data
    interaction_df, resources = prepare_recommender_data(df)
    print(f"[OK] Recommender data: {len(interaction_df)} interactions, {len(resources)} resources")
    
    # Save everything
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, "data", "processed")
    
    df.to_csv(os.path.join(processed_dir, "features_engineered.csv"), index=False)
    interaction_df.to_csv(os.path.join(processed_dir, "resource_interactions.csv"), index=False)
    
    # Save feature columns and encoders
    with open(os.path.join(processed_dir, "feature_config.pkl"), 'wb') as f:
        pickle.dump({
            'feature_cols': feature_cols,
            'label_encoders': label_encoders,
            'resources': resources
        }, f)
    
    # Save sequences
    with open(os.path.join(processed_dir, "dkt_sequences.pkl"), 'wb') as f:
        pickle.dump(sequences, f)
    
    print(f"\n[SAVE] All feature-engineered data saved to: {processed_dir}")
    
    return df, feature_cols, sequences, X_train, X_test, y_train, y_test, interaction_df, resources


if __name__ == "__main__":
    run_feature_engineering()
