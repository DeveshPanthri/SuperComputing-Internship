"""
Resource Recommender
=====================
Collaborative filtering-based recommender that suggests
learning resources based on student profile and knowledge gaps.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict


class ResourceRecommender:
    """Content + Collaborative filtering resource recommender."""
    
    def __init__(self):
        self.student_resource_matrix = None
        self.resources = {}
        self.similarity_matrix = None
        self.student_profiles = None
    
    def fit(self, interaction_df, resources, student_features_df=None):
        """
        Train the recommender.
        
        Args:
            interaction_df: DataFrame with student_id, resource_id, rating
            resources: Dict of resource metadata
            student_features_df: Optional student feature matrix
        """
        self.resources = resources
        
        # Build student-resource matrix
        n_students = interaction_df['student_id'].max() + 1
        n_resources = len(resources)
        
        self.student_resource_matrix = np.zeros((n_students, n_resources))
        
        for _, row in interaction_df.iterrows():
            sid = int(row['student_id'])
            rid = int(row['resource_id'])
            rating = row['rating']
            if sid < n_students and rid < n_resources:
                self.student_resource_matrix[sid, rid] = rating
        
        # Compute item-item similarity
        resource_matrix = self.student_resource_matrix.T
        # Add small noise to avoid zero vectors
        resource_matrix = resource_matrix + np.random.normal(0, 0.01, resource_matrix.shape)
        self.similarity_matrix = cosine_similarity(resource_matrix)
        
        # Store student profiles if provided
        if student_features_df is not None:
            self.student_profiles = student_features_df
        
        print(f"    Recommender trained: {n_students} students, {n_resources} resources")
        print(f"    Total interactions: {len(interaction_df)}")
        
        return self
    
    def recommend(self, student_id, weak_areas=None, top_n=3):
        """
        Recommend resources for a student.
        
        Args:
            student_id: Student index
            weak_areas: List of identified weak areas from gap detector
            top_n: Number of recommendations to return
        """
        if self.student_resource_matrix is None:
            return []
        
        n_students = self.student_resource_matrix.shape[0]
        student_id = min(student_id, n_students - 1)
        
        # Get student's existing ratings
        student_ratings = self.student_resource_matrix[student_id]
        
        # Calculate predicted ratings using item-item CF
        predicted_ratings = np.zeros(len(self.resources))
        
        for res_id in range(len(self.resources)):
            if student_ratings[res_id] == 0:  # Not yet interacted
                # Weighted sum of similar items the student has rated
                similarities = self.similarity_matrix[res_id]
                rated_mask = student_ratings > 0
                
                if rated_mask.sum() > 0:
                    numerator = np.sum(similarities[rated_mask] * student_ratings[rated_mask])
                    denominator = np.sum(np.abs(similarities[rated_mask])) + 1e-8
                    predicted_ratings[res_id] = numerator / denominator
            else:
                predicted_ratings[res_id] = -1  # Already seen
        
        # Boost scores based on weak areas
        if weak_areas:
            for area in weak_areas:
                severity_boost = {'high': 0.5, 'medium': 0.3, 'low': 0.1}.get(area.get('severity', 'low'), 0.1)
                
                for res_id, res_info in self.resources.items():
                    # Boost resources matching weak area difficulty
                    if area.get('severity') == 'high' and res_info['difficulty'] <= 2:
                        predicted_ratings[res_id] += severity_boost
                    elif area.get('severity') == 'medium' and res_info['difficulty'] == 2:
                        predicted_ratings[res_id] += severity_boost
                    elif area.get('severity') == 'low' and res_info['difficulty'] >= 2:
                        predicted_ratings[res_id] += severity_boost
        
        # Get top-N recommendations
        top_indices = np.argsort(predicted_ratings)[::-1][:top_n]
        
        recommendations = []
        for idx in top_indices:
            if predicted_ratings[idx] > 0 and idx in self.resources:
                rec = self.resources[idx].copy()
                rec['score'] = float(predicted_ratings[idx])
                rec['resource_id'] = idx
                recommendations.append(rec)
        
        # If not enough recommendations, add by difficulty matching
        if len(recommendations) < top_n:
            for res_id, res_info in self.resources.items():
                if len(recommendations) >= top_n:
                    break
                if not any(r['resource_id'] == res_id for r in recommendations):
                    rec = res_info.copy()
                    rec['score'] = 0.5
                    rec['resource_id'] = res_id
                    recommendations.append(rec)
        
        return recommendations[:top_n]


def save_recommender(recommender, path):
    """Save recommender model."""
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(recommender, f)
    print(f"    [SAVE] Recommender saved to: {path}")


def load_recommender(path):
    """Load recommender model."""
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)
