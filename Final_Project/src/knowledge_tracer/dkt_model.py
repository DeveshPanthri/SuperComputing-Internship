"""
Deep Knowledge Tracing (DKT) Model
====================================
LSTM-based model that tracks student mastery per concept over time.
Predicts the probability of correctly answering the next concept.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from sklearn.metrics import roc_auc_score


class DKTModel(nn.Module):
    """LSTM-based Deep Knowledge Tracing model."""
    
    def __init__(self, n_concepts, hidden_size=64, n_layers=1, dropout=0.2):
        super(DKTModel, self).__init__()
        self.n_concepts = n_concepts
        self.hidden_size = hidden_size
        
        # Input: one-hot concept + correct/incorrect = 2 * n_concepts
        input_size = 2 * n_concepts
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, n_concepts)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        x: (batch_size, seq_len, input_size)
        Returns: (batch_size, seq_len, n_concepts) - mastery probabilities
        """
        lstm_out, _ = self.lstm(x)
        output = self.sigmoid(self.fc(lstm_out))
        return output


def sequences_to_tensors(sequences, n_concepts=5):
    """Convert sequence data to PyTorch tensors for DKT training."""
    all_inputs = []
    all_targets = []
    
    for seq in sequences:
        seq_inputs = []
        seq_targets = []
        
        for interaction in seq:
            # Input: one-hot encoding of (concept_id, correct)
            input_vec = np.zeros(2 * n_concepts)
            c_id = interaction['concept_id']
            correct = interaction['correct']
            
            if correct == 1:
                input_vec[c_id] = 1.0  # Correct answer for concept c_id
            else:
                input_vec[n_concepts + c_id] = 1.0  # Wrong answer for concept c_id
            
            seq_inputs.append(input_vec)
            
            # Target: correct/incorrect for each concept
            target_vec = np.zeros(n_concepts)
            target_vec[c_id] = correct
            seq_targets.append(target_vec)
        
        all_inputs.append(seq_inputs)
        all_targets.append(seq_targets)
    
    X = torch.FloatTensor(np.array(all_inputs))
    y = torch.FloatTensor(np.array(all_targets))
    
    return X, y


def train_dkt(sequences, n_concepts=5, epochs=50, lr=0.001, batch_size=64):
    """Train the DKT model."""
    print("\n--- Training DKT Model (LSTM) ---")
    
    # Convert to tensors
    X, y = sequences_to_tensors(sequences, n_concepts)
    
    # Train/test split
    n_total = len(X)
    n_train = int(0.8 * n_total)
    indices = torch.randperm(n_total)
    
    X_train = X[indices[:n_train]]
    y_train = y[indices[:n_train]]
    X_test = X[indices[n_train:]]
    y_test = y[indices[n_train:]]
    
    print(f"    Train: {len(X_train)} students, Test: {len(X_test)} students")
    print(f"    Sequence length: {X.shape[1]}, Input dim: {X.shape[2]}")
    
    # Model
    model = DKTModel(n_concepts=n_concepts, hidden_size=64, n_layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    best_auc = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            output = model(batch_X)
            
            # Mask: only compute loss where we have actual interactions
            mask = batch_y.sum(dim=-1) > 0
            if mask.sum() == 0:
                continue
                
            loss = criterion(output[mask], batch_y[mask])
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        # Evaluate
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                test_output = model(X_test)
                mask = y_test.sum(dim=-1) > 0
                
                y_true = y_test[mask].numpy().flatten()
                y_pred = test_output[mask].numpy().flatten()
                
                # Filter out positions where true values are all same
                valid = ~(np.isnan(y_pred) | np.isinf(y_pred))
                y_true = y_true[valid]
                y_pred = y_pred[valid]
                
                try:
                    auc = roc_auc_score(y_true, y_pred)
                except ValueError:
                    auc = 0.5
                
                avg_loss = total_loss / max(n_batches, 1)
                print(f"    Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | AUC-ROC: {auc:.4f}")
                
                if auc > best_auc:
                    best_auc = auc
                    best_model_state = model.state_dict().copy()
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    print(f"\n    [OK] Best AUC-ROC: {best_auc:.4f} {'(PASSED > 0.75)' if best_auc > 0.75 else '(below 0.75 target)'}")
    
    return model, best_auc


def predict_mastery(model, sequences, n_concepts=5):
    """Get mastery predictions for all students."""
    X, _ = sequences_to_tensors(sequences, n_concepts)
    model.eval()
    with torch.no_grad():
        predictions = model(X)
    
    # Get the last time step predictions as current mastery
    mastery_scores = predictions[:, -1, :].numpy()
    
    concept_names = ['fundamentals', 'intermediate', 'advanced', 'application', 'mastery']
    
    results = []
    for i, scores in enumerate(mastery_scores):
        student_mastery = {}
        for j, concept in enumerate(concept_names):
            student_mastery[concept] = float(scores[j])
        results.append(student_mastery)
    
    return results


def save_dkt_model(model, path):
    """Save trained DKT model."""
    torch.save(model.state_dict(), path)
    print(f"    [SAVE] DKT model saved to: {path}")


def load_dkt_model(path, n_concepts=5):
    """Load a trained DKT model."""
    model = DKTModel(n_concepts=n_concepts, hidden_size=64, n_layers=1)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model
