# -*- coding: utf-8 -*-
"""
Created on Mon May  6 05:23:51 2024

@author: pruth
"""

from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, log_loss

class ModelEnsembler:
    def __init__(self, models):
        # Initialize with models, where models is a dictionary of trained models
        self.ensemble = VotingClassifier(estimators=list(models.items()), voting='soft')
        self.models = models

    def train_ensemble(self, X_train, y_train):
        print("Training ensemble...")
        # Split training data for model validation
        X_train_part, X_val, y_train_part, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        # Train the ensemble model
        self.ensemble.fit(X_train_part, y_train_part)

        # Evaluate the ensemble on validation data
        y_pred = self.ensemble.predict(X_val)
        y_proba = self.ensemble.predict_proba(X_val)[:, 1] if hasattr(self.ensemble, "predict_proba") else None

        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        logloss = log_loss(y_val, y_proba) if y_proba is not None else None
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_val, y_proba)
            roc_auc = auc(fpr, tpr)
            print(f"ROC AUC: {roc_auc:.4f}")
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Log Loss: {logloss:.4f}" if y_proba is not None else "")

    def get_ensemble(self):
        return self.ensemble
