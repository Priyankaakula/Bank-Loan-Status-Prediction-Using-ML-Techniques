import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier  # Import Random Forest classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, log_loss
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer, f1_score
scorer = make_scorer(f1_score, average='binary')  # Use 'macro' or 'weighted' for multi-class

class ModelTrainer:
    def __init__(self):
        self.models = {
            'logistic_regression': LogisticRegression(solver='saga', max_iter=2000),
            'knn': KNeighborsClassifier(),
            'random_forest': RandomForestClassifier(),  
            'naive_bayes': GaussianNB(),
            'xgboost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        }
        self.model = None

    def train_model(self, model_type, X_train, y_train):
        if model_type not in self.models:
            print(f"Error: '{model_type}' is not a supported model. Available models: {list(self.models.keys())}")
            return None
        
        print(f"Starting training for {model_type}...")
        self.model = self.models[model_type]
        
        # Split training data for model validation
        X_train_part, X_val, y_train_part, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        self.model_normal = self.models[model_type]
        self.model_normal.fit(X_train_part[:100], y_train_part[:100])
        y_pred = self.model_normal.predict(X_val)
        y_proba = self.model_normal.predict_proba(X_val)[:, 1] if hasattr(self.model_normal, "predict_proba") else None
        
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
        
        try:
            params = {
                'logistic_regression': {'C': [0.01, 0.1, 1]},
                'knn': {'n_neighbors': [5, 10, 15]},
                'random_forest': {'n_estimators': [50, 100, 200], 'max_features': ['sqrt', 'log2'], 'max_depth': [None, 10, 20, 30]},  
                'naive_bayes': {},
                'xgboost': {'learning_rate': [0.1, 0.2], 'max_depth': [3, 5, 7], 'n_estimators': [50, 100, 200]}


            }
            if params[model_type]:
                grid_search = GridSearchCV(self.model, param_grid=params[model_type], cv=3, scoring=scorer)
                grid_search.fit(X_train_part, y_train_part)
                self.model = grid_search.best_estimator_
                print("scores of cv",grid_search.cv_results_['mean_test_score'][-1])
                print("Best Score: train scores", grid_search.best_score_)

                print(f"Best parameters for {model_type}: {grid_search.best_params_}")
            else:
                self.model.fit(X_train_part, y_train_part)

            # Evaluate the model on validation data
            y_pred = self.model.predict(X_val)
            y_proba = self.model.predict_proba(X_val)[:, 1] if hasattr(self.model, "predict_proba") else None

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
            
            
            
        except Exception as e:
            print(f"An error occurred while training {model_type}: {e}")

        return self.model
