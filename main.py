import os
import gradio as gr
from models import ModelTrainer
from ensemble import ModelEnsembler
from data_visualizer import DataVisualizer
from data_preprocessor1 import DataPreprocessor
import pandas as pd

class LoanStatusPredictor:
    def __init__(self):
        self.trained_models = {}
        self.ensemble_model = None

    def train_models(self, data):
        # Train individual models
        model_trainer = ModelTrainer()
        models_to_train = ['logistic_regression', 'knn', 'random_forest', 'naive_bayes', 'xgboost']
        print("dataset")
        print(data.X_train, data.y_train)
        for model_type in models_to_train:
            model = model_trainer.train_model(model_type, data.X_train, data.y_train)
            if model:
                self.trained_models[model_type] = model
        print("Models trained: " + ", ".join(self.trained_models.keys()))

    def ensemble_models(self, data):
        # Ensemble the trained models
    
        ensembler = ModelEnsembler(self.trained_models)
        self.ensemble_model = ensembler.train_ensemble(data.X_train, data.y_train)
    def save_visualizations(self, data):
        visualizer = DataVisualizer()
        for column in data.columns:
            visualizer.visualize_distribution(data, column)
            try:
                 visualizer.visualize_boxplot(data, column)
            except:
                 print(f"Failed to generate boxplot for {column}")
            try:
                 visualizer.visualize_count(data, column)
            except:
                 print(f"Failed to generate count plot for {column}")
            try:
                  visualizer.visualize_scatter(data, column, column)  # Assumes scatterplot needs the same column for x and y
            except:
                 print(f"Failed to generate scatter plot for {column}")
            try:
                 visualizer.visualize_heatmap(data)
            except:
                 print("Failed to generate heatmap")   

    def predict_loan_status(self, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p):
        # Predict loan status using the ensemble model
        print([a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p])
        # Use ensemble model if available
        if self.ensemble_model:
            prediction = self.ensemble_model.predict([[a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p]])
        else:
            prediction = self.trained_models['logistic_regression'].predict([[a, 2, c, d, e, 6, 7, h, i, j, k, l, m, n, o,]+[0 for i in range(27)]])
        print(prediction)
        
        return "Fully Paid" if prediction[0] == 1 else "Charged Off"

def main():
    # Load the dataset and preprocess it
    data = DataPreprocessor(r"C:\Users\pinky\Data Mining\archive\loan_status.csv")
    data.preprocess()

    # Initialize LoanStatusPredictor
    predictor = LoanStatusPredictor()

    # Train models
    predictor.train_models(data)

    # Ensemble models
    predictor.ensemble_models(data)

    # Save visualizations
    predictor.save_visualizations(pd.read_csv(r"C:\Users\pinky\Data Mining\archive\loan_status.csv").head(1000))

     # Define Gradio components
    components = [
         gr.Number(label="Current Loan Amount"),
         gr.Radio(choices=["Short Term", "Long Term"], label="Term"),
         gr.Number(label="Credit Score"),
         gr.Number(label="Annual Income"),
         gr.Number(label="Years in current job"),
         gr.Radio(choices=["Home Mortgage", "Rent", "Own Home", "HaveMortgage"], label="Home Ownership"),
         gr.Radio(choices=["Debt Consolidation", "Home Improvements", "Other", "Business Loan", "Buy House", "Buy Car", "Medical Bills", "Take a Trip", "Educational Expenses"], label="Purpose"),
        gr.Number(label="Monthly Debt"),
         gr.Number(label="Years of Credit History"),
         gr.Number(label="Months since last delinquent"),
         gr.Number(label="Number of Open Accounts"),
         gr.Number(label="Number of Credit Problems"),
         gr.Number(label="Current Credit Balance"),
         gr.Number(label="Maximum Open Credit"),
        gr.Number(label="Bankruptcies"),
        gr.Number(label="Tax Liens")
     ]
    # Create Gradio interface
    gr.Interface(fn=predictor.predict_loan_status, inputs=components, outputs="text", title="Loan Status Predictor", description="Enter the details to predict the loan status:").launch()

if __name__ == "__main__":
    main()
