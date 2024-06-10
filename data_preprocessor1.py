import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler,OneHotEncoder
from imblearn.over_sampling import SMOTE

class DataPreprocessor:
    def __init__(self, filepath):
        self.data = pd.read_csv(filepath)
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.scaler = StandardScaler()
        print(f"Data loaded from {filepath}. Initial shape: {self.data.shape}")

    def preprocess(self):
        print("Starting preprocessing...")

        # Drop unnecessary columns
        self.data.drop(columns=['Loan ID', 'Customer ID', 'Months since last delinquent'], inplace=True)
        print(f"Columns dropped. Shape after dropping columns: {self.data.shape}")

        # Drop duplicates
        initial_count = self.data.shape[0]
        self.data.drop_duplicates(inplace=True)
        duplicates_dropped = initial_count - self.data.shape[0]
        print(f"{duplicates_dropped} duplicates found and dropped. Shape after dropping duplicates: {self.data.shape}")

        # Drop rows with missing values
        initial_count = self.data.shape[0]
        self.data.dropna(inplace=True)
        missing_values_dropped = initial_count - self.data.shape[0]
        print(f"{missing_values_dropped} rows with missing values dropped. Shape after dropping missing values: {self.data.shape}")
        # Encode categorical columns
       # for col in self.data.columns:
        #   if self.data[col].dtype == 'object':
          #    original_unique_values = self.data[col].nunique()
           #   self.data[col] = OneHotEncoder().fit_transform(self.data[col])
           #   print(f"Encoded column {col}. Number of unique values was: {original_unique_values}")
     
        # Removing outliers specifically in the 'Credit Score' column
        # Apply One-Hot Encoding to categorical columns except the target variable 'loan_status'
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        print("Categorical columns before excluding target:", categorical_cols)

        # Exclude the target variable 'loan_status' from the categorical columns
        categorical_cols = categorical_cols[categorical_cols != 'Loan Status']
        print("Categorical columns after excluding target:", categorical_cols)

        # Apply one-hot encoding to the remaining categorical columns
        self.data = pd.get_dummies(self.data, columns=categorical_cols, drop_first=False)
        print("Applied One-Hot Encoding. Columns now:", self.data.columns)
        print("Data after encoding:", self.data)

        self.remove_outliers()

        # Check if 'Loan Status' column exists before proceeding
        if 'Loan Status' in self.data.columns:
            X = self.data.drop('Loan Status', axis=1)
            y = self.data['Loan Status']
        else:
            print("Error: 'Loan Status' column not found.")
            return  # Stop further processing if critical column is missing
        mapping_dict = {'Charged Off': 0, 'Fully Paid': 1}
        y = y.map(mapping_dict)
        #done with one hot encoding
        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Data split into training and testing sets. Training shape: {self.X_train.shape}, Testing shape: {self.X_test.shape}")

        # Print class distribution before applying SMOTE
        print("Class distribution before SMOTE:")
        print(self.y_train.value_counts())

        # Apply SMOTE to balance the training set

        smote = SMOTE(random_state=42)


        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
        print(f"Applied SMOTE. Balanced training data shape: {self.X_train.shape}")

        # Print class distribution after applying SMOTE
        print("Class distribution after SMOTE:")
        print(pd.Series(self.y_train).value_counts())

        # Print data before scaling
        print("Data before scaling:")
        print(self.X_train[:5])  # Display first 5 rows

        # Scale the features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        # Print data after scaling
        print("Data after scaling:")
        print(self.X_train[:5])  # Display first 5 rows
    def remove_outliers(self):
        # Focus on 'Credit Score' column
        if 'Credit Score' in self.data.columns:
            # Calculate IQR for 'Credit Score'
            Q1 = self.data['Credit Score'].quantile(0.25)
            Q3 = self.data['Credit Score'].quantile(0.75)
            IQR = Q3 - Q1

            # Define outliers as those that are below Q1 - 1.5*IQR or above Q3 + 1.5*IQR
            condition = (self.data['Credit Score'] < (Q1 - 1.5 * IQR)) | (self.data['Credit Score'] > (Q3 + 1.5 * IQR))
            outliers = self.data[condition]

            # Print outliers
            print("Outliers in 'Credit Score':")
            print(outliers[['Credit Score']])  # You can add more columns to print if necessary

            # Remove outliers
            self.data = self.data[~condition]
            print(f"Outliers removed. New data shape: {self.data.shape}")
        else:
            print("No 'Credit Score' column found in the dataset.")

        print("Preprocessing completed.")

# Example usage
if __name__ == "__main__":
    filepath = r"C:\Users\pinky\Data Mining\archive\loan_status.csv"
    preprocessor = DataPreprocessor(filepath)
    preprocessor.preprocess()
