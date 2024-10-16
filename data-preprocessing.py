import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from azureml.core import Dataset

def load_data(file_path):
    """Load the data from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Preprocess the data."""
    # Drop any rows with missing values
    data = data.dropna()
    
    # Convert categorical variables to dummy variables
    data = pd.get_dummies(data, drop_first=True)
    
    return data

def split_data(data, target_column):
    """Split the data into features and target, then into train and test sets."""
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def scale_features(X_train, X_test):
    """Scale the features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler

def create_azure_dataset(workspace, dataframe, name):
    """Create an Azure ML dataset from a pandas dataframe."""
    return Dataset.from_pandas_dataframe(dataframe, name=name)

