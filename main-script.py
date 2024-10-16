from azureml.core import Workspace
import data_preprocessing as dp
import model_training as mt
import model_evaluation as me

def main():
    # Set up Azure ML workspace
    ws = Workspace.from_config()

    # Load and preprocess data
    data = dp.load_data("data/raw/customer_churn_data.csv")
    preprocessed_data = dp.preprocess_data(data)
    
    # Split data
    X_train, X_test, y_train, y_test = dp.split_data(preprocessed_data, target_column="Churn")
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = dp.scale_features(X_train, X_test)
    
    # Create Azure dataset
    train_dataset = dp.create_azure_dataset(ws, X_train_scaled.join(y_train), "train_data")
    
    # Configure and run AutoML
    automl_config = mt.configure_automl(train_dataset, target_column="Churn")
    automl_run = mt.run_automl(ws, "customer_churn_experiment", automl_config)
    
    # Get best model
    best_run, fitted_model = mt.get_best_model(automl_run)
    
    # Register model
    registered_model = mt.register_model(ws, best_run, "customer_churn_model")
    
    # Evaluate model
    y_pred = fitted_model.predict(X_test_scaled)
    y_prob = fitted_model.predict_proba(X_test_scaled)[:, 1]
    
    evaluation_results = me.evaluate_model(y_test, y_pred, y_prob)
    me.plot_roc_curve(y_test, y_prob)
    
    # Get and plot feature importance
    feature_importance = me.get_feature_importance(fitted_model, X_train.columns)
    if feature_importance:
        me.plot_feature_importance(feature_importance)

if __name__ == "__main__":
    main()

