from azureml.core import Experiment
from azureml.train.automl import AutoMLConfig
from azureml.core.model import Model

def configure_automl(dataset, target_column, experiment_timeout_minutes=30):
    """Configure AutoML settings."""
    automl_config = AutoMLConfig(
        task="classification",
        primary_metric="AUC_weighted",
        training_data=dataset,
        label_column_name=target_column,
        n_cross_validations=5,
        max_concurrent_iterations=4,
        max_cores_per_iteration=-1,
        iterations=10,
        experiment_timeout_minutes=experiment_timeout_minutes,
        enable_early_stopping=True
    )
    return automl_config

def run_automl(workspace, experiment_name, automl_config):
    """Run AutoML experiment."""
    experiment = Experiment(workspace, experiment_name)
    automl_run = experiment.submit(automl_config, show_output=True)
    return automl_run

def get_best_model(automl_run):
    """Get the best model from AutoML run."""
    best_run, fitted_model = automl_run.get_output()
    return best_run, fitted_model

def register_model(workspace, model, model_name):
    """Register the model in Azure ML workspace."""
    model = Model.register(workspace=workspace, model_path=model.outputs['model_file'], model_name=model_name)
    print(f"Model {model_name} has been registered")
    return model

