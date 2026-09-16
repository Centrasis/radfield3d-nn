from .logger import LoggerBase, Figure
# check if mlflow is installed
try:
    import mlflow
    MLFLOW_FOUND = True
except:
    MLFLOW_FOUND = False
from pytorch_lightning.loggers import MLFlowLogger as plMLFlowLogger
import os
import re


class MLFlowLogger(LoggerBase):
    """
    Logger for MLflow.
    """

    def __init__(self, project_name: str = None, logs_dir: str = None):
        if not MLFLOW_FOUND:
            raise ImportError("MLflow is not installed.")
        super().__init__(project_name=project_name, logs_dir=logs_dir)
        
        # Handle network paths (UNC paths like \\server.de\folder)
        if logs_dir.startswith(('\\\\', '//')):
            self.tracking_uri = f"file://{logs_dir.replace('\\', '/')}"

        elif not logs_dir.startswith(('file://', 'http://', 'https://', 'sqlite://', 'mysql://', 'postgresql://')):
            # Local path - convert to file URI
            self.tracking_uri = f"file://{logs_dir}"
        else:
            # Already a proper URI
            self.tracking_uri = logs_dir

        if project_name is not None:
            self.tracking_uri = f"{self.tracking_uri}/{project_name}"

    def setup_experiment(self, experiment_name, settings):
        # End current run if active and optionally delete it
        super().setup_experiment(experiment_name, settings)
        if mlflow.active_run():
            run_id = mlflow.active_run().info.run_id
            #experiment_id = mlflow.active_run().info.experiment_id
            mlflow.end_run()
            
            # Delete the current run
            try:
                mlflow.delete_run(run_id)
            except Exception as e:
                print(f"Warning: Could not delete run {run_id}: {e}")
        
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(experiment_name)
        mlflow.start_run()

        self.logger = plMLFlowLogger(
            experiment_name=mlflow.get_experiment(mlflow.active_run().info.experiment_id).name,
            save_dir=os.path.join(self.logs_dir, self.project_name) if self.project_name is not None else self.logs_dir,
            tracking_uri=mlflow.get_tracking_uri(),
            run_id=mlflow.active_run().info.run_id,
            log_model=True
        )

        # Log parameters
        self.logger.log_hyperparams({
            "batch_size": settings.batch_size,
            "dataset_path": settings.dataset_path,
            "epochs": settings.epochs,
            "num_workers": settings.num_workers,
            "model_name": settings.model_name,
            "dataset_loading_mode": settings.dataset_loading_mode
        })
        
        if settings.hyper_parameters is not None:
            self.logger.log_hyperparams(settings.hyper_parameters)
            
        if settings.data_augmentations is not None:
            aug_params = {}
            for aug_name, aug_params_dict in settings.data_augmentations:
                for key, value in aug_params_dict.items():
                    aug_params[f"aug_{aug_name}_{key}"] = value
            self.logger.log_hyperparams(aug_params)

    def get_lightning_callback(self) -> plMLFlowLogger:
        """
        Returns the MLflow logger callback for PyTorch Lightning.
        """
        return self.logger

    def reset_current_experiment(self):
        # End current run if active and optionally delete it
        if mlflow.active_run():
            run_id = mlflow.active_run().info.run_id
            #experiment_id = mlflow.active_run().info.experiment_id
            mlflow.end_run()
            
            # Delete the current run
            try:
                mlflow.delete_run(run_id)
            except Exception as e:
                print(f"Warning: Could not delete run {run_id}: {e}")
        self.logger = None
        return super().reset_current_experiment()
    
    def log_model(self, model):
        """
        Log model with MLflow. Note: This logs the model architecture info.
        """
        if hasattr(model, '__class__'):
            mlflow.log_param("model_class", model.__class__.__name__)
        
        # Count parameters
        if hasattr(model, 'parameters'):
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            mlflow.log_param("total_parameters", total_params)
            mlflow.log_param("trainable_parameters", trainable_params)

    def finalize_logging(self):
        if mlflow.active_run():
            mlflow.end_run()
        self.logger = None
        return super().finalize_logging()

    def log_plot(self, name: str, plot: Figure, step: int = None):
        """
        Logs a plot to MLflow using the native log_figure method.
        If step is provided, it will be used for epoch-bound history.
        """
        # Sanitize artifact path to comply with MLflow allowed characters: [a-zA-Z0-9/._-]
        safe_name = re.sub(r"[^a-zA-Z0-9_\-./]", "_", name).strip("/")
        if step is not None:
            mlflow.log_figure(plot, f"{safe_name}/step_{step}.html")
        else:
            mlflow.log_figure(plot, f"{safe_name}.html")
