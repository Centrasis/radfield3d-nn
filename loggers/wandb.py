from .logger import LoggerBase, Figure
import time
import os
from pytorch_lightning.loggers import WandbLogger
# check if wandb is installed
try:
    import wandb
    WANDB_FOUND = True
except:
    WANDB_FOUND = False


class WandBLogger(LoggerBase):
    """
    Logger for Weights & Biases (wandb).
    """

    def __init__(self, project_name: str = None, logs_dir: str = None, offline: bool = False):
        if not WANDB_FOUND:
            raise ImportError("WandB is not installed.")
        super().__init__(project_name=project_name, logs_dir=logs_dir)
        self.offline = offline

        if offline:
            wandb.init(mode="offline")

    def setup_experiment(self, experiment_name, settings):
        super().setup_experiment(experiment_name, settings)
        self.logger = WandbLogger(name=experiment_name, save_dir=self.logs_dir, project=self.project_name, offline=self.offline)

        self.logger.experiment.config["batch_size"] = settings.batch_size
        self.logger.experiment.config["dataset_path"] = settings.dataset_path
        self.logger.experiment.config["epochs"] = settings.epochs
        self.logger.experiment.config["num_workers"] = settings.num_workers
        self.logger.experiment.config["model_name"] = settings.model_name
        if settings.hyper_parameters is not None:
            self.logger.experiment.config.update(settings.hyper_parameters)
        if settings.data_augmentations is not None:
            for aug_name, aug_params in settings.data_augmentations:
                self.logger.experiment.config[f"aug_{aug_name}"] = aug_params
        if self.offline:
            self.logger.experiment.config["offline"] = True

    def get_lightning_callback(self) -> WandbLogger:
        """
        Returns the wandb logger callback for PyTorch Lightning.
        """
        return self.logger

    def reset_current_experiment(self):
        run = wandb.run
        run_id = run.id if run else None
        project_name = run.project if run else None
        entity_name = run.entity if run else None
        wandb.finish()
        time.sleep(5)
        api = wandb.Api()
        run_path = f"{entity_name}/{project_name}/{run_id}" if run_id else None
        if run_path:
            api_run = api.run(run_path)
            if api_run:
                api_run.delete()  # Delete the run to avoid conflicts
        wandb.finish()
        return super().reset_current_experiment()
    
    def log_model(self, model):
        wandb.watch(model, log="all")

    def finalize_logging(self):
        wandb.finish()
        return super().finalize_logging()

    def log_plot(self, name: str, plot: Figure, step: int = None):
        """
        Logs a plot to the logger.
        """
        if not os.path.exists("tmp"):
            os.makedirs("tmp")
        
        filename = f"{name}.html"
        plot.write_html(f"tmp/{filename}")
        
        log_data = {name: wandb.Html(f"tmp/{filename}")}
        if step is not None:
            wandb.log(log_data, step=step)
        else:
            wandb.log(log_data)
