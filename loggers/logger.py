from typing import Any
from typing import NamedTuple, List, Tuple, Dict
from plotly.graph_objects import Figure


class TrainingSettings(NamedTuple):
    batch_size: int
    dataset_path: str
    epochs: int
    num_workers: int
    model_name: str
    dataset_loading_mode: str
    hyper_parameters: dict = None
    data_augmentations: List[Tuple[str, Dict[str, float]]] = None


class LoggerBase(object):
    """
    Base class for all loggers like wandb or MLflow.
    """

    def __init__(self, project_name: str = None, logs_dir: str = None):
        self.project_name = project_name
        self.logs_dir = logs_dir

    def setup_experiment(self, experiment_name: str, settings: TrainingSettings):
        """
        Sets up the experiment with the given name and training settings.
        Constructs the lightning logger callback object.
        """
        self._experiment_name = experiment_name
        self._training_settings = settings

    def log_model(self, model):
        """
        Specify a model to log during training.
        If not supported by the specific logger, this is just ignored.
        """
        pass

    @property
    def training_settings(self) -> TrainingSettings:
        return self._training_settings
    
    @property
    def experiment_name(self) -> str:
        return self._experiment_name

    def get_lightning_callback(self) -> Any:
        """
        Returns a callback function for logging.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def log_plot(self, name: str, plot: Figure, step: int = None):
        """
        Logs a plot to the logger.
        If step is provided, it will be used for epoch-bound history.
        """
        raise NotImplementedError("Subclasses must implement this method for logging plots.")
    
    def reset_current_experiment(self):
        """
        Resets the current experiment and deletes any associated resources.
        """
        self.logger = None

    def finalize_logging(self):
        """
        Finalizes the logging process and invalidates current logger callback.
        """
        self.logger = None
