"""Public API uses the same core as the CLI."""

from .baselines import fit_ahri10, predict_ahri10
from .calibration import calibrate
from .core import CompressorSpec, Dataset, Observation, OperatingPoint, load_dataset, load_points
from .design import rank_information_groups, training_information
from .development import run_development_study
from .discrepancy import add_residual_correction
from .evaluation import run_learning_curve
from .kernels import rank_kernel_groups
from .models import MODEL_REGISTRY, get_model
from .replay import run_adaptive_replay
from .workflow import benchmark, build_artifact, load_artifact, predict_point, recommend

__all__ = [
    "CompressorSpec",
    "OperatingPoint",
    "Observation",
    "Dataset",
    "load_dataset",
    "load_points",
    "get_model",
    "MODEL_REGISTRY",
    "calibrate",
    "fit_ahri10",
    "predict_ahri10",
    "build_artifact",
    "load_artifact",
    "predict_point",
    "recommend",
    "benchmark",
    "run_learning_curve",
    "run_adaptive_replay",
    "training_information",
    "rank_information_groups",
    "rank_kernel_groups",
    "add_residual_correction",
    "run_development_study",
]
