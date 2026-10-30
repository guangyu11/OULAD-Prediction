"""Early identification of at-risk students in the Open University Learning Analytics Dataset."""

from .config import CUTOFF_DAYS, REFERENCE_CUTOFF, REFERENCE_FEATURES, SEED, TARGET
from .data import OULADData, build_enrolments, prepare_assessments, prepare_vle
from .experiment import ExperimentResult, ExperimentRunner, best_by_validation
from .features import FeatureBuilder, feature_sets
from .models import BASELINE, MODELS, ModelSpec
from .modelling import Split, confusion_frame, describe_split, evaluate, make_split
from .pipeline import PipelineOutput, build_tables, recall_by_module, run_pipeline

__version__ = "1.0.0"
