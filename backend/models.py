import enum


class ArtifactPaths(enum.Enum):

    DATASET = "./artifacts/{token}-dataset.parquet"
    MODEL = "./artifacts/{token}-model.pickle"
    EVALUATION = "./artifacts/{token}-evaluation.json"
    TRAINING_ARGS = "./artifacts/{token}-training_args.json"


class Task(enum.Enum):

    REGRESSION = 'regression'
    CLASSIFICATION = 'classification'


class Iterations(enum.Enum):
    """
    Enum class for the number of iterations
    """
    LOW = 15
    MEDIUM = 30
    HIGH = 60
