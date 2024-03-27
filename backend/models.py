import enum


class ArtifactPaths(enum.Enum):

    DATASET = "./artifacts/{token}-dataset.parquet"
    MODEL = "./artifacts/{token}-model.pickle"
    EVALUATION = "./artifacts/{token}-evaluation.json"
    TRAINING_ARGS = "./artifacts/{token}-training_args.json"
    STATUS = "./artifacts/{token}-status.json"
    METADATA = "./artifacts/{token}-metadata.json"


class Task(enum.Enum):

    REGRESSION = 'regression'
    CLASSIFICATION = 'classification'
