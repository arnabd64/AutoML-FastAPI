import os
from typing import Any, Dict

import pandas as pd
from flaml import AutoML
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from backend.models import Task


def dataset_preprocess(df: pd.DataFrame):
    """
    Preprocess the dataframe before training the model.
    The following preprocessing steps are performed:
    - Remove features that contain string values
    - Missing value treatment
    - Convert 'object' type to Categorical
    - Convert non-negative 'int64' type to 'uint32'
    - Convert 'int64' type to 'int32'
    - Convert 'float64' type to 'float32'

    Arguments:
    - `df`: pd.DataFrame - The dataframe to be preprocessed

    Returns:
    - pd.DataFrame: The preprocessed dataframe
    """
    for column, dtype in zip(df.columns, df.dtypes.values):
        # Remove features that contain string values
        if dtype == 'object' and pd.unique(df[column]).size > 10:
            df.drop(column, axis=1, inplace=True)
            continue
        
        # Missing value treatment
        if dtype == 'object':
            # fill missing values with the mode
            df[column] = df[column].fillna(df[column].mode()[0])
        else:
            # fill missing values with the median
            df[column] = df[column].fillna(df[column].median())

        # Convert 'object' type to Categorical
        if dtype == 'object':
            df[column] = df[column].astype('category')
            continue

        # Convert non-negative 'int64' type to 'uint32'
        if dtype == 'int64' and df[column].min() >= 0:
            df[column] = df[column].astype('uint32')
            continue

        # Convert 'int64' type to 'int32'
        if dtype == 'int64':
            df[column] = df[column].astype('int32')
            continue

        # Convert 'float64' type to 'float32'
        if dtype == 'float64':
            df[column] = df[column].astype('float32')

    return df


def trainer(dataframe: pd.DataFrame, training_args: Dict[str, Any]):
    # init the model
    model = AutoML()

    # update the training args
    settings = dict(
        label = training_args['target'],
        task = training_args['task'],
        eval_method = 'cv',
        n_splits = 3,
        max_iter = training_args['iterations'],
        estimator_list = ['lgbm', 'extra_trees', 'xgboost'],
        n_jobs = int(os.getenv("THREADS", 4)),
        verbose = int(os.getenv("VERBOSE", 0)),
        seed = int(os.getenv("SEED", 42)),
        early_stop = True,
        sample = True
    )

    # train the model
    model.fit(dataframe=dataframe, **settings)

    return model


def evaluate_model(model: AutoML, dataframe: pd.DataFrame, training_args: Dict[str, Any]) -> Dict[str, float]:
    # generate a sample for evaluations
    dataframe = dataframe.sample(frac=0.25, random_state=42)

    # get y_true
    y_true = dataframe.loc[:, training_args['target']].values

    # get y_pred
    y_pred = model.predict(dataframe.drop(columns=[training_args['target']]))

    # get the evaluation results
    if training_args['task'] == Task.REGRESSION:
        return {
        "r2_score": r2_score(y_true, y_pred),
        "mean_squared_error": mean_squared_error(y_true, y_pred)
        }

    else:
        average = 'binary' if len(set(y_true)) == 2 else 'weighted'
        return {
            "accuracy_score": accuracy_score(y_true, y_pred),
            "precision_score": precision_score(y_true, y_pred, average=average),
            "recall_score": recall_score(y_true, y_pred, average=average),
            "f1_score": f1_score(y_true, y_pred, average=average)
        }