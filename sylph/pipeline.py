"""
The interface of this module is intended to be compatible with torchvision.Transform and
torchvision.Compose.
"""
from collections import Counter
from dataclasses import dataclass
from typing import List
from typing import Optional
from typing import Tuple
from typing_extensions import Protocol

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sylph.classifier import Classifier
from sylph.dataset import DataSet
from sylph.utils.modal_values import get_modal_values


class Transform:
    """
    A Transform takes in one DataSet and outputs another.
    """

    def transform_observations(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return a 2-tuple:

        1. The transformed observations as a >=1-dimensional numpy array. Let n be the length of
           the first dimension of this array.

        2. A 1D numpy array of length n, representing a map from the row indices of the transformed
           data to the row indices of the input data. I.e. Let this array be named Z; if row j of
           the transformed output derives from row i of the input, then Z[j] = i.

        If the result of the transformation is the same length as the input, and the order of the
        rows is unchanged, then the second element of the tuple may be None (i.e. use the identity
        map).
        """
        raise NotImplementedError

    def __call__(self, dataset: DataSet) -> DataSet:
        observations, index_map = self.transform_observations(dataset.observations)
        if index_map is None:
            index_map = np.arange(len(observations))
        transformed_columns = {
            k: v[index_map] for k, v in dataset._columns.items() if k != "observations"
        }
        transformed_columns["observations"] = observations
        return DataSet(transformed_columns)


class Learner(Protocol):
    """
    A Learner takes in a DataSet and outputs a Classifier.
    """

    def __call__(self, dataset: DataSet) -> Classifier:
        raise NotImplementedError


class Compose:
    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms

    def __call__(self, dataset: DataSet) -> DataSet:
        for transform in self.transforms:
            dataset = transform(dataset)
        return dataset


@dataclass
class TrainingPipeline:
    transform: Transform
    learn: Learner
    output: Optional[dict] = None

    def run(self, dataset) -> dict:
        transformed_training_dataset = self.transform(dataset.training_dataset)
        classifier = self.learn(transformed_training_dataset)
        transformed_testing_dataset = self.transform(dataset.testing_dataset)
        transformed_testing_dataset_predictions = classifier.predict(
            transformed_testing_dataset.observations
        )
        return {
            "classifier": classifier,
            "transformed_testing_dataset": transformed_testing_dataset,
            "transformed_testing_dataset_predictions": transformed_testing_dataset_predictions,
            "transformed_training_dataset": transformed_training_dataset,
        }

    def get_metrics(self, dataset, output) -> dict:
        id2prediction = get_modal_values(
            group_ids=output["transformed_testing_dataset"]["ids"],
            values=output["transformed_testing_dataset_predictions"],
        )
        testing_dataset_predictions = np.array(
            [id2prediction[id] for id in dataset.testing_dataset["ids"]]
        )
        return {
            "testing_dataset_predictions": testing_dataset_predictions,
            "transformed_testing_accuracy": accuracy_score(
                output["transformed_testing_dataset"].labels,
                output["transformed_testing_dataset_predictions"],
            ),
            "testing_accuracy": accuracy_score(
                dataset.testing_dataset.labels, testing_dataset_predictions
            ),
            "testing_confusion_matrix": confusion_matrix(
                dataset.testing_dataset.labels, testing_dataset_predictions
            ),
        }
