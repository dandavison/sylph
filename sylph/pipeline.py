"""
The interface of this module is intended to be compatible with torchvision.Transform and
torchvision.Compose.
"""
from dataclasses import dataclass
from typing import List
from typing import Optional
from typing_extensions import Protocol

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sylph.classifier import Classifier
from sylph.dataset import Dataset
from sylph.utils import get_group_modes


class Transform(Protocol):
    """
    A Transform takes in one Dataset and outputs another.
    """

    def __call__(self, dataset: Dataset) -> Dataset:
        raise NotImplementedError


class Learner(Protocol):
    """
    A Learner takes in a Dataset and outputs a Classifier.
    """

    def __call__(self, dataset: Dataset) -> Classifier:
        raise NotImplementedError


class Compose:
    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms

    def __call__(self, dataset: Dataset) -> Dataset:
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
        testing_dataset_predictions = np.array(
            get_group_modes(
                output["transformed_testing_dataset_predictions"],
                output["transformed_testing_dataset"].n_examples_per_audio,
            )
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
