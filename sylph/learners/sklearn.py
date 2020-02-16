from dataclasses import dataclass

import numpy as np

import sklearn.base

from sylph.classifier import Classifier
from sylph.dataset import Dataset
from sylph.pipeline import Learner
from sylph.utils.numpy import unpack_objects


@dataclass
class SklearnClassifier(Classifier):
    """
    A wrapper around a trained sklearn classifier, used for prediction.
    """

    model: sklearn.base.ClassifierMixin

    def predict(self, observations: np.ndarray) -> np.ndarray:
        return self.model.predict(unpack_objects(observations))


class SklearnClassifierLearner(Learner):
    """
    Takes a Dataset and outputs a SklearnClassifier.
    """

    model: sklearn.base.ClassifierMixin

    def __call__(self, dataset: Dataset) -> SklearnClassifier:
        X, y = dataset.observations, dataset.labels
        X, y = map(unpack_objects, (X, y))
        model = self.model.fit(X, y)
        return SklearnClassifier(model=model)
