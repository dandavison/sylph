from dataclasses import dataclass

import numpy as np

import sklearn.base

from sylph.classifier import Classifier
from sylph.dataset import DataSet
from sylph.pipeline import Learner
from sylph.utils.color import red
from sylph.utils.numpy import unpack_objects


@dataclass
class SklearnClassifier(Classifier):
    """
    A wrapper around a trained sklearn classifier, used for prediction.
    """

    model: sklearn.base.ClassifierMixin

    def predict(self, observations: np.ndarray) -> np.ndarray:
        return self.model.predict(unpack_objects(observations))

    def predict_proba(self, observations: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(unpack_objects(observations))


class SklearnClassifierLearner(Learner):
    """
    Takes a DataSet and outputs a SklearnClassifier.
    """

    model: sklearn.base.ClassifierMixin
    classifier_cls = SklearnClassifier

    def __call__(self, dataset: DataSet) -> SklearnClassifier:
        print(red(f"Running {type(self).__name__} on {dataset}"))
        X, y = dataset.observations, dataset.labels
        X, y = map(unpack_objects, (X, y))
        model = self.model.fit(X, y)
        return self.classifier_cls(model=model)
