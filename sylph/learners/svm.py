import numpy as np

import sklearn.svm
from sklearn.utils.extmath import softmax

from sylph.learners.sklearn import SklearnClassifier
from sylph.learners.sklearn import SklearnClassifierLearner
from sylph.utils.numpy import unpack_objects


class SVMClassifier(SklearnClassifier):
    def predict_proba(self, observations: np.ndarray) -> np.ndarray:
        # https://scikit-learn.org/dev/modules/svm.html#scores-and-probabilities
        # https://stackoverflow.com/questions/49507066/predict-probabilities-using-svm
        return softmax(self.model.decision_function(unpack_objects(observations)))


class SVMLearner(SklearnClassifierLearner):
    model = sklearn.svm.SVC()
    classifier_cls = SVMClassifier
