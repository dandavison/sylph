from typing_extensions import Protocol

import numpy as np


class Classifier(Protocol):
    def predict(self, observations: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, observations: np.ndarray) -> np.ndarray:
        raise NotImplementedError
