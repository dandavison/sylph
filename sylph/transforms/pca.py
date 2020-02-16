import sklearn.decomposition

import numpy as np

from sylph.pipeline import Transform
from sylph.utils.numpy import unpack_objects


class PCA(Transform):
    def __init__(self, **kwargs):
        self.pca = sklearn.decomposition.PCA(**kwargs)

    def transform_observations(self, observations: np.ndarray):
        X = unpack_objects(observations)
        return self.pca.fit(X).transform(X), None
