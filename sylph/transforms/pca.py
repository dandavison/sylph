import sklearn.decomposition

from sylph.dataset import Dataset
from sylph.utils.numpy import unpack_objects


class PCA:
    def __init__(self, **kwargs):
        self.pca = sklearn.decomposition.PCA(**kwargs)

    def __call__(self, dataset: Dataset) -> Dataset:
        X = unpack_objects(dataset.observations)
        X = self.pca.fit(X).transform(X)
        new_dataset = Dataset(observations=X, labels=dataset.labels)
        new_dataset.n_examples_per_audio = dataset.n_examples_per_audio
        return new_dataset
