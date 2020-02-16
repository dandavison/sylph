import numpy as np

from sylph.audio import Audio
from sylph.dataset import DataSet


class AudioDataSet(DataSet):
    @classmethod
    def from_files(cls, paths, labels, **kwargs):
        assert len(paths) == len(labels)
        observations = np.array([Audio.from_file(path) for path in paths])
        labels = labels
        return cls(
            {"observations": observations, "labels": labels, "ids": np.array(paths)}, **kwargs
        )
