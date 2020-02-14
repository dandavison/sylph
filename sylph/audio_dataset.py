import numpy as np

from sylph.audio import Audio
from sylph.dataset import Dataset


class AudioDataset(Dataset):
    def __init__(self, paths, labels, **kwargs):
        assert len(paths) == len(labels)
        observations = np.array([Audio.from_file(path) for path in paths])
        labels = labels
        super().__init__(observations, labels, **kwargs)
