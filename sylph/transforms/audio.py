from dataclasses import dataclass

import numpy as np

from sylph.audio import Audio
from sylph.dataset import Dataset


@dataclass
class Audio2Audio16Bit:
    """
    Convert audio to 16 bit; optionally normalize to be in [-1, 1).
    """

    normalize_amplitude: bool = False

    def __call__(self, dataset: Dataset) -> Dataset:
        observations = []
        for audio in dataset.observations:
            audio = audio.as_16_bit_pcm()
            if self.normalize_amplitude:
                audio.time_series = audio.time_series / (2 ** 15)
            observations.append(audio)
        return Dataset(observations=np.array(observations), labels=np.array(dataset.labels))
