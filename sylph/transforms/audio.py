from dataclasses import dataclass

import numpy as np

from sylph.audio import Audio
from sylph.dataset import DataSet
from sylph.pipeline import Transform


@dataclass
class Audio2Audio16Bit(Transform):
    """
    Convert audio to 16 bit; optionally normalize to be in [-1, 1).
    """

    normalize_amplitude: bool = False

    def transform_observations(self, audios: np.ndarray):
        new_audios = []
        for audio in audios:
            audio = audio.as_16_bit_pcm()
            if self.normalize_amplitude:
                audio.time_series = audio.time_series / (2 ** 15)
            new_audios.append(audio)
        return np.array(new_audios), None
