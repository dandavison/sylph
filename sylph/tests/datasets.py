import random

import librosa
import numpy as np

from sylph.audio import Audio
from sylph.dataset import DataSet


class TwoPureTonesDataSet(DataSet):
    @classmethod
    def generate(cls, n, duration, **kwargs):
        random.seed(0)
        duration = 5
        sampling_rate = 44100
        middle_a_audio, high_a_audio = [
            Audio(librosa.audio.tone(freq, sr=sampling_rate, duration=duration), sampling_rate)
            for freq in [440, 440 * 2]
        ]
        n1 = int(n / 2)
        n2 = n - n1
        dataset = [(middle_a_audio, "A440")] * n1 + [(high_a_audio, "A880")] * n2
        random.shuffle(dataset)
        observations, labels = map(np.array, zip(*dataset))
        ids = np.arange(len(observations))
        return cls({"observations": observations, "ids": ids, "labels": labels}, **kwargs)
