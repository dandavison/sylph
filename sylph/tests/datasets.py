import numpy as np
import random

import librosa

from sylph.audio import Audio
from sylph.dataset import Dataset


class TwoPureTonesDataset(Dataset):
    def __init__(self, n, duration, **kwargs):
        random.seed(0)
        duration = 5
        sampling_rate = 44100
        middle_a_audio, high_a_audio = [
            Audio(librosa.audio.tone(freq, sr=sampling_rate, duration=duration), sampling_rate)
            for freq in [440, 440 * 2]
        ]
        n1 = int(n / 2)
        n2 = n - n1
        dataset = [(middle_a_audio, "middle_a")] * n1 + [(high_a_audio, "high_a")] * n2
        random.shuffle(dataset)
        observations, labels = map(np.array, zip(*dataset))
        super().__init__(observations, labels, **kwargs)
