from dataclasses import dataclass
from tempfile import NamedTemporaryFile

import numpy as np
import soundfile

from sylph.utils import librosa as librosa_utils


@dataclass
class Audio:
    time_series: np.ndarray
    sampling_rate: float

    @classmethod
    def from_file(cls, path):
        ts, sr = librosa_utils.load(path)
        return cls(time_series=ts, sampling_rate=sr)

    def as_16_bit_pcm(self):
        with NamedTemporaryFile(suffix=".wav") as fp:
            path = fp.name
            soundfile.write(path, self.time_series, self.sampling_rate, subtype="PCM_16")
            ts, sr = soundfile.read(path, dtype="int16")
        assert ts.dtype == "int16"
        return type(self)(time_series=ts, sampling_rate=sr)

    def __repr__(self):
        _type = type(self).__name__
        duration = len(self.time_series) / self.sampling_rate
        return f"{_type}(duration={duration}, sampling_rate={self.sampling_rate})"

    def __iter__(self):
        return iter(self.time_series)
