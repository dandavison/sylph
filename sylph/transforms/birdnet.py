import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile
from dask import bag as dask_bag

from sylph.pipeline import Transform

PYTHON = "python3"
BIRDNET_DIR = "/"


@dataclass
class Audio2Embeddings(Transform):
    """
    Compute BirdNET embeddings for segments of each audio file.
    """

    def transform_observations(self, recordings: np.ndarray):
        real_n = len(recordings)
        recordings = recordings[:1]
        recordings_bag = dask_bag.from_sequence(recordings)
        embeddings = recordings_bag.map(get_segment_embeddings).compute()
        embeddings = list(embeddings) * real_n
        index_map = np.repeat(range(len(embeddings)), [len(segments) for segments in embeddings])
        embeddings = np.concatenate(embeddings)
        return embeddings, index_map


def get_segment_embeddings(recording):
    # This may be executing in a docker container on ECS.
    with TemporaryDirectory() as input_directory, TemporaryDirectory() as output_directory:
        input_directory = Path(input_directory)
        output_directory = Path(output_directory)

        path = input_directory / f"{recording.id}.wav"
        soundfile.write(
            path, recording.audio.time_series, recording.audio.sampling_rate, subtype="PCM_16"
        )
        os.chdir(BIRDNET_DIR)
        subprocess.check_call(
            [PYTHON, "analyze.py", "--i", input_directory, "--o", output_directory]
        )

        segment_embeddings = []
        for path in output_directory.glob("*-embedding.npz"):
            embedding = np.load(path)["embedding"]  # (1, 1024, 1, 3)
            embedding = np.concatenate(embedding.squeeze())
            segment_embeddings.append(embedding)

        return np.array(segment_embeddings)
