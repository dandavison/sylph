import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import numpy as np
from toolz import groupby

from sylph.dataset import DataSet
from sylph.pipeline import Transform


@dataclass
class Audio2Embeddings(Transform):
    """
    Compute BirdNET embeddings for segments of each audio file.
    """

    # TOPO: the variables named audio can be XenoCantoRecordings

    birdnet_output_dir: Optional[Path]
    filetype: str = "mp3"

    def __call__(self, dataset: DataSet) -> DataSet:
        # `transform_observations` needs access to the IDs because it is handling the case in which
        # BirdNET has already been run. In that case, the output files are on disk, and we need to
        # ensure that we associate them correctly with the dataset rows.
        self.dataset_ids = dataset.ids
        return super().__call__(dataset)

    def transform_observations(self, audios: np.ndarray):
        if self.birdnet_output_dir:
            return self._get_embeddings_and_index_map(audios, self.birdnet_output_dir)
        else:
            # with TemporaryDirectory() as birdnet_output_dir:

            birdnet_output_dir = "~/tmp/elaenia/birdnet_embeddings_Xiphorhynchus"

            birdnet_output_dir = Path(birdnet_output_dir).expanduser()
            self._write_audio_files(audios, birdnet_output_dir)
            self._run_birdnet(birdnet_output_dir)
            return self._get_embeddings_and_index_map(audios, birdnet_output_dir)

    def _write_audio_files(self, audios, birdnet_output_dir: Path):
        for audio in audios:
            path = birdnet_output_dir / f"{audio.id}.mp3"
            if not path.exists():
                os.link(audio.get_recording_path().absolute(), path.absolute())

    def _run_birdnet(self, birdnet_output_dir: Path):
        return subprocess.check_output(
            [
                "nice",
                "-n",
                "20",
                "docker",
                "run",
                "-v",
                f"{birdnet_output_dir}:/audio",
                "birdnet",
                "--filetype",
                "mp3",
                "--i",
                "audio",
            ]
        )

    def _get_embeddings_and_index_map(self, audios: np.ndarray, birdnet_output_dir: Path):
        # File names look like 478727-8-embedding.npz
        id2paths = groupby(
            lambda path: int(path.name.split("-")[0]), birdnet_output_dir.glob("*-embedding.npz")
        )

        embeddings = []
        index_map = []

        for i, audio_path in enumerate(self.dataset_ids):
            xc_id = int(audio_path.name.split(".")[0])
            for path in id2paths[xc_id]:
                embedding = np.load(path)["arr_0"]  # (1, 1024, 1, 3)
                embedding = np.concatenate(embedding.squeeze())
                embeddings.append(embedding)
                index_map.append(i)

        return np.array(embeddings), np.array(index_map)
