from itertools import chain

import numpy as np

from sylph.dataset import DataSet
from sylph.xeno_quero.recording import XenoQueroRecording
from sylph.xeno_quero.species import XenoQueroSpecies


class XenoQueroDataset(DataSet):
    """
    A DataSet of XenoQueroRecordings.
    """

    @classmethod
    def from_species(cls, names, **kwargs):
        species = [XenoQueroSpecies(name) for name in names]
        species_paths = [sp.audio_paths for sp in species]
        paths = list(chain.from_iterable(species_paths))
        labels = np.repeat(names, [len(paths) for paths in species_paths])
        observations = np.array([XenoQueroRecording.from_path(p) for p in paths])
        return cls(
            {"observations": observations, "labels": labels, "ids": np.array(paths)}, **kwargs
        )
