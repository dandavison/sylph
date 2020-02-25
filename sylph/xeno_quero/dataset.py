from itertools import chain

import numpy as np

from sylph.dataset import DataSet
from sylph.xeno_quero import get_download_directory
from sylph.xeno_quero.recording import XenoQueroRecording
from sylph.xeno_quero.species import XenoQueroSpecies

from clint.textui import colored

red = lambda s: colored.red(s, bold=True)


class XenoQueroDataset(DataSet):
    """
    A DataSet of XenoQueroRecordings.
    """

    @classmethod
    def from_species(cls, names, **kwargs):
        species = [XenoQueroSpecies(name) for name in names]
        return cls._from_species_objects(species, **kwargs)

    @classmethod
    def from_species_globs(cls, globs, **kwargs):
        species_dirs = list(
            chain.from_iterable(get_download_directory().glob(glob) for glob in globs)
        )
        print(red(f"Got {len(species_dirs)} species dirs for globs: {globs}"))
        species = [XenoQueroSpecies(dir.name) for dir in species_dirs]
        self = cls._from_species_objects(species, **kwargs)
        print(red(f"Got {self.n} row dataset with {len(set(self.labels))} species"))
        return self

    @classmethod
    def _from_species_objects(cls, species, **kwargs):
        species_audio_paths = [sp.audio_paths for sp in species]
        paths = list(chain.from_iterable(species_audio_paths))
        labels = np.repeat(
            [sp.name for sp in species], [len(paths) for paths in species_audio_paths]
        )
        observations = np.array([XenoQueroRecording.from_path(p) for p in paths])
        return cls(
            {"observations": observations, "labels": labels, "ids": np.array(paths)}, **kwargs
        )
