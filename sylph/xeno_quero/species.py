from dataclasses import dataclass

from sylph.xeno_quero import get_download_directory


@dataclass
class XenoQueroSpecies:
    name: str  # f"{genus} {species}"

    @property
    def audio_paths(self):
        return list(self.dir.glob("*.mp3"))

    @property
    def dir(self):
        return get_download_directory() / self.name.replace(" ", "-")
