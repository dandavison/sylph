from pathlib import Path


def get_egg_path():
    dir = Path(__file__).resolve().parents[2] / "dist"
    try:
        [path] = dir.glob("sylph-*.egg")
    except ValueError:
        raise AssertionError(f"Failed to find egg file in directory: {dir}")
    else:
        return path
