from setuptools import find_packages
from setuptools import setup


setup(
    name="sylph",
    author="Dan Davison",
    author_email="dandavison7@gmail.com",
    description="Machine learning pipelines for audio classification",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        "black",
        "cached-property",
        "flake8",
        "librosa",
        "mypy",
        "numpy",
        "pytest",
        "scikit-learn",
        "sounddevice",
        "tensorflow==1.15",
        "tf_slim",
        "tox",
    ],
)
