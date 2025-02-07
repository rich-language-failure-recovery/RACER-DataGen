from setuptools import setup, find_packages

requirements = [
    "cffi",
    "pip==24.0",
    "omegaconf==2.0.6",
]

__version__ = "0.0.1"
setup(
    name="racer_datagen",
    version=__version__,
    description="RACER Data Generation",
    long_description="",
    author="Jayjun Lee",
    author_email="jayjun@umich.edu",
    url="",
    keywords="robotics,language",
    packages=['racer_datagen'],
    install_requires=requirements,
)