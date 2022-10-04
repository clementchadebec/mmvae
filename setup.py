from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="bivae",
    version="0.0.1",
    author="Agathe Senellart (HekA team INRIA)",
    author_email="agathe.senellart@inria.fr",
    description="Bimodal variational Autoencoders in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AgatheSenellart/mmvae",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.6",
)
