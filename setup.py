import setuptools

from neural_tree.__version__ import __version__

with open(file="README.md", mode="r", encoding="utf-8") as fh:
    long_description = fh.read()

base_packages = [
    "torch >= 1.13",
    "tqdm >= 4.66",
    "transformers >= 4.34.0",
    "sentence-transformers >= 2.2.2",
    "neural-cherche >= 1.1.0",
    "scikit-learn >= 1.4.0",
]

eval = ["ranx >= 0.3.16", "beir >= 2.0.0"]

dev = ["mkdocs-material == 9.2.8"]

setuptools.setup(
    name="neural-tree",
    version=f"{__version__}",
    license="MIT",
    author="Raphael Sourty",
    author_email="raphael.sourty@gmail.com",
    description="Neural-Tree",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/raphaelsty/neural-tree",
    download_url="https://github.com/user/neural-tree/archive/v_01.tar.gz",
    keywords=[
        "tree search",
        "neural search",
        "information retrieval",
        "semantic search",
        "colbert",
        "tree",
    ],
    packages=setuptools.find_packages(),
    install_requires=base_packages,
    extras_require={"eval": base_packages + eval, "dev": base_packages + eval + dev},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
