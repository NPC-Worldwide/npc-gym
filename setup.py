"""Setup for npc-gym package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="npc-gym",
    version="0.1.0",
    author="Christopher Agostino",
    author_email="",
    description="Gymnasium-style framework for training hybrid LLM+ML agents through games",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cagostino/npc-gym",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.20",
        "npcpy>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
            "black",
            "isort",
            "mypy",
        ],
        "viz": [
            "flask>=2.0",
            "plotly>=5.0",
        ],
        "full": [
            "torch>=2.0",
            "transformers>=4.30",
            "datasets>=2.0",
            "trl>=0.7",
            "peft>=0.5",
        ],
    },
)
