"""setup.py — kept for editable install compatibility."""
from setuptools import setup, find_packages

setup(
    name="edge-tpu-v6-bench",
    version="0.1.0",
    packages=find_packages(exclude=["tests*"]),
    install_requires=["numpy"],
    python_requires=">=3.8",
)
