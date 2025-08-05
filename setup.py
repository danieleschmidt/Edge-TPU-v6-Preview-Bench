#!/usr/bin/env python3
"""
Edge TPU v6 Preview Benchmark Suite
Setup configuration for installation and development
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Edge TPU v6 Preview Benchmark Suite"

# Core dependencies
INSTALL_REQUIRES = [
    'tensorflow>=2.14.0',
    'numpy>=1.21.0',
    'tflite-runtime>=2.14.0',
    'pycoral>=2.0.0',
    'matplotlib>=3.5.0',
    'seaborn>=0.11.0',
    'pandas>=1.3.0',
    'scipy>=1.7.0',
    'psutil>=5.8.0',
    'click>=8.0.0',
    'tqdm>=4.62.0',
    'pyyaml>=6.0',
    'requests>=2.28.0',
    'pillow>=9.0.0',
    'opencv-python>=4.5.0',
]

# Optional dependencies for different features
EXTRAS_REQUIRE = {
    'dev': [
        'pytest>=7.0.0',
        'pytest-cov>=4.0.0',
        'black>=22.0.0',
        'flake8>=5.0.0',
        'mypy>=0.991',
        'pre-commit>=2.20.0',
        'sphinx>=5.0.0',
        'sphinx-rtd-theme>=1.0.0',
    ],
    'models': [
        'transformers>=4.20.0',
        'torch>=1.12.0',
        'torchvision>=0.13.0',
        'timm>=0.6.0',
        'ultralytics>=8.0.0',
    ],
    'analysis': [
        'plotly>=5.10.0',
        'dash>=2.6.0',
        'jupyter>=1.0.0',
        'ipywidgets>=7.7.0',
        'bokeh>=2.4.0',
    ],
    'power': [
        'pyserial>=3.5',
        'smbus2>=0.4.0',
        'adafruit-circuitpython-ina260>=1.2.0',
    ]
}

# All extras combined
EXTRAS_REQUIRE['all'] = list(set(sum(EXTRAS_REQUIRE.values(), [])))

setup(
    name='edge-tpu-v6-bench',
    version='0.1.0',
    description='Future-ready benchmark harness for Google Edge TPU v6',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    author='Daniel Schmidt',
    author_email='daniel@terragonlabs.com',
    url='https://github.com/danieleschmidt/Edge-TPU-v6-Preview-Bench',
    license='Apache 2.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.8',
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points={
        'console_scripts': [
            'edge-tpu-v6-bench=edge_tpu_v6_bench.cli:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: System :: Benchmark',
        'Topic :: System :: Hardware',
    ],
    keywords='edge-tpu benchmark ai ml tensorflow quantization coral',
    project_urls={
        'Documentation': 'https://edge-tpu-v6-bench.readthedocs.io',
        'Source': 'https://github.com/danieleschmidt/Edge-TPU-v6-Preview-Bench',
        'Tracker': 'https://github.com/danieleschmidt/Edge-TPU-v6-Preview-Bench/issues',
    },
)