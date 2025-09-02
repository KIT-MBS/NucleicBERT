from pathlib import Path
from setuptools import find_namespace_packages, setup, find_packages

# Read the contents of requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='nucleicbert',
    version='0.1.0-alpha',
    author='NucleicBERT Team',
    author_email='nucleicbert@example.com',
    description='Language model for RNA sequences',
    long_description=open('docs/README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'pretrain = pretrain.train:main',
            'downstream = downstream.downstream_train:main',
        ],
    },
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX',  # For Unix-based systems
        'Operating System :: POSIX :: Linux',  # Specifically for Linux
        'Operating System :: MacOS :: MacOS X',  # Specifically for macOS
        'Operating System :: Unix',  # Generic Unix OS
        'Programming Language :: Python :: 3.12',
    ],
    python_requires='>=3.6',
    install_requires=requirements,  # Automatically load dependencies from requirements.txt
)
