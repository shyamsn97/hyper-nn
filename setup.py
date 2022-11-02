import io
import os
import re
from os import path

from setuptools import find_packages
from setuptools import setup


this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name="hyper-nn",
    packages=find_packages(exclude=('tests',)),
    version="0.2.0",
    url="https://github.com/shyamsn97/hyper-nn",
    license='MIT',

    author="Shyam Sudhakaran",
    author_email="shyamsnair@protonmail.com",

    description="Easy hypernetworks in Pytorch and Flax",

    long_description=long_description,
    long_description_content_type="text/markdown",

    install_requires=[
        'numpy',
        'torch',
        'functorch',
        'flax',
        'jax',
    ],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
