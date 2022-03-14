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
    name="hypernn",
    version="0.1.0",
    url="https://github.com/shyamsn97/hyper-nn",
    license='MIT',

    author="Shyam Sudhakaran",
    author_email="shyamsnair@protonmail.com",

    description="Easy hypernetworks in Jax and Pytorch",

    long_description=long_description,
    long_description_content_type="text/markdown",

    packages=find_packages(exclude=('tests',)),

    install_requires=[
        # 'functorch==0.1.0',
        'torch==1.11.0',
        'flax',
        'jax',
        'einops'
    ],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
)
