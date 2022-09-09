#!/usr/bin/env python
"""Set up the package."""

from setuptools import find_packages, setup


def readme():  # noqa: D103
    with open("README.rst") as f:
        return f.read()


def requirements():  # noqa: D103
    req_path = "requirements.txt"
    with open(req_path) as f:
        reqs = f.read().splitlines()
    return reqs


setup(
    name="factor_analyzer",
    version="0.4.1",
    description="A Factor Analysis tool written in Python",
    long_description=readme(),
    keywords="factor analysis",
    packages=find_packages(),
    author="Jeremy Biggs",
    author_email="jeremy.m.biggs@gmail.com",
    maintainer="Nitin Madnani",
    maintainer_email="nmadnani@ets.org",
    url="https://github.com/EducationalTestingService/factor_analyzer",
    install_requires=requirements(),
    include_package_data=True,
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    zip_safe=True,
)
