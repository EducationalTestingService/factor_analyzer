#!/usr/bin/env python

from setuptools import find_packages, setup


def readme():
    with open('README.rst') as f:
        return f.read()


def requirements():
    req_path = 'requirements.txt'
    with open(req_path) as f:
        reqs = f.read().splitlines()
    return reqs


setup(name='factor_analyzer',
      version='0.2.3',
      description='A Factor Analysis class',
      long_description=readme(),
      keywords='factor analysis',
      packages=find_packages(),
      author="Jeremy Biggs",
      author_email="jbiggs@ets.org",
      url="https://github.com/EducationalTestingService/factor_analyzer",
      install_requires=requirements(),
      include_package_data=True,
      entry_points={'console_scripts':
                    ['factor_analyzer = factor_analyzer.analyze:main']},
      classifiers=['Intended Audience :: Science/Research',
                   'Intended Audience :: Developers',
                   'Programming Language :: Python',
                   'Topic :: Scientific/Engineering',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX',
                   'Operating System :: Unix',
                   'Operating System :: MacOS',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.4',
                   'Programming Language :: Python :: 3.5',
                   'Programming Language :: Python :: 3.6',
                   ],
      zip_safe=True)
