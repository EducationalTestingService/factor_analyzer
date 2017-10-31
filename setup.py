#!/usr/bin/env python

from setuptools import find_packages, setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='factor_analyzer',
      version=1.0,
      description='A Factor Analysis class',
      long_description=readme(),
      keywords='factor analysis',
      packages=find_packages(),
      include_package_data=True,
      entry_points={'console_scripts':
                    ['factor_analyzer = factor_analyzer.factor_analyzer:main']},
      classifiers=['Intended Audience :: Science/Research',
                   'Intended Audience :: Developers',
                   'Programming Language :: Python',
                   'License :: OSI Approved :: BSD License',
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
