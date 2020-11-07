
import setuptools

NAME = 'bmm'
DESCRIPTION = 'Bayesian Map-matching'

with open('README.md') as f:
    long_description = f.read()

METADATA = dict(
    name="bmm",
    version='0.1',
    url='http://github.com/SamDuffield/bmm',
    author='Sam Duffield',
    install_requires=['numpy',
                      'matplotlib',
                      'scipy',
                      'numba',
                      'pandas',
                      'geopandas',
                      'osmnx',
                      'networkx'
                      ],
    author_email='sddd2@cam.ac.uk',
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_namespace_packages(),
    include_package_data=True,
    platforms='any',
    classifiers=[
        "Programming Language :: Python :: 3.8",
    ]
)

setuptools.setup(**METADATA)
