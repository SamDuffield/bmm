
import setuptools

NAME = 'bmm'
DESCRIPTION = 'Bayesian Map-matching'

with open('README.md') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

METADATA = dict(
    name="bmm",
    version='1.3',
    url='http://github.com/SamDuffield/bmm',
    author='Sam Duffield',
    install_requires=install_requires,
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
