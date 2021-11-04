from cl import __author__, __author_email__, __version__
import setuptools


setuptools.setup(
    name='ConformalLayers',
    version=__version__,
    description='The implementation of the ConformalLayers: A non-linear sequential neural network with associative layers',
    author=__author__,
    author_email=__author_email__,
    url='https://github.com/Prograf-UFF/ConformalLayers/',
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        'torch>=1.8',
        'MinkowskiEngine>=0.5.4',
        'tqdm',  # Just for the experiments.
    ],
    zip_safe=False)
