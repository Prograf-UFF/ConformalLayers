import setuptools


setuptools.setup(
    name='cl',
    version='1.0.1',
    description='ConformalLayers: A non-linear sequential neural network with associative layers',
    author='Eduardo V. Sousa, Leandro A. F. Fernandes',
    author_email='eduardovera@ic.uff.br, laffernandes@ic.uff.br',
    url='https://github.com/Prograf-UFF/ConformalLayers/',
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        'torch>=1.8',
        'MinkowskiEngine==0.4.3',
        'tqdm',  # Just for the experiments and tests.
    ],
    zip_safe=False)
