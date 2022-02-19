import os
import pkg_resources, setuptools


base_dir = os.path.dirname(os.path.abspath(__file__))

# Set some variables related to the version of the package.
with open(os.path.join(base_dir, 'cl', 'about.py'), 'r') as about:
    exec(about.read())

# Read dependencies.
with open(os.path.join(base_dir, 'requirements.txt'), 'r') as fin:
    requirements = [*map(str, pkg_resources.parse_requirements(fin.readlines()))]

# Setup the package.
setuptools.setup(
    name='ConformalLayers',
    version=__version__,
    description=__description__,
    author=__author__,
    author_email=__author_email__,
    url=__url__,
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
)
