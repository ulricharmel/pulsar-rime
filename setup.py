import os
from setuptools import setup, find_packages
from rime import __version__

build_root = os.path.dirname(__file__)

def requirements():
    """Get package requirements"""
    with open(os.path.join(build_root, 'requirements.txt')) as f:
        return [pname.strip() for pname in f.readlines()]


with open("README.rst") as tmp:
    readme = tmp.read()

console_scripts = ['pulsar-rime=rime.main:main', 'pulsar-beam=rime.beam_variation:main']

setup(
    author='Ulrich A. Mbou Sob',
    author_email='mulricharmel@gmail.com',
    name='pulsar-rime',
    version=__version__,
    description='Computing visibilities for variable source with a beamgain using jax',
    long_description=readme,
    long_description_content_type="text/x-rst",
    url='https://github.com/ulricharmel/pulsar-rime',
    license='GNU GPL v2',
    install_requires=requirements(),
    packages=find_packages(include=['pulsar-rime','pulsar-rime.*', 'pulsar-beam', 'pulsar-beam.*']),
    entry_points={
        'console_scripts': console_scripts
    },
    keywords='pulsar',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Programming Language :: Python :: 3.9',
        ],
)