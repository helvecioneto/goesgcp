import os

try:
    os.system("python -m ensurepip --upgrade")
    os.system("python -m pip install --upgrade setuptools")
except:
    pass

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="goesgcp",
    version='1.0.7',
    author="Helvecio B. L. Neto",
    author_email="helvecioblneto@gmail.com",
    description="A package to download and process GOES-16/17 data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/helvecioneto/goesgcp",
    packages=find_packages(),
    install_requires=requirements,
    license="LICENSE",
    classifiers=[
        "Programming Language :: Python",
        "Development Status :: 5 - Production/Stable",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
        "Topic :: Utilities",
    ],
    entry_points={
        'console_scripts': [
            'goesgcp=goesgcp.main:main',
        ],
    },
)
