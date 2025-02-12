
import os

try:
    os.system("python -m ensurepip --upgrade")
    os.system("python -m pip install --upgrade setuptools")
except:
    pass

from setuptools import setup, find_packages

req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
if os.path.exists(req_file):
    with open(req_file) as f:
        requirements = f.read().splitlines()
else:
    requirements = []

setup(
    name="goesgcp",
    version="2.0.8",
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
        "Topic :: Scientific/Engineering :: Remote Sensing",
    ]
)
