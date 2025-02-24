import subprocess

# Upgrade pip and setuptools before installing dependencies
try:
    subprocess.run(["python", "-m", "ensurepip", "--upgrade"], check=True)
    subprocess.run(["python", "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run(["python", "-m", "pip", "install", "--upgrade", "setuptools"], check=True)
except subprocess.CalledProcessError:
    print("Error upgrading pip and setuptools.")

import os
import sys
import platform
from setuptools import setup, find_packages


def install_gdal():
    """Installs GDAL based on the operating system."""
    system = platform.system()

    try:
        if system == "Linux":
            print("Installing GDAL on Linux...")
            subprocess.run(
                ["python", "-m", "pip", "install", "gdal", "-f", "https://girder.github.io/large_image_wheels"],
                check=True
            )

        elif system == "Windows":
            python_version = sys.version_info
            if python_version[0] == 3:
                version_map = {
                    10: "cp310",
                    11: "cp311",
                    12: "cp312",
                    13: "cp313"
                }
                if python_version[1] in version_map:
                    gdal_url = f"https://github.com/cgohlke/geospatial-wheels/releases/download/v2025.1.20/GDAL-3.10.1-{version_map[python_version[1]]}-win_amd64.whl"
                    print(f"Downloading GDAL for Python {python_version[0]}.{python_version[1]}...")
                    subprocess.run(["python", "-m", "pip", "install", gdal_url], check=True)
                else:
                    print("Unsupported Python version for GDAL.")
                    sys.exit(1)
            else:
                print("Unsupported Python version for GDAL.")
                sys.exit(1)

        elif system == "Darwin":
            print("GDAL is not available for macOS via this installer.")
            sys.exit(1)

        else:
            print("Unsupported operating system.")
            sys.exit(1)
    
    except subprocess.CalledProcessError:
        print("Error installing GDAL.")

# Install GDAL before running setup
install_gdal()

# Read dependencies from requirements.txt
req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
if os.path.exists(req_file):
    with open(req_file) as f:
        requirements = f.read().splitlines()
else:
    requirements = []

setup(
    name="goesgcp",
    version="3.0.0",
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
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: GIS",
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
