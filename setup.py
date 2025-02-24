
import os

try:
    os.system("python -m ensurepip --upgrade")
    os.system("python -m pip install --upgrade pip")
    os.system("python -m pip install --upgrade setuptools")
except:
    pass

from setuptools import setup, find_packages
import subprocess
import sys
import platform

def install_gdal():
    system = platform.system()

    if system == "Linux":
        print("Instalando GDAL no Linux...")
        subprocess.run(
            "pip install gdal -f https://girder.github.io/large_image_wheels", 
            shell=True, check=True
        )

    elif system == "Windows":
        python_version = sys.version_info
        if python_version[0] == 3:
            if python_version[1] == 10:
                gdal_url = "https://github.com/cgohlke/geospatial-wheels/releases/download/v2025.1.20/GDAL-3.10.1-cp310-cp310-win_amd64.whl"
            elif python_version[1] == 11:
                gdal_url = "https://github.com/cgohlke/geospatial-wheels/releases/download/v2025.1.20/GDAL-3.10.1-cp311-cp311-win_amd64.whl"
            elif python_version[1] == 12:
                gdal_url = "https://github.com/cgohlke/geospatial-wheels/releases/download/v2025.1.20/GDAL-3.10.1-cp312-cp312-win_amd64.whl"
            elif python_version[1] == 13:
                gdal_url = "https://github.com/cgohlke/geospatial-wheels/releases/download/v2025.1.20/GDAL-3.10.1-cp313-cp313-win_amd64.whl"
            else:
                print("Python version not supported for GDAL download.")
                sys.exit(1)

            print(f"Downloading Gdal (Python {python_version[0]}.{python_version[1]})...")
            subprocess.run(f"python3 -m pip install {gdal_url}", shell=True, check=True)

        else:
            print("Python version not supported for GDAL download.")
            sys.exit(1)

    elif system == "Darwin":
        print("The GDAL package is not available for macOS.")
        sys.exit(1)

    else:
        print("Operating system not supported.")
        sys.exit(1)

# Install GDAL
install_gdal()

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
