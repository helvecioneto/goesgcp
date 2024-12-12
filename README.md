# goesgcp

goesgcp is a utility script for downloading and reprojecting GOES-R satellite data. The script uses the `google.cloud` library to download data from the Google Cloud Platform (GCP) and the `pyproj` library to reproject the data to EPSG:4326 and crop it to a specified bounding box.


## Installation

You can install the necessary dependencies using `pip`:

```bash
pip install goesgcp
```

## Usage

### Command-Line Arguments

The script uses the `argparse` module for handling command-line arguments. Below are the available options:

```bash
goesgcp [OPTIONS]
```

| Option               | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| `--satellite`         | Name of the satellite (e.g., goes16).                                      |
| `--product`           | Name of the satellite product (e.g., ABI-L2-CMIPF). |
| `--var_name`          | Variable name to extract (e.g., CMI).                                       |
| `--channel`           | Channel to use (e.g., 13).                                                   |                      |
| `--between_minutes`   | Filter data between these minutes (default: `[0, 60]`).                    |
| `--output_path`       | Path for saving output files (default: `output/`).                           |                  |
| `--lat_min`           | Minimum latitude of the bounding box (default: `-56`).                      |
| `--lat_max`           | Maximum latitude of the bounding box (default: `35`).                       |
| `--lon_min`           | Minimum longitude of the bounding box (default: `-116`).                    |
| `--lon_max`           | Maximum longitude of the bounding box (default: `-25`).                     |
| `--max_attempts`      | Number of attempts to download a file before logging a failure (default: `3`).|

### Examples

To download and process recent data for the GOES-16 satellite, ABI-L2-CMIPF product, variable CMI, and channel 13, run the following command:

```bash
goesgcp --satellite goes16 --product ABI-L2-CMIPF --var_name CMI --channel 13 --recent 3 --output_path "output/"
```

### Credits
All the credit goes to the original author of the **goes2go** library.
And this is a otimization by Helvecio Neto - 2025
