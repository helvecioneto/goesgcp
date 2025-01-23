# goesgcp
<!-- badges: start -->
[![pypi](https://badge.fury.io/py/goesgcp.svg)](https://pypi.python.org/pypi/goesgcp)
[![Downloads](https://img.shields.io/pypi/dm/goesgcp.svg)](https://pypi.python.org/pypi/goesgcp)
[![Contributors](https://img.shields.io/github/contributors/helvecioneto/goesgcp.svg)](https://github.com/helvecioneto/goesgcp/graphs/contributors)
[![License](https://img.shields.io/pypi/l/goesgcp.svg)](https://github.com/helvecioneto/goesgcp/blob/main/LICENSE)
<!-- badges: end -->


`goesgcp` is a Python utility designed for downloading and reprojecting GOES-R satellite data. This script leverages the `google.cloud` library for accessing data from the Google Cloud Platform (GCP) and `rioxarray` for reprojecting data to EPSG:4326 (rectangular grid), as well cropping it to a user-defined bounding box.

## Features

- **Download GOES-R satellite data**: Supports GOES-16 and GOES-18.
- **Reprojection and cropping**: Reprojects data to EPSG:4326 and crops to a specified bounding box.
- **Flexible command-line interface**: Customize download options, variables, channels, time range, and output format.
- **Efficient processing**: Handles large datasets with optimized performance.

## Installation

Install the package via `pip`:

```bash
pip install goesgcp
```


## Usage

### Available Command-Line Arguments

The script uses the `argparse` module for handling command-line arguments. Below are the available options:

```bash
goesgcp [OPTIONS]
```

| Option               | Description                                                                |
|----------------------|----------------------------------------------------------------------------|
| `--satellite`         | Name of the satellite (e.g., goes16).                                     |
| `--product`           | Name of the satellite product (e.g., ABI-L2-CMIPF).                       |
| `--var_name`          | Variable name to extract (e.g., CMI).                                     |
| `--channel`           | Channel to use (e.g., 13).                                                |
| `--output`            | Path for saving output files (default: `output/`).                        | 
| `--lat_min`           | Minimum latitude of the bounding box (default: `-56`).                    |
| `--lat_max`           | Maximum latitude of the bounding box (default: `35`).                     |
| `--lon_min`           | Minimum longitude of the bounding box (default: `-116`).                  |
| `--lon_max`           | Maximum longitude of the bounding box (default: `-25`).                   |
| `--resolution`        | Set the reprojet data resolution in degree (default: `-0.045`).           |
| `--recent`            | Number of most recent data to download (default: `1`).                    |
| `--start`             | Start date for downloading data (default: `None`).                        |
| `--end`               | End date for downloading data (default: `None`).                          |
| `--bt_hour`           | Hour of the day to download data (default: [0, 1, ..., 23]).              |
| `--bt_minute`         | Minute of the hour to download data (default: [0, 15, 30, 45]).           |
| `--save_format`       | Format for saving output files (default: `by_date`).                      |

#### Available GOES Products
A comprehensive list of available GOES products can be found at the following link: [https://console.cloud.google.com/storage/browser/gcp-public-data-goes-16](https://console.cloud.google.com/storage/browser/gcp-public-data-goes-16)

### Examples

#### Download Recent Data
In the example below, the command downloads the 3 most recent files from the GOES-16 satellite for the product ABI-L2-CMIPF. It focuses on the variable CMI (Cloud and Moisture Imagery) from channel 13, which is commonly used for infrared observations. The downloaded files are saved to the specified output directory output/.

```bash
goesgcp --satellite goes-16 --recent 3 --product ABI-L2-CMIPF --var_name CMI --channel 13 --output "output/"
```

#### Download Data for a Specific Time Range
This command retrieves GOES-16 satellite data for the product ABI-L2-CMIPF within the date range 2022-12-15 00:00:00 to 2022-12-20 10:00:00, focusing on hours 5:00 and 6:00 AM. The data is cropped to the geographic bounds of -35° to 5° latitude and -80° to -30° longitude, reprojected with a resolution of 0.045 degrees, and saved in a by_date format for easy organization.

```bash
goesgcp --satellite goes-16 --product ABI-L2-CMIPF --start '2022-12-15 00:00:00' --end '2022-12-20 10:00:00' --bt_hour 5 6 --save_format by_date --resolution 0.045 --lat_min -35 --lat_max 5 --lon_min -80 --lon_max -30
```

### Contributing
Contributions are welcome! If you encounter issues or have suggestions for improvements, please submit them via GitHub issues or pull requests.

### Credits
This project was developed and optimized by Helvecio Neto (2025).
It builds upon NOAA GOES-R data and leverages resources provided by the Google Cloud Platform.

### License
This project is licensed under the MIT License. 
