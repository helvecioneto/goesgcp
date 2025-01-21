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

All GOES products are avaiable in: [https://github.com/awslabs/open-data-docs/tree/main/docs/noaa/noaa-goes16](https://github.com/awslabs/open-data-docs/tree/main/docs/noaa/noaa-goes16)

### Examples

To download most 3 recent data for the GOES-16 satellite, ABI-L2-CMIPF product, variable CMI, and channel 13, run the following command:

```bash
goesgcp --satellite goes16 --product ABI-L2-CMIPF --var_name CMI --channel 13 --recent 3 --output "output/"
```

To download data for a specific date range, use the `--start` and `--end` options:

```bash
goesgcp --start '2022-12-15 00:00:00' --end '2022-12-20 10:00:00' --bt_hour 5 6 --save_format by_date --resolution 0.045 --lat_min -35 --lat_max 5 --lon_min -80 --lon_max -30
```

### Credits
And this is a otimization by Helvecio Neto - 2025
