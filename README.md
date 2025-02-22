# goesgcp
<!-- badges: start -->
[![pypi](https://badge.fury.io/py/goesgcp.svg)](https://pypi.python.org/pypi/goesgcp)
[![Downloads](https://img.shields.io/pypi/dm/goesgcp.svg)](https://pypi.python.org/pypi/goesgcp)
[![Upload Python Package](https://github.com/helvecioneto/goesgcp/actions/workflows/python-publish.yml/badge.svg)](https://github.com/helvecioneto/goesgcp/actions/workflows/python-publish.yml)
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
```bash
git clone --branch labren --single-branch https://github.com/helvecioneto/goesgcp.git
```

```bash
cd goesgcp 
```

```bash
conda env create -f goesgcp.yml
```

```bash
conda activate goesgcp
```


```bash
nohup python goesgcp/main.py --start "2023-01-01 00:00:00" --end "2023-06-30 23:50:00" --bt_hour 9 22  --output /media/qnap2/goes16_teste/ --remap grade_1km.nc &
```

### Contributing
Contributions are welcome! If you encounter issues or have suggestions for improvements, please submit them via GitHub issues or pull requests.

### Credits
This project was developed and optimized by Helvecio Neto (2025).
It builds upon NOAA GOES-R data and leverages resources provided by the Google Cloud Platform.

### License
This project is licensed under the MIT License. 
