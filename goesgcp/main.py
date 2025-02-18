import pathlib
import shutil
import time
import xarray as xr
import subprocess
import argparse
import sys
import tqdm
import pandas as pd
from distutils.util import strtobool
from multiprocessing import Pool
from google.cloud import storage
from datetime import datetime, timedelta, timezone
from pyproj import CRS, Transformer
from google.api_core.exceptions import GoogleAPIError

import warnings
warnings.filterwarnings('ignore')


def list_blobs(connection, bucket_name, prefix):
    """
    Lists blobs in a GCP bucket with a specified prefix.
    Returns a list of blobs with their metadata.
    """
    bucket = connection.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=prefix)

    return blobs

def get_directory_prefix(year, julian_day, hour):
    """Generates the directory path based on year, Julian day, and hour."""
    return f"{year}/{julian_day}/{str(hour).zfill(2)}/"


def get_files_period(connection, bucket_name, base_prefix, pattern, 
                     start, end, bt_hour=[], bt_min=[], freq=None):
    """
    Fetches files from a GCP bucket within a specified time period and returns them as a DataFrame.

    :param connection: The GCP storage client connection.
    :param bucket_name: Name of the GCP bucket.
    :param base_prefix: Base directory prefix for the files.
    :param pattern: Search pattern for file names.
    :param start: Start datetime (inclusive).
    :param end: End datetime (exclusive).
    :return: DataFrame containing the file names and their metadata.
    """

    print(f"GOESGCP: Fetching files between {start} and {end}...")

    # Ensure datetime objects
    start = pd.to_datetime(start).tz_localize('UTC')
    end = pd.to_datetime(end).tz_localize('UTC')

    # Initialize list to store file metadata
    files_metadata = []

    # Generate the list of dates from start to end
    temp = start
    while temp <= end:
        year = temp.year
        julian_day = str(temp.timetuple().tm_yday).zfill(3)  # Julian day
        hour = temp.hour

        # Generate the directory prefix
        prefix = f"{base_prefix}/{get_directory_prefix(year, julian_day, hour)}"

        # List blobs in the bucket for the current prefix
        blobs = list_blobs(connection, bucket_name, prefix)

        # Filter blobs by pattern
        for blob in blobs:
            if pattern in blob.name:
                files_metadata.append({
                    'file_name': blob.name,
                })

        # Move to the next hour
        temp += timedelta(hours=1)

    # Create a DataFrame from the list of files
    df = pd.DataFrame(files_metadata)

    if df.empty:
        print("No files found matching the pattern and time range.")
        print(prefix)
        sys.exit(1)
    
    # Transform file_name to datetime
    df['last_modified'] = pd.to_datetime(df['file_name'].str.extract(r'(\d{4}\d{3}\d{2}\d{2})').squeeze(), format='%Y%j%H%M')

    # Ensure 'last_modified' is in the correct datetime format without timezone
    df['last_modified'] = pd.to_datetime(df['last_modified']).dt.tz_localize('UTC')

    # Filter the DataFrame based on the date range (inclusive)
    df = df[(df['last_modified'] >= start) & (df['last_modified'] <= end)]

    # Filter the DataFrame based on the hour range
    if len(bt_hour) > 1:
        df['hour'] = df['last_modified'].dt.hour
        df = df[(df['hour'] >= bt_hour[0]) & (df['hour'] <= bt_hour[1])]

    # Filter the DataFrame based on the minute range
    if len(bt_min) > 1:
        df['minute'] = df['last_modified'].dt.minute
        df = df[(df['minute'] >= bt_min[0]) & (df['minute'] <= bt_min[1])]

    # Filter the DataFrame based on the frequency
    if freq is not None:
        df['freq'] = df['last_modified'].dt.floor(freq)
        df = df.groupby('freq').first().reset_index()

    return df['file_name'].tolist()

def get_recent_files(connection, bucket_name, base_prefix, pattern, min_files):
    """
    Fetches the most recent files in a GCP bucket.

    :param bucket_name: Name of the GCP bucket.
    :param base_prefix: Base directory prefix (before year/Julian day/hour).
    :param pattern: Search pattern for file names.
    :param min_files: Minimum number of files to return.
    :return: List of the n most recent files.
    """
    files = []
    current_time = datetime.now(timezone.utc)

    # Loop until the minimum number of files is found
    while len(files) < min_files:
        year = current_time.year
        julian_day = current_time.timetuple().tm_yday  # Get the Julian day
        # Add 3 digits to the Julian day
        julian_day = str(julian_day).zfill(3)
        hour = current_time.hour

        # Generate the directory prefix for the current date and time
        prefix = f"{base_prefix}/{get_directory_prefix(year, julian_day, hour)}"

        # List blobs from the bucket
        blobs = list_blobs(connection, bucket_name, prefix)

        # Filter blobs based on the pattern
        for blob in blobs:
            if pattern in blob.name:
                files.append((blob.name, blob.updated))

        # Go back one hour
        current_time -= timedelta(hours=1)

    # Sort files by modification date in descending order
    files.sort(key=lambda x: x[1], reverse=True)

    # Return only the names of the most recent files, according to the minimum requested
    return [file[0] for file in files[:min_files]]


def crop_reproject(args):
    """
    Crops and reprojects a GOES-16 file to EPSG:4326.
    """

    file, output, var_name, lat_min, lat_max, lon_min, lon_max, resolution, save_format, \
    more_info, file_pattern, classic_format, remap, method = args

    # Open the file
    ds = xr.open_dataset(file, engine="netcdf4")

    if var_name is None:
        # Get all variables are 2D
        var_names = [var for var in ds.data_vars if len(ds[var].dims) == 2]

        # Remove DQF variables
        var_names = [var for var in var_names if 'DQF' not in var]
    else:
        var_names = [var_name]

    # Select only var_name and goes_imager_projection
    ds = ds[var_names + ["goes_imager_projection"]]

    # Get projection
    sat_height = ds["goes_imager_projection"].attrs["perspective_point_height"]
    ds = ds.assign_coords({
                "x": ds["x"].values * sat_height,
                "y": ds["y"].values * sat_height,
            })
    # Set CRS from goes_imager_projection
    crs = CRS.from_cf(ds["goes_imager_projection"].attrs)
    ds = ds.rio.write_crs(crs)

    # Try to reduce the size of the dataset
    try:
        # Create a transformer
        transformer = Transformer.from_crs(CRS.from_epsg(4326), crs)
        # Calculate the margin
        margin_ratio = 0.40  # 40% margin

        # Get the bounding box
        min_x, min_y = transformer.transform(lat_min, lon_min)
        max_x, max_y = transformer.transform(lat_max, lon_max)

        # Calculate the range
        x_range = abs(max_x - min_x)
        y_range = abs(max_y - min_y)

        margin_x = x_range * margin_ratio
        margin_y = y_range * margin_ratio

        # Expand the bounding box
        min_x -= margin_x
        max_x += margin_x
        min_y -= margin_y
        max_y += margin_y

        # Select the region
        if ds["y"].values[0] > ds["y"].values[-1]:  # Eixo y decrescente
            ds_ = ds.sel(x=slice(min_x, max_x), y=slice(max_y, min_y))
        else:  # Eixo y crescente
            ds_ = ds.sel(x=slice(min_x, max_x), y=slice(min_y, max_y))
        # Sort by y
        if ds_["y"].values[0] > ds_["y"].values[-1]:
            ds_ = ds_.sortby("y")
        # Assign to ds
        ds = ds_
    except:
        pass

    try:
        # Reproject to EPSG:4326
        ds = ds.rio.reproject("EPSG:4326", resolution=resolution)

        # Rename lat/lon coordinates
        ds = ds.rename({"x": "lon", "y": "lat"})

        # Add resolution to attributes
        for var in var_names:
            ds[var].attrs['resolution'] = "x={:.2f} y={:.2f} degree".format(resolution, resolution)
            ds[var].attrs['comments'] = 'Cropped and reprojected to EPSG:4326 by goesgcp'

        # Crop using lat/lon coordinates, in parallel
        ds = ds.rio.clip_box(minx=lon_min, miny=lat_min, maxx=lon_max, maxy=lat_max)

        # Add global metadata comments
        ds.attrs['comments'] = "Data processed by goesgcp, author: Helvecio B. L. Neto (helvecioblneto@gmail.com)"

        if more_info:
            # Get all scale_factor and add_offset
            ds_s = xr.open_dataset(file, engine="netcdf4", decode_cf=False)
            scale_factors = [ds_s[var].attrs["scale_factor"] for var in var_names]
            add_offsets = [ds_s[var].attrs["add_offset"] for var in var_names]
            sat_lat = ds_s["goes_imager_projection"].attrs["latitude_of_projection_origin"]
            sat_lon = ds_s["goes_imager_projection"].attrs["longitude_of_projection_origin"]
            ds_s.close()
            ds["satlat"] = sat_lat.astype('float32')
            ds["satlat"].attrs['comment'] = ds['satlat'].values
            # Add satellite longitude
            ds["satlon"] = sat_lon.astype('float32')
            ds["satlon"].attrs['comment'] = ds['satlon'].values
            # Restore scale_factor and add_offset change name to scale-factor and add-offset
            for var, scale_factor, add_offset in zip(var_names, scale_factors, add_offsets):
                ds[var].attrs['scale-factor'] = scale_factor
                ds[var].attrs['add-offset'] = add_offset

    except Exception as e:
        print(f"Error processing file {file}: {e}")
        pass
   
    if save_format == 'by_date':
        file_datetime = datetime.strptime(ds.time_coverage_start, 
                                          "%Y-%m-%dT%H:%M:%S.%fZ")
        year = file_datetime.strftime("%Y")
        month = file_datetime.strftime("%m")
        day = file_datetime.strftime("%d")
        output_directory = f"{output}{year}/{month}/{day}/"
    elif save_format == 'julian':
        file_datetime = datetime.strptime(ds.time_coverage_start, 
                                          "%Y-%m-%dT%H:%M:%S.%fZ")
        year = file_datetime.strftime("%Y")
        julian_day = file_datetime.timetuple().tm_yday
        output_directory = f"{output}{year}/{julian_day}/"
    else:
        output_directory = output

    
    if more_info:
        ds_stamp = datetime.strptime(ds.time_coverage_start, 
                                          "%Y-%m-%dT%H:%M:%S.%fZ")
        ds["julian_day"] = ds_stamp.strftime("%j")
        ds["julian_day"].attrs['comment'] = ds['julian_day'].values
        # Add variable time_of_day is a Hour and Minute, save as char
        ds["time_of_day"] = ds_stamp.strftime("%H%M")
        ds["time_of_day"].attrs['comment'] = ds['time_of_day'].values

    # Create the output directory
    pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)

    if file_pattern is not None:
        # Get the timestamp
        file_datetime = datetime.strptime(ds.time_coverage_start, 
                                          "%Y-%m-%dT%H:%M:%S.%fZ")
        
        # Format to file pattern
        file_pattern = file_datetime.strftime(file_pattern)

        # Save the file
        output_file = f"{output_directory}{file_pattern}.nc"
    else:
        # Save the file
        output_file = f"{output_directory}{file.split('/')[-1]}"

     # Write the file
    if classic_format:
        ds.to_netcdf(output_file, mode='w', format='NETCDF3_CLASSIC', encoding={var: {'zlib': True} for var in var_names})
    else:
        ds.to_netcdf(output_file, mode='w', encoding={var: {'zlib': True} for var in var_names})
    ds.close()

    # Remap the file
    if remap:
        remap_file((remap, output_file, output, method))


def remap_file(args):
    """ Remap the download file based on the input file. """

    base_file, target_file, output, method = args

    # Open the files
    base_ds = xr.open_dataset(base_file, engine="netcdf4")

    # Get output directory based on target_file
    output_file = f"{output}{target_file.split('/')[-1]}"
    output_directory = output_file.replace(target_file.split('/')[-1], "")

    # Create the output directory
    pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)

    output_file = f"{output_directory}{target_file.split('/')[-1]}"

    # Add _ into output_file to prevent overwrite
    output_file = output_file.replace(".nc", "_remap.nc")

    # Run the cdo command
    cdo_command = [
        "cdo", method+"," + base_file, target_file, output_file
    ]

    try:
        subprocess.run(cdo_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"Error remapping file {target_file}: {e}")
        pass

    # Close the files
    base_ds.close()

    # Delete the target file
    pathlib.Path(target_file).unlink()

    # Rename the output file
    pathlib.Path(output_file).rename(target_file)


def process_file(args):
    """
    Downloads and processes a GOES-16 file.
    """
    
    bucket_name, blob_name, local_path, output_path, var_name, lat_min, lat_max, lon_min, lon_max, resolution, \
    save_format, retries, remap, met, more_info, file_pattern, classic_format = args

    attempt = 0
    while attempt < retries:
        try:
            # Connect to the bucket
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            # Download the file
            blob.download_to_filename(local_path, timeout=120)
            break  # Exit the loop if the download is successful
        except (GoogleAPIError, Exception) as e:  # Catch any exception
            attempt += 1
            if attempt < retries:
                time.sleep(2 ** attempt)  # Backoff exponencial
            else:
                # Log the error to a file
                with open('fail.log', 'a') as log_file:
                    log_file.write(f"Failed to download {blob_name} after {retries} attempts. Error: {e}\n")

    # Crop the file
    crop_reproject((local_path, output_path, var_name,
                    lat_min, lat_max, lon_min, lon_max,
                    resolution, save_format, more_info,
                    file_pattern, classic_format, remap, met))

    # Remove the local file
    pathlib.Path(local_path).unlink()

# Create connection
storage_client = storage.Client.create_anonymous_client()

def main():
    ''' Main function to download and process GOES-16 files. '''

    epilog = """
    Example usage:
    
    - To download recent 3 files from the GOES-16 satellite for the ABI-L2-CMIPF product,
    change resolution to 0.045, and crop the files between latitudes -35 and 5 and longitudes -80 and -30:

    goesgcp --satellite goes-16 --product ABI-L2-CMIPF --recent 3 --resolution 0.045 --lat_min -35 --lat_max 5 --lon_min -80 --lon_max -30

    - To download files from the GOES-16 satellite for the ABI-L2-CMIPF product between 2022-12-15 and 2022-12-20:

    goesgcp --satellite goes-16 --product ABI-L2-CMIPF --start "2022-12-15 09:00:00" --end "2022-12-15 09:50:00" --resolution 0.045 --lat_min -35 --lat_max 5 --lon_min -80 --lon_max -30

    """

    product_names = [
    "ABI-L1b-RadF", "ABI-L1b-RadC", "ABI-L1b-RadM", "ABI-L2-ACHAC", "ABI-L2-ACHAF", "ABI-L2-ACHAM",
    "ABI-L2-ACHTF", "ABI-L2-ACHTM", "ABI-L2-ACMC", "ABI-L2-ACMF", "ABI-L2-ACMM", "ABI-L2-ACTPC",
    "ABI-L2-ACTPF", "ABI-L2-ACTPM", "ABI-L2-ADPC", "ABI-L2-ADPF", "ABI-L2-ADPM", "ABI-L2-AICEF",
    "ABI-L2-AITAF", "ABI-L2-AODC", "ABI-L2-AODF", "ABI-L2-BRFC", "ABI-L2-BRFF", "ABI-L2-BRFM",
    "ABI-L2-CMIPC", "ABI-L2-CMIPF", "ABI-L2-CMIPM", "ABI-L2-CODC", "ABI-L2-CODF", "ABI-L2-CPSC",
    "ABI-L2-CPSF", "ABI-L2-CPSM", "ABI-L2-CTPC", "ABI-L2-CTPF", "ABI-L2-DMWC", "ABI-L2-DMWF",
    "ABI-L2-DMWM", "ABI-L2-DMWVC", "ABI-L2-DMWVF", "ABI-L2-DMWVF", "ABI-L2-DSIC", "ABI-L2-DSIF",
    "ABI-L2-DSIM", "ABI-L2-DSRC", "ABI-L2-DSRF", "ABI-L2-DSRM", "ABI-L2-FDCC", "ABI-L2-FDCF",
    "ABI-L2-FDCM", "ABI-L2-LSAC", "ABI-L2-LSAF", "ABI-L2-LSAM", "ABI-L2-LSTC", "ABI-L2-LSTF",
    "ABI-L2-LSTM", "ABI-L2-LVMPC", "ABI-L2-LVMPF", "ABI-L2-LVMPM", "ABI-L2-LVTPC", "ABI-L2-LVTPF",
    "ABI-L2-LVTPM", "ABI-L2-MCMIPC", "ABI-L2-MCMIPF", "ABI-L2-MCMIPM", "ABI-L2-RRQPEF",
    "ABI-L2-RSRC", "ABI-L2-RSRF", "ABI-L2-SSTF", "ABI-L2-TPWC", "ABI-L2-TPWF", "ABI-L2-TPWM",
    "ABI-L2-VAAF", "EXIS-L1b-SFEU", "EXIS-L1b-SFXR", "GLM-L2-LCFA", "MAG-L1b-GEOF", "SEIS-L1b-EHIS",
    "SEIS-L1b-MPSH", "SEIS-L1b-MPSL", "SEIS-L1b-SGPS", "SUVI-L1b-Fe093", "SUVI-L1b-Fe131",
    "SUVI-L1b-Fe171", "SUVI-L1b-Fe195", "SUVI-L1b-Fe284", "SUVI-L1b-He303"
    ]

    # Set arguments
    parser = argparse.ArgumentParser(description='Download and process GOES Satellite data files from GCP.',
                                    epilog=epilog,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # Satellite and product settings
    parser.add_argument('--satellite', type=str, default='goes-16', choices=['goes-16', 'goes-18'], help='Name of the satellite (e.g., goes16)')
    parser.add_argument('--product', type=str, default='ABI-L2-CMIPF', help='Name of the satellite product', choices=product_names)
    parser.add_argument('--var_name', type=str, default=None, help='Variable name to extract (e.g., CMI)')
    parser.add_argument('--channel', type=int, default=13, help='Channel to use (e.g., 13)')
    parser.add_argument('--op_mode', type=str, default='M6', help='Operational mode to use (e.g., M6C)')

    # Recent files settings
    parser.add_argument('--recent', type=int, help='Number of recent files to download (e.g., 3)')

    # Date and time settings
    parser.add_argument('--start', type=str, help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end', type=str, help='End date in YYYY-MM-DD format')
    parser.add_argument('--freq', type=str, default='10 min', help='Frequency for the time range (e.g., "10 min")')
    parser.add_argument('--bt_hour', nargs=2, type=int, default=[0, 23], help='Filter data between these hours (e.g., 0 23)')
    parser.add_argument('--bt_min', nargs=2, type=int, default=[0, 60], help='Filter data between these minutes (e.g., 0 60)')

    # Geographic bounding box
    parser.add_argument('--lat_min', type=float, default=-81.3282, help='Minimum latitude of the bounding box')
    parser.add_argument('--lat_max', type=float, default=81.3282, help='Maximum latitude of the bounding box')
    parser.add_argument('--lon_min', type=float, default=-156.2995, help='Minimum longitude of the bounding box')
    parser.add_argument('--lon_max', type=float, default=6.2995, help='Maximum longitude of the bounding box')
    parser.add_argument('--resolution', type=float, default=0.01, help='Resolution of the output file')
    parser.add_argument('--output', type=str, default='./output/', help='Path for saving output files')

    # Remap
    parser.add_argument('--remap', type=str, default=None, help='Give a input file to remap the output')
    parser.add_argument('--method', type=str, default='remapnn', help='Remap method to use (e.g., remapnn)')

    # Other settings
    parser.add_argument('--parallel', type=lambda x: bool(strtobool(x)), default=True, help='Use parallel processing')
    parser.add_argument('--processes', type=int, default=4, help='Number of processes for parallel execution')
    parser.add_argument('--max_attempts', type=int, default=3, help='Number of attempts to download a file')
    parser.add_argument('--info', type=lambda x: bool(strtobool(x)), default=False, help='Show information messages')
    parser.add_argument('--save_format', type=str, default='flat', choices=['flat', 'by_date','julian'],
                    help="Save the files in a flat structure or by date")
    parser.add_argument('--file_pattern', type=str, default=None, help='Pattern for the files')
    parser.add_argument('--netcdf_classic', type=lambda x: bool(strtobool(x)), default=False, help='Save the files in netCDF classic format')
    # Parse arguments
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    # Set global variables
    output_path = args.output
    satellite = args.satellite
    product = args.product
    op_mode = args.op_mode
    channel = args.channel
    var_name = args.var_name
    lat_min = args.lat_min
    lat_max = args.lat_max
    lon_min = args.lon_min
    lon_max = args.lon_max
    resolution = args.resolution
    max_attempts = args.max_attempts
    parallel = args.parallel
    recent = args.recent
    start = args.start
    end = args.end
    freq = args.freq
    bt_hour = args.bt_hour
    bt_min = args.bt_min
    save_format = args.save_format
    remap = args.remap
    method = args.method
    more_info = args.info
    file_pattern = args.file_pattern
    classic_format = args.netcdf_classic

    # Check mandatory arguments
    if not args.recent and not (args.start and args.end):
        print("You must provide either the --recent or --start and --end arguments. Exiting...")
        sys.exit(1)

    # Set bucket name and pattern
    bucket_name = "gcp-public-data-" + satellite

    # Create output directory
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    # Check if the bucket exists
    try:
        storage_client.get_bucket(bucket_name)
    except Exception as e:
        print(f"Bucket {bucket_name} not found. Exiting...")
        sys.exit(1)

    # Check if the channel exists
    if not channel:
        channel = ''
    else:
        channel = str(channel).zfill(2)
        channel = f"C{channel}"

    # Set pattern for the files
    pattern = "OR_"+product+"-"+op_mode+channel+"_G" + satellite[-2:]

    # Check operational mode if is recent or specific date
    if start and end:
        files_list = get_files_period(storage_client, bucket_name,
                                    product, pattern, start, end,
                                    bt_hour, bt_min, freq)
    else:
        # Get recent files
        files_list = get_recent_files(storage_client, bucket_name, product, pattern, recent)

    # Check if any files were found
    if not files_list:
        print(f"No files found with the pattern {pattern}. Exiting...")
        sys.exit(1)

    # Create a temporary directory
    pathlib.Path('tmp/').mkdir(parents=True, exist_ok=True)

    # Download files
    print(f"GOESGCP: Downloading and processing {len(files_list)} files...")
    loading_bar = tqdm.tqdm(total=len(files_list), ncols=100, position=0, leave=True,
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} + \
                        [Elapsed:{elapsed} Remaining:<{remaining}]')
    
    if parallel: # Run in parallel
        # Create a list of tasks
        tasks = [(bucket_name, file, f"tmp/{file.split('/')[-1]}", output_path, var_name, 
        lat_min, lat_max, lon_min, lon_max, resolution,
        save_format, max_attempts, remap, method, 
        more_info, file_pattern, classic_format) for file in files_list]

        # Download files in parallel
        with Pool(processes=args.processes) as pool:
            for _ in pool.imap_unordered(process_file, tasks):
                loading_bar.update(1)
        loading_bar.close()
    else: # Run in serial
        for file in files_list:
            local_path = f"tmp/{file.split('/')[-1]}"
            process_file((bucket_name, file, local_path, output_path, var_name,
            lat_min, lat_max, lon_min, lon_max, resolution,
            save_format, max_attempts, remap, method, more_info,
              file_pattern, classic_format))
            loading_bar.update(1)
        loading_bar.close()

    shutil.rmtree('tmp/')

if __name__ == '__main__':
    main()
