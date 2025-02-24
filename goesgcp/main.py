import pathlib
import shutil
import time
import subprocess
import argparse
import sys
import tqdm
import pandas as pd
from distutils.util import strtobool
from multiprocessing import Pool
from google.cloud import storage
from google.api_core.exceptions import GoogleAPIError
from datetime import datetime, timedelta, timezone
import xarray as xr
import numpy as np
from osgeo import gdal, osr
import netCDF4
import warnings
warnings.filterwarnings('ignore')
gdal.PushErrorHandler('CPLQuietErrorHandler')


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

    file, output, var_name, lat_min, lat_max, lon_min, lon_max, \
    resolution, save_format, \
    file_pattern, classic_format, resample = args

    # Read file using gdal
    try:
        src_ds = gdal.Open(f"NETCDF:{file}:"+var_name)
    except:
        # open using netCDF4 and show all 2D variables available
        ds = netCDF4.Dataset(file)
        vars = [var for var in ds.variables if len(ds.variables[var].shape) == 2]
        print(f"\nAvailable 2D variables in the file: {vars}")
        ds.close()
        return

    # Get source projection
    source_prj = osr.SpatialReference()
    source_prj.ImportFromProj4(src_ds.GetProjectionRef())

    # Set target projection
    target_prj = osr.SpatialReference()
    target_prj.ImportFromProj4("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")

    # Get name of file without extension
    file_name = file.split('/')[-1].split('.')[0]
    file_datetime = datetime.strptime(file_name[27:40], '%Y%j%H%M%S')
    
    # Set output file based on save_format
    if save_format == 'by_date':
        year = file_datetime.strftime("%Y")
        month = file_datetime.strftime("%m")
        day = file_datetime.strftime("%d")
        output_directory = f"{output}{year}/{month}/"
    elif save_format == 'julian':
        year = file_datetime.strftime("%Y")
        julian_day = file_datetime.timetuple().tm_yday
        output_directory = f"{output}{year}/{julian_day}/"
    else:
        output_directory = output

    # Set output file name based on file_pattern
    if file_pattern is not None:
        file_name = f"{file_datetime.strftime(file_pattern)}.nc"
    else:
        file_name = f"{file_name}.nc"

    # Create the output directory
    pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)

    # Get resample algorithm
    if resample == 'remapbil':
        resample_alg = gdal.GRA_Bilinear
    elif resample == 'remapcub':
        resample_alg = gdal.GRA_Cubic
    elif resample == 'remapcubicspline':
        resample_alg = gdal.GRA_CubicSpline
    elif resample == 'remaplanczos':
        resample_alg = gdal.GRA_Lanczos
    else:
        resample_alg = gdal.GRA_NearestNeighbour

    kwargs = {
        'format': 'netCDF',
        'srcSRS': source_prj.ExportToWkt(),
        'dstSRS': target_prj.ExportToWkt(),
        'outputBounds': (lon_min, lat_min, lon_max, lat_max),
        'xRes': resolution,
        'yRes': resolution,
        'resampleAlg': resample_alg,
    }

    # Reproject the file
    gdal.Warp(f"tmp/{file_name}_tmp1.nc", src_ds, **kwargs)
    src_ds = None

    # Open the reprojected file
    ds = xr.open_dataset(f"tmp/{file_name}_tmp1.nc", decode_cf=False)
    ds.attrs["comments"] = "Data processed by goesgcp, author: Helvecio B. L. Neto (2025)"
    ds = ds.rename({"Band1": var_name})
    ds[var_name].attrs.update({
        "resolution": f"{resolution} degrees"
    })    

    # Save based on classic_format
    if classic_format:
        ds.to_netcdf(output_directory + file_name, format='NETCDF3_CLASSIC')
    else:
        ds.to_netcdf(output_directory + file_name)


def remap_file(args):
    """ Remap the download file based on the input file. """

    base_remap, src_file, method = args
    # Target is a temporary file based on the output_file add _tmp2
    target_file = src_file.replace("_tmp1.nc", "_tmp2.nc")
    # Remap the file using CDO
    cdo_command = ["cdo", method+"," + base_remap, src_file, target_file]
    subprocess.run(cdo_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Remove the temporary file
    pathlib.Path(src_file).unlink
    # Rename the target file to the output_file
    pathlib.Path(target_file).rename(src_file)
    return


def process_file(args):
    """
    Downloads and processes a GOES-16 file.
    """
    
    bucket_name, blob_name, local_path, output_path, var_name, \
    lat_min, lat_max, lon_min, lon_max, resolution, \
    save_format, retries, resample, file_pattern, classic_format = args

    #Download the file
    attempt = 0
    while attempt < retries:
        try:
            # Connect to the bucket
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(local_path, timeout=120)
            break  # Exit the loop if the download is successful
        except (GoogleAPIError, Exception) as e:  # Catch any exception
            attempt += 1
            if attempt < retries:
                time.sleep(2 ** attempt)  # Backoff exponencial
            else:
                with open('fail.log', 'a') as log_file:
                    log_file.write(f"Failed to download {blob_name} after {retries} attempts. Error: {e}\n")
    #Crop the file
    try:
        crop_reproject((local_path, output_path,
                        var_name, lat_min, lat_max, lon_min, lon_max,
                        resolution, save_format, 
                        file_pattern, classic_format, resample))
    except Exception as e:
        with open('fail.log', 'a') as log_file:
            log_file.write(f"Failed to process {blob_name}. Error: {e}\n")
        pass
    #Remove the local file
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
    parser.add_argument('--resolution', type=float, default=0.1, help='Resolution of the output file')
    parser.add_argument('--resample', type=str, default='near', help='Resample algorithm to use (e.g., near, bilinear, cubic, lanczos)')
    parser.add_argument('--output', type=str, default='./output/', help='Path for saving output files')

    # Other settings
    parser.add_argument('--parallel', type=lambda x: bool(strtobool(x)), default=True, help='Use parallel processing')
    parser.add_argument('--processes', type=int, default=4, help='Number of processes for parallel execution')
    parser.add_argument('--max_attempts', type=int, default=3, help='Number of attempts to download a file')
    parser.add_argument('--save_format', type=str, default='flat', choices=['flat', 'by_date','julian'],
                    help="Save the files in a flat structure or by date")
    parser.add_argument('--file_pattern', type=str, default=None, help='Pattern for the files')
    parser.add_argument('--netcdf_classic', type=lambda x: bool(strtobool(x)), default=False, help='Save the files in netCDF classic format')
    # Parse arguments
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    # Check mandatory arguments
    if not args.recent and not (args.start and args.end):
        print("You must provide either the --recent or --start and --end arguments. Exiting...")
        sys.exit(1)

    # Check if args.start and args.end are provided together and are in correct format
    if args.start and args.end:
        try:
            datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S")
            datetime.strptime(args.end, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            print("Incorrect date format. Dates must be in the format 'YYYY-MM-DD HH:MM:SS'. Exiting...")
            sys.exit(1)

    # Set bucket name and pattern
    bucket_name = "gcp-public-data-" + args.satellite

    # Create output directory
    pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)

    # Check if the bucket exists
    try:
        storage_client.get_bucket(bucket_name)
    except Exception as e:
        print(f"Bucket {bucket_name} not found. Exiting...")
        sys.exit(1)

    # Check if the channel exists
    channel = args.channel
    if not channel:
        channel = ''
    else:
        channel = str(channel).zfill(2)
        channel = f"C{channel}"

    # Set pattern for the files
    pattern = "OR_"+args.product+"-"+args.op_mode+channel+"_G" + args.satellite[-2:]

    # Check operational mode if is recent or specific date
    if args.start and args.end:
        files_list = get_files_period(storage_client, bucket_name,
                                    args.product, pattern, args.start, args.end,
                                    args.bt_hour, args.bt_min, args.freq)
    else:
        # Get recent files
        files_list = get_recent_files(storage_client, bucket_name, args.product, pattern, args.recent)

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
    
    if args.parallel:
        # Create a list of tasks
        tasks = [(bucket_name, file, f"tmp/{file.split('/')[-1]}", args.output, args.var_name,
        args.lat_min, args.lat_max, args.lon_min, args.lon_max, args.resolution,
        args.save_format, args.max_attempts, args.resample,
        args.file_pattern, args.netcdf_classic) for file in files_list]

        # Download files in parallel
        with Pool(processes=args.processes) as pool:
            for _ in pool.imap_unordered(process_file, tasks):
                loading_bar.update(1)
        loading_bar.close()
    else: # Run in serial
        for file in files_list:
            local_path = f"tmp/{file.split('/')[-1]}"
            process_file((bucket_name, file, local_path, args.output, args.var_name,
                        args.lat_min, args.lat_max, args.lon_min, args.lon_max, args.resolution,
                        args.save_format, args.max_attempts, args.resample,
                        args.file_pattern, args.netcdf_classic))
            loading_bar.update(1)
        loading_bar.close()

    # Clean up the temporary directory
    shutil.rmtree('tmp/')

if __name__ == '__main__':
    main()
