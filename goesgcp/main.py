import pathlib
import shutil
import time
import xarray as xr
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
                     start, end, bt_hour=[0, 23], bt_min=[0, 60], freq='10 min'):
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
    current_time = start
    while current_time < end:
        year = current_time.year
        julian_day = str(current_time.timetuple().tm_yday).zfill(3)  # Julian day
        hour = current_time.hour

        # Generate the directory prefix
        prefix = f"{base_prefix}/{get_directory_prefix(year, julian_day, hour)}"

        # List blobs in the bucket for the current prefix
        blobs = list_blobs(connection, bucket_name, prefix)

        # Filter blobs by pattern
        for blob in blobs:
            if pattern in blob.name:
                files_metadata.append({
                    'file_name': blob.name,
                    'last_modified': blob.updated
                })

        # Move to the next hour
        current_time += timedelta(hours=1)

    # Create a DataFrame from the list of files
    df = pd.DataFrame(files_metadata)

    if df.empty:
        print("No files found matching the pattern.")
        return pd.DataFrame()
    
    # Ensure 'last_modified' is in the correct datetime format without timezone
    df['last_modified'] = pd.to_datetime(df['last_modified']).dt.tz_localize(None)
    start = pd.to_datetime(start).tz_localize(None)
    end = pd.to_datetime(end).tz_localize(None)
    
    # Filter the DataFrame based on the date range
    df = df[(df['last_modified'] >= start) & (df['last_modified'] < end)]

    # Filter the DataFrame based on the hour range
    df['hour'] = df['last_modified'].dt.hour
    df = df[(df['hour'] >= bt_hour[0]) & (df['hour'] <= bt_hour[1])]

    # Filter the DataFrame based on the minute range
    df['minute'] = df['last_modified'].dt.minute
    df = df[(df['minute'] >= bt_min[0]) & (df['minute'] <= bt_min[1])]

    # Filter the DataFrame based on the frequency
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
            if pattern in blob.name:  # You can use "re" here for more complex patterns
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

    file, output = args

    # Open the file
    ds = xr.open_dataset(file, engine="netcdf4")

    # Select only var_name and goes_imager_projection
    ds = ds[[var_name, "goes_imager_projection"]]

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

    # Reproject to EPSG:4326
    ds = ds.rio.reproject("EPSG:4326", resolution=resolution)

    # Rename lat/lon coordinates
    ds = ds.rename({"x": "lon", "y": "lat"})

    # Add resolution to attributes
    ds[var_name].attrs['resolution'] = "x={:.2f} y={:.2f} degree".format(resolution, resolution) 

    # Crop using lat/lon coordinates, in parallel
    ds = ds.rio.clip_box(minx=lon_min, miny=lat_min, maxx=lon_max, maxy=lat_max)

    # Add comments
    ds[var_name].attrs['comments'] = 'Cropped and reprojected to EPSG:4326 by goesgcp'

    # Add global metadata comments
    ds.attrs['comments'] = "Data processed by goesgcp, author: Helvecio B. L. Neto (helvecioblneto@gmail.com)"
        
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

    # Create the output directory
    pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)

    # Save the file
    output_file = f"{output_directory}{file.split('/')[-1]}"
    ds.to_netcdf(output_file, mode='w', format='NETCDF4_CLASSIC')

    # Fechar o dataset
    ds.close()
    return


def process_file(args):
    """ Downloads and processes a file in parallel. """

    bucket_name, blob_name, local_path = args

    # Download options
    retries = 5
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
    crop_reproject((local_path, output_path))

    # Remove the local file
    pathlib.Path(local_path).unlink()


def main():
    ''' Main function to download and process GOES-16 files. '''


    global output_path, var_name, \
          lat_min, lat_max, lon_min, lon_max, \
          max_attempts, parallel, recent, resolution, storage_client, \
            satellite, product, op_mode, channel, save_format

    epilog = """
    Example usage:
    
    - To download recent 3 files from the GOES-16 satellite for the ABI-L2-CMIPF product:

    goesgcp --satellite goes16 --product ABI-L2-CMIP --recent 3

    - To download files from the GOES-16 satellite for the ABI-L2-CMIPF product between 2022-12-15 and 2022-12-20:

    goesgcp --start '2022-12-15 00:00:00' --end '2022-12-20 10:00:00' --bt_hour 5 6 --save_format by_date --resolution 0.045 --lat_min -35 --lat_max 5 --lon_min -80 --lon_max -30

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
    parser = argparse.ArgumentParser(description='Converts GOES-16 L2 data to netCDF',
                                    epilog=epilog,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # Satellite and product settings
    parser.add_argument('--satellite', type=str, default='goes-16', choices=['goes-16', 'goes-17'], help='Name of the satellite (e.g., goes16)')
    parser.add_argument('--product', type=str, default='ABI-L2-CMIP', help='Name of the satellite product', choices=product_names)
    parser.add_argument('--var_name', type=str, default='CMI', help='Variable name to extract (e.g., CMI)')
    parser.add_argument('--channel', type=int, default=13, help='Channel to use (e.g., 13)')
    parser.add_argument('--op_mode', type=str, default='M6C', help='Operational mode to use (e.g., M6C)')

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
    parser.add_argument('--resolution', type=float, default=0.03208, help='Resolution of the output file')
    parser.add_argument('--output', type=str, default='output/', help='Path for saving output files')

    # Other settings
    parser.add_argument('--parallel', type=lambda x: bool(strtobool(x)), default=True, help='Use parallel processing')
    parser.add_argument('--processes', type=int, default=4, help='Number of processes for parallel execution')
    parser.add_argument('--max_attempts', type=int, default=3, help='Number of attempts to download a file')
    parser.add_argument('--save_format', type=str, default='flat', choices=['flat', 'by_date','julian'],
                    help="Save the files in a flat structure or by date")


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
    channel = str(args.channel).zfill(2)
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


    # Check mandatory arguments
    if not args.recent and not (args.start and args.end):
        print("You must provide either the --recent or --start and --end arguments. Exiting...")
        sys.exit(1)

    # Set bucket name and pattern
    bucket_name = "gcp-public-data-" + satellite

    # Create output directory
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    # Create connection
    storage_client = storage.Client.create_anonymous_client()

    # Check if the bucket exists
    try:
        storage_client.get_bucket(bucket_name)
    except Exception as e:
        print(f"Bucket {bucket_name} not found. Exiting...")
        sys.exit(1)

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
        tasks = [(bucket_name, file, f"tmp/{file.split('/')[-1]}") for file in files_list]

        # Download files in parallel
        with Pool(processes=args.processes) as pool:
            for _ in pool.imap_unordered(process_file, tasks):
                loading_bar.update(1)
        loading_bar.close()
    else: # Run in serial
        for file in files_list:
            local_path = f"tmp/{file.split('/')[-1]}"
            process_file((bucket_name, file, local_path))
            loading_bar.update(1)
        loading_bar.close()

    # Remove temporary directory
    shutil.rmtree('tmp/')

if __name__ == '__main__':
    main()
