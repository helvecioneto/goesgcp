import pathlib
import shutil
import xarray as xr
import argparse
import sys
import tqdm
from concurrent.futures import ThreadPoolExecutor
from google.cloud import storage
from datetime import datetime, timedelta, timezone
from pyproj import CRS



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

def download_file(connection, bucket_name, blob_name, local_path):
    """Downloads a file from a GCP bucket."""
    bucket = connection.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)

def crop_reproject(file, output):
    """
    Crops and reprojects a GOES-16 file to EPSG:4326.
    """


    ds = xr.open_dataset(file)
    # Select only var_name and goes_imager_projection
    ds = ds[[var_name, "goes_imager_projection"]]
    # Get projection
    sat_height = ds["goes_imager_projection"].attrs["perspective_point_height"]
    ds = ds.assign_coords({
                "x": ds["x"].values * sat_height,
                "y": ds["y"].values * sat_height,
            })
    # Set CRS
    crs = CRS.from_cf(ds["goes_imager_projection"].attrs)
    ds = ds.rio.write_crs(crs)

    # Reproject to EPSG:4326 using parallel processing
    ds = ds.rio.reproject(dst_crs="EPSG:4326",
                          resolution=(resolution, resolution),
                          num_threads=-1)

    # Rename lat/lon coordinates
    ds = ds.rename({"x": "lon", "y": "lat"})

    # # Crop using lat/lon coordinates, in parallel
    ds = ds.rio.clip_box(minx=lon_min, miny=lat_min, maxx=lon_max, maxy=lat_max)

    # Remove any previous file
    if pathlib.Path(f'{output}{file.split("/")[-1]}.nc').exists():
        pathlib.Path(f'{output}{file.split("/")[-1]}.nc').unlink()

    # Add comments
    ds[var_name].attrs['comments'] = 'Cropped and reprojected to EPSG:4326 by helvecioblneto@gmail.com'

    # # Save as netcdf
    ds.to_netcdf(f'{output}{file.split("/")[-1]}')

    # Remove original file
    pathlib.Path(file).unlink()

    return



def main():

    global output_path, var_name, \
          lat_min, lat_max, lon_min, lon_max, \
          max_attempts, parallel, recent, resolution

    epilog = """
    Example usage:
    
    - To download recent files from the GOES-16 satellite for the ABI-L2-CMIPF product, extracting the CMI variable from channel 13, in the last 30 minutes:

    goesgcp --satellite goes16 --product ABI-L2-CMIP --domain F --var_name CMI --channel 13 --recent 10 --output_path "output/"
    """


    # Set arguments
    parser = argparse.ArgumentParser(description='Converts GOES-16 L2 data to netCDF',
                                    epilog=epilog,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # Satellite and product settings
    parser.add_argument('--satellite', type=str, default='goes-16', help='Name of the satellite (e.g., goes16)')
    parser.add_argument('--product', type=str, default='ABI-L2-CMIP', help='Name of the satellite product')
    parser.add_argument('--var_name', type=str, default='CMI', help='Variable name to extract (e.g., CMI)')
    parser.add_argument('--channel', type=int, default=13, help='Channel to use (e.g., 13)')
    parser.add_argument('--domain', type=str, default='F', help='Domain to use (e.g., F or C)')
    parser.add_argument('--recent', type=int, default=3, help='Number of recent files to download')

    # Geographic bounding box
    parser.add_argument('--lat_min', type=float, default=-56, help='Minimum latitude of the bounding box')
    parser.add_argument('--lat_max', type=float, default=35, help='Maximum latitude of the bounding box')
    parser.add_argument('--lon_min', type=float, default=-116, help='Minimum longitude of the bounding box')
    parser.add_argument('--lon_max', type=float, default=-25, help='Maximum longitude of the bounding box')
    parser.add_argument('--resolution', type=float, default=0.045, help='Resolution of the output file')
    parser.add_argument('--output', type=str, default='output/', help='Path for saving output files')

    # Other settings
    parser.add_argument('--parallel', type=bool, default=True, help='Use parallel processing')
    parser.add_argument('--processes', type=int, default=4, help='Number of processes for parallel execution')
    parser.add_argument('--max_attempts', type=int, default=3, help='Number of attempts to download a file')

    # Parse arguments
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    # Set global variables
    output_path = args.output
    satellite = args.satellite
    product = args.product
    domain = args.domain
    channel = str(args.channel).zfill(2)
    var_name = args.var_name
    lat_min = args.lat_min
    lat_max = args.lat_max
    lon_min = args.lon_min
    lon_max = args.lon_max
    resolution = args.resolution
    max_attempts = args.max_attempts
    parallel = args.parallel

    # Set bucket name and pattern
    bucket_name = "gcp-public-data-" + satellite
    pattern = "OR_"+product+domain+"-M6C"+channel+"_G" + satellite[-2:]
    min_files = args.recent

    output = 'output/'
    pathlib.Path(output).mkdir(parents=True, exist_ok=True)

    # Create connection
    storage_client = storage.Client.create_anonymous_client()

    # Check if the bucket exists
    try:
        storage_client.get_bucket(bucket_name)
    except Exception as e:
        print(f"Bucket {bucket_name} not found. Exiting...")
        sys.exit(1)

    # Search for recent files
    recent_files = get_recent_files(storage_client, bucket_name, product + domain, pattern, min_files)

    # Check if any files were found
    if not recent_files:
        print(f"No files found with the pattern {pattern}. Exiting...")
        sys.exit(1)
    print('Downloading files...')
    # Loading bar
    loading_bar = tqdm.tqdm(total=len(recent_files), ncols=100, position=0, leave=True,
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} + \
                        [Elapsed:{elapsed} Remaining:<{remaining}]')
    
    # Create a temporary directory
    pathlib.Path('tmp/').mkdir(parents=True, exist_ok=True)

    # Download all files to a temporary directory
    with ThreadPoolExecutor(max_workers=args.processes) as executor:
        for file in recent_files:
            download_file(storage_client, bucket_name, file, f'tmp/{file.split("/")[-1]}')
            loading_bar.update(1)
    loading_bar.close()

    print('Cropping and reprojecting files...')
    # Crop and reproject all files in serial mode
    for file in recent_files:
        crop_reproject(f'tmp/{file.split("/")[-1]}', output)
        loading_bar.update(1)
    loading_bar.close()

    # Remove temporary directory
    shutil.rmtree('tmp/')

if __name__ == '__main__':
    main()
