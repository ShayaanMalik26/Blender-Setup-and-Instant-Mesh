import os
import glob
import time
import subprocess
from datetime import datetime
import argparse

# Set up the argument parser
parser = argparse.ArgumentParser(description="Process .glb files with Blender")
parser.add_argument(
    "--directory", 
    type=str, 
    required=True, 
    help="The directory containing .glb files"
)

args = parser.parse_args()

# Record the start time and convert to date and time
start_time = time.time()
start_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

DIRECTORY = args.directory  # data absolute path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Process each .glb file in the directory
for glb_file in glob.glob(os.path.join(DIRECTORY, '*.glb')):
    print(f"Processing {glb_file}")

    basename = os.path.basename(glb_file).split(".")[0]
    subprocess.run([
        'blender.exe', '-b', '-noaudio', '-P',
        f'{ROOT_DIR}/blender_script.py',
        '--', '--object_path', glb_file,
        '--output_dir', os.path.join(DIRECTORY, f'rendering_random_32views/{basename}')
    ])

# Record the end time and convert to date and time
end_time = time.time()
end_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Calculate the total duration
elapsed = int(end_time - start_time)

# Convert the total duration to hours, minutes, and seconds
hours, remainder = divmod(elapsed, 3600)
minutes, seconds = divmod(remainder, 60)

# Print the execution results
print(f"Start time: {start_date}")
print(f"End time: {end_date}")
print("Total time elapsed: ", end="")

# Print only if hours, minutes, or seconds are not zero
if hours > 0:
    print(f"{hours} hours ", end="")

if minutes > 0 or hours > 0:
    print(f"{minutes} minutes ", end="")

if seconds > 0 or minutes > 0 or hours > 0:
    print(f"{seconds} seconds")

print("")
