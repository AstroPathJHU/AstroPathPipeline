#import sys
import os
from pathlib import Path

# replace with your preferred directory path
dir_path = Path("C:\\Users\\Public\\astropath\\py")
file_name = "condapythoncheck.txt"
file_path = dir_path.joinpath(file_name)

# check that directory exists
if dir_path.is_dir():
    with open(file_path, "w") as f:
        f.write("This text is written with Python.")
else:
    raise Exception("Directory does not exist")

if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
    os.remove(file_path)
else:
    raise Exception("Test file not created correctly, error using python in conda")