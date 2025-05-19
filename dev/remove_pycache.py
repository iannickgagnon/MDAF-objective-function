"""
This script removes all __pycache__ directories from the specified path.
"""

import os
from os.path import join

# Number of files removed
nb_removed = 0

# Go through all directories and files starting from project root
for root, dirs, files in os.walk("src"):
    for file in files:

        # Construct the full path to the file
        file_path = join(root, file)
        print("Checking:", file_path)

        # Check if the file is a __pycache__ directory and remove it
        if "__pycache__" in root:  
            try:
                os.remove(file_path)
                print(f"Removed {file_path}")
                nb_removed += 1
            except Exception as e:
                print(f"Error removing {file_path}: {e}")

# Print the number of files removed
print(f"Removed {nb_removed} __pycache__ files.")
