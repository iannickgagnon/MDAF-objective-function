"""
This script updates the `__init__.py` file in the `src/MDAF_benchmarks` directory to automatically import 
all concrete implementations of the `ObjectiveFunction` class.

This simplified import syntax for clients in the following way:

    from MDAF_benchmarks import Eggholder

Instead of:

    from MDAF_benchmarks.implementations.eggholder import Eggholder
"""

import os

# Path to concrete implementations of the ObjectiveFunction class
implementations_path = os.path.join("src", "MDAF_benchmarks", "implementations")

# Path to the __init__.py folder of the implementations folder
init_file_path = os.path.join("src", "MDAF_benchmarks", "__init__.py")

# Helper to convert file name to PascalCase
def to_class_name(file_name):
    return ''.join(part.capitalize() for part in file_name.split('_'))

# Storage for text lines to write in updated __init__.py
lines = []
all_entries = []

# Loop through all files in the implementations directory
for filename in sorted(os.listdir(implementations_path)):
    if filename.endswith(".py") and filename != "__init__.py":
        
        # Extract the filen's name and convert it to PascalCase (i.e., the class name)
        module = filename[:-3]
        class_name = to_class_name(module)

        # Add import statement and type hint to the lines list
        lines.append(f"from .implementations.{module} import {class_name} as _{class_name}")
        lines.append(f"{class_name}: type[_{class_name}] = _{class_name}\n")
        
        # Add the class name to the __all__ list
        all_entries.append(f'"{class_name}"')

# Add __all__ at the end
lines.append(f"__all__ = [{', '.join(all_entries)}]")

# Write to __init__.py
with open(init_file_path, "w", encoding="utf-8") as f:

    # Write the header
    f.write("# Auto-generated re-exports for top-level API\n")
    
    # Write the imports
    f.write("from typing import Type\n\n")
    f.write('\n'.join(lines))

    # Add helper function to list all objective functions
    f.write("\n\ndef show_all() -> None:\n")
    f.write('    """\n')
    f.write('    Helper function to list all objective functions available in the package.\n')
    f.write('    """\n')
    f.write('    for entry in __all__:\n')
    f.write('        print(entry)\n')
    f.write('\n')
    
print(f"Updated {init_file_path} with {len(all_entries)} exports.")

