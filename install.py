import sys
import os

# Fix module path to import from sd_reforge root
# This ensures we import the correct launch.py from parent directory
extension_dir = os.path.dirname(os.path.realpath(__file__))
sd_reforge_dir = os.path.dirname(os.path.dirname(extension_dir))

# Remove extension directory from path to prevent importing local modules
if extension_dir in sys.path:
    sys.path.remove(extension_dir)

# Add sd_reforge to path first
sys.path.insert(0, sd_reforge_dir)

import launch


def check_and_install():
    pass


check_and_install()
