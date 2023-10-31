import os
import subprocess

# Define the command
command = "yes | python -m dataset_librarian.dataset -n mvtec-ad --download --preprocess -d /cnvrg/data"

# Execute the command
subprocess.run(command, shell=True)