#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys


from pathlib import Path
import os

def IN_COLAB():
    try:
      from google.colab import drive
      IN_COLAB=True
    except:
      IN_COLAB=False
    
    return IN_COLAB

def upload(data_file, file_desc="", data_dir=None):
  # What directory are we in ?
  notebook_dir = os.getcwd()

  print("Current directory is: ", notebook_dir)

  # Check that the notebook directory is in sys.path
  # This is needed for the import of the data_file to succeed
  if not notebook_dir in sys.path:
    print(f"Adding {notebook_dir} to sys.path")

    sys.path.append(notebook_dir)

  if data_dir is not None:
    data_path     = Path(notebook_dir) / data_dir
    datafile_path = Path(notebook_dir) / data_dir / data_file
  else:
    data_path = Path(notebook_dir)
    datafile_path = Path(notebook_dir) / data_file


  if not data_path.is_dir():
    print(f"Creating the {data_dir} directory")
    os.makedirs(data_path)

  if not datafile_path.exists():
    print(f"Upload the {file_desc} file: {data_file} to directory {data_path}")
    print("\tIf file is large it may take a long time to upload.  Make sure it is completely uploaded before proceeding")

    print()
    print("\tAs an alternative: place the file on a Google Drive and mount the drive.")
    print("\t\tYou will have to add the path to the directory to sys.path -- see code above for modifying sys.path")

    # We will upload to the directory stored in variable data_path
    # This will necessitate changing the current directory; we will save it and restore it after the upload
    current_dir = os.getcwd()
    os.chdir(data_path)

    if IN_COLAB():
        from google.colab import files
        _= files.upload()
    else:
        print(f"Upload the {file_desc} file: {data_file} to directory {data_path}.")
        uploaded = input("Press ENTER when done.")

    # Restore the current working directory to the original directory
    os.chdir(current_dir)

from pathlib import Path


notebook_dir = os.getcwd()

def get_API_token(token_file=f"/{notebook_dir}/hf.token"):
    # Check for file containing API token to HuggingFace
    p = Path(token_file).expanduser()
    if not p.exists():
      print(f"Token file {p} not found.")
      return

    with open(token_file, 'r') as fp:
        token = fp.read()

    # Remove trailing newline
    token = token.rstrip()

    return token

