#!/bin/bash

# Check if folder path is provided
if [ -z "$1" ]; then
  echo "Usage: $0 /path/to/folder"
  exit 1
fi

# The folder to clear
FOLDER="$1"

# Check if the folder exists
if [ ! -d "$FOLDER" ]; then
  echo "Error: Folder '$FOLDER' does not exist."
  exit 1
fi

echo "Are you sure you want to clear the contents of the folder '$FOLDER'? (y/n)"
read -r response

# Check if the response is 'y' or 'Y'
if [[ "$response" =~ ^[Yy]$ ]]; then
  # Clear the contents of the folder
  rm  "$FOLDER"/*
  echo "The folder '$FOLDER' has been cleared."
else
  echo "Operation cancelled. The folder '$FOLDER' was not cleared."
  exit 0
fi
