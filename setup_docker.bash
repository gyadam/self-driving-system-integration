#!/bin/bash
# This script updates the docker container to be compatible with our repository. Has to be run every time you enter the docker container.

pip install --upgrade pip
pip install -r requirements.txt
pip install mock
pip install matplotlib
sudo apt-get install python-tk
pip install pillow --upgrade
