#!/bin/bash
set -e

echo "Updating system..."
sudo apt update && sudo apt upgrade -y

echo "Installing Python and system packages..."
sudo apt install -y python3 python3-pip python3-gpiozero python3-rpi.gpio python3-picamera2 libopencv-dev python3-opencv

echo "Upgrading pip..."
python3 -m pip install --upgrade pip

echo "Installing Python libraries..."
python3 -m pip install \
    requests \
    numpy \
    supervision \
    ultralytics

echo "All dependencies installed successfully!"
