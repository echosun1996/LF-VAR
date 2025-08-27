#!/bin/bash

# Check if it's Ubuntu 22.04
if [[ "$(lsb_release -rs)" != "22.04" ]]; then
    echo "This script is only for Ubuntu 22.04. Current system is $(lsb_release -rs)."
    exit 1
fi

echo "Updating package list..."
sudo apt update -y

# Install zip
echo "Installing zip..."
sudo apt install zip -y

echo "Installing mosh..."
sudo apt-get install mosh -y

# Check if NVIDIA GPU exists
if ! lspci | grep -i nvidia > /dev/null; then
    echo "No NVIDIA GPU detected, skipping driver installation."
    exit 1
fi


sudo systemctl stop google-cloud-ops-agent
cd $HOME
curl -L https://github.com/GoogleCloudPlatform/compute-gpu-installation/releases/download/cuda-installer-v1.1.0/cuda_installer.pyz --output cuda_installer.pyz
sudo python3 cuda_installer.pyz install_driver
sudo python3 cuda_installer.pyz install_cuda

# Add NVIDIA CUDA repository
echo "Adding NVIDIA CUDA repository..."
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:graphics-drivers/ppa
sudo apt update -y

sudo apt install -y ubuntu-drivers-common

# Determine recommended driver version
echo "Detecting recommended NVIDIA driver version..."
RECOMMENDED_DRIVER=$(ubuntu-drivers devices | grep "recommended" | awk '{print $3}')
if [ -z "$RECOMMENDED_DRIVER" ]; then
    echo "Could not find recommended driver version, please check manually."
    exit 1
fi

# Install recommended driver
echo "Installing NVIDIA driver version $RECOMMENDED_DRIVER..."
sudo apt install -y "nvidia-driver-${RECOMMENDED_DRIVER}"

# Check after installation completion
echo "Verifying NVIDIA driver installation success..."
nvidia-smi || {
    echo "NVIDIA driver installation failed, please check system logs for more information."
    exit 1
}

echo "Installation completed. System needs to restart to apply changes."