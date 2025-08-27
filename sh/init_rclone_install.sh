#!/bin/bash

# Define installation directory
INSTALL_DIR="$HOME/bin"

# Create the bin directory if it doesn't exist
mkdir -p "$INSTALL_DIR"

# Function to install rclone
install_rclone() {
    echo "Downloading rclone..."

    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux installation
#        curl -O https://downloads.rclone.org/rclone-current-linux-amd64.zip
        wget https://downloads.rclone.org/rclone-current-linux-amd64.zip
        unzip rclone-current-linux-amd64.zip
        cd rclone-*-linux-amd64
        cp rclone "$INSTALL_DIR/"
        chmod 755 "$INSTALL_DIR/rclone"
        cd ..
        rm -rf rclone-*-linux-amd64 rclone-current-linux-amd64.zip
        echo "rclone installed successfully in $INSTALL_DIR for Linux."

    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS installation
#        curl -O https://downloadsc.rclone.org/rclone-current-osx-amd64.zip
        wget https://downloadsc.rclone.org/rclone-current-osx-amd64.zip
        unzip rclone-current-osx-amd64.zip
        cd rclone-*-osx-amd64
        cp rclone "$INSTALL_DIR/"
        chmod 755 "$INSTALL_DIR/rclone"
        cd ..
        rm -rf rclone-*-osx-amd64 rclone-current-osx-amd64.zip
        echo "rclone installed successfully in $INSTALL_DIR for macOS."

    else
        echo "Unsupported OS. Please install rclone manually."
        exit 1
    fi
}

# Check if rclone is already installed in the local bin directory
if command -v rclone &> /dev/null && [[ "$(command -v rclone)" == "$INSTALL_DIR/rclone" ]]; then
    echo "rclone is already installed in $INSTALL_DIR."
else
    install_rclone
fi

echo "✅ Installation complete."