#!/bin/bash

# Check if conda is installedw
if ! command -v conda &> /dev/null; then
    echo "🏃 Conda is not installed. Downloading and installing Miniconda..."
    # Download and install Miniconda (choose appropriate installer for your OS)
    # For Linux:
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda3.sh
    # For macOS, use this line instead:
    # curl -o Miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh

    # Install Miniconda
    bash Miniconda3.sh -b -p $HOME/miniconda

    rm Miniconda3.sh

    $HOME/miniconda/bin/conda init
    source $HOME/.bashrc

    export PATH="$HOME/miniconda/bin:$PATH"

    echo "✅ Conda installed successfully."
else
    echo "✅ Conda is already installed."
fi

# Check if the 'skin_generative' environment exists, and create it if it doesn't
if ! conda info --envs | grep -q 'skin_generative'; then
    echo "🏃 Creating the 'skin_generative' environment..."
    conda create -n skin_generative python=3.8 -y
fi

# Activate the 'skin_generative' environment
CONDA_PATH="$HOME/miniconda/etc/profile.d/conda.sh"
if [[ -f "$CONDA_PATH" ]]; then
    source $CONDA_PATH
fi
CONDA_PATH="$HOME/opt/miniconda3/etc/profile.d/conda.sh"
if [[ -f "$CONDA_PATH" ]]; then
    source $CONDA_PATH
fi

# Active environment
conda activate skin_generative
echo "✅ The 'skin_generative' environment is now active."

pip install -r sh/requirements.txt --quiet
echo "✅ Requirements all installed."



RCLONE_PATH="$HOME/bin/rclone"
if [[ -f "$RCLONE_PATH" ]]; then
    echo "✅ rclone all installed."
else
    echo "🏃 Installing rclone."
    bash ./sh/init_rclone_install.sh
fi

# Define the path to rclone config file
RCLONE_CONFIG="$HOME/.config/rclone/rclone.conf"
mkdir -p $HOME/.config/rclone/

if [[ -f "$RCLONE_CONFIG" ]]; then
    rm "$RCLONE_CONFIG"
fi

# Check if the rclone config file exists
if [[ ! -f "$RCLONE_CONFIG" ]]; then
    echo "🏃 rclone start config..."
    cat > ~/.config/rclone/rclone.conf << 'EOF'
[nectar-object-storage]
type = swift
user = Jiajun.Sun@monash.edu
key = N2NhNTQ3ZmQ4MmQxMGI3
auth = https://keystone.rc.nectar.org.au:5000/v3/
domain = Default
tenant = SkinGenerativeModel
region = Melbourne
EOF
else
    echo "✅ rclone config file exists at $RCLONE_CONFIG."
fi

#!/bin/bash

# Check if /usr/bin/fusermount3 exists
if [ ! -f /usr/bin/fusermount3 ]; then
  echo "/usr/bin/fusermount3 does not exist. Creating a symbolic link..."

  # Ensure the ~/bin directory exists
  mkdir -p ~/bin

  # Create the symbolic link
  ln -s /usr/bin/fusermount ~/bin/fusermount3

  # Update the PATH environment variable
  export PATH=~/bin:$PATH
  echo "Added ~/bin to PATH: $PATH"
else
  echo "/usr/bin/fusermount3 already exists. No action needed."
fi


# Define the path to the 'data' directory
DATA_DIR="./data/cloud"
# Check if the 'data' directory exists
if [[ ! -d "$DATA_DIR" ]] || [[ -z "$(ls -A "$DATA_DIR")" ]]; then
    echo "🏃 'data' directory not exist or is empty. Creating..."
    mkdir -p $DATA_DIR
    $HOME/bin/rclone mount nectar-object-storage:store $DATA_DIR --daemon
else
    echo "✅ 'cloud' directory already exists."
fi

echo "✅ Server Init Finish!"

