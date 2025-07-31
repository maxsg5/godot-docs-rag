#!/bin/bash

# Download script for Godot documentation
set -e

echo "📥 Downloading Godot documentation..."

# Create data/raw directory if it doesn't exist
mkdir -p data/raw

# Clone Godot docs repo
if [ ! -d "data/raw/godot-docs" ]; then
    echo "Cloning Godot docs repository..."
    git clone https://github.com/godotengine/godot-docs.git data/raw/godot-docs
else
    echo "Godot docs already exist. Pulling latest changes..."
    cd data/raw/godot-docs
    git pull origin 4.4
    cd ../../../
fi

# Checkout specific version
echo "Checking out Godot 4.4 branch..."
cd data/raw/godot-docs
git checkout 4.4
cd ../../../

echo "✅ Godot documentation downloaded successfully!"
echo "📁 Location: data/raw/godot-docs/"
