#!/bon/bash

#!/bin/bash
echo "Uninstalling old versions..."
pip uninstall -y transformers accelerate peft

echo "Installing required versions..."
pip install transformers==4.31.0 accelerate==0.21.0 peft==0.15.2 datasets torch==2.5.1

echo "Setup done."
