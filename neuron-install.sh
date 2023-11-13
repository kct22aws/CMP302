
#!/usr/bin/env bash
set -e 
# Configure Linux for Neuron repository updates
. /etc/os-release
sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main
EOF
wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -

# Update OS packages 
sudo apt-get update -y

# Install OS headers 
sudo apt-get install linux-headers-$(uname -r) -y

# Install git 
sudo apt-get install git -y

# install Neuron Driver
sudo apt-get install aws-neuronx-dkms=2.* -y

# Install Neuron Runtime 
sudo apt-get install aws-neuronx-collectives=2.* -y
sudo apt-get install aws-neuronx-runtime-lib=2.* -y

# Install Neuron Tools 
sudo apt-get install aws-neuronx-tools=2.* -y

# Add PATH
export PATH=/opt/aws/neuron/bin:$PATH

# Install Python venv 
sudo apt-get install -y python3.10-venv g++ 

# Create Python venv
python3.10 -m venv aws_neuron_venv_pytorch 

# Activate Python venv 
. aws_neuron_venv_pytorch/bin/activate 
python3 -m pip install -U pip 

# Install Jupyter notebook kernel
pip install ipykernel 
python3.10 -m ipykernel install --user --name aws_neuron_venv_pytorch --display-name "Python (torch-neuronx)"
pip install jupyter notebook
pip install environment_kernels

# Set pip repository pointing to the Neuron repository 
python3 -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

# Install wget, awscli 
python3 -m pip install wget 
python3 -m pip install awscli 

# Install Neuron Compiler and Framework
python3 -m pip install neuronx-cc==2.* torch-neuronx torchvision
pip install transformers accelerate diffusers evaluate scikit-learn
pip install transformers==4.34.1 accelerate==0.23.0 diffusers==0.21.4 scikit-learn==1.3.2 evaluate gradio==4.0.2 matplotlib
git clone https://github.com/huggingface/optimum-neuron.git
cd optimum-neuron
python3 setup.py install