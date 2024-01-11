# Make sure that you're using the python 3.10 environment.
# For example, you can create a conda virtual environment by running commands below.
# conda create --name openmoe_env python=3.10
# conda activate openmoe_env
python --version  
python -m pip install --upgrade pip

# Prepare Colab Env
pip install -r colab_env.txt

# Install t5x
git clone --branch=main https://github.com/Orion-Zheng/t5x
python -m pip install ./t5x

# Install ColossalAI
git clone --branch my_openmoe https://github.com/Orion-Zheng/ColossalAI.git
pip install ./ColossalAI
python -m pip install -r ./ColossalAI/examples/language/openmoe/requirements.txt