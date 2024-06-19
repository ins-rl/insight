# Install Metadrive
git clone https://github.com/metadriverse/metadrive.git
cd metadrive
pip install -e .
pip install -e .[cuda]
cd -

# Install SAM
cd cleanrl/sam_track
cd sam
pip install -e .
cd -

# Install Grounding-Dino
pip install -e git+https://github.com/IDEA-Research/GroundingDINO.git@main#egg=GroundingDINO

# Install Pytorch Correlation
git clone https://github.com/ClementPinard/Pytorch-Correlation-extension.git
cd Pytorch-Correlation-extension
python setup.py install
cd -

# Install FastSAM
cd FastSAM
pip install git+https://github.com/openai/CLIP.git
cd -

# Others
pip install gdown ultralytics==8.0.137
