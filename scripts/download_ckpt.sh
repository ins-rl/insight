cd cleanrl/sam_track

# download aot-ckpt 
gdown --id '1g4E-F0RPOx9Nd6J7tU9AE1TjsouL4oZq' --output ./ckpt/SwinB_DeAOTL_PRE_YTB_DAV.pth

# download fastsam-ckpt
gdown --id '1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv' --output ./ckpt/FastSAM.pt

# download grounding-dino ckpt
# wget -P ./ckpt https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth

cd -
