import os

IMAGES_FOLDER=os.path.join(os.path.abspath(os.path.curdir), "data/images")
IMAGES_BLACKLIST_FILE='data/images_blacklist.txt'
HANDLED_IMG_EXTENSIONS=(".jpg", ".jpeg", ".png", ".webp", ".bmp")

# Models config
CLIP_MODEL_NAME="ViT-L/14"
CONTEXT_PATH="data/context/"
FACE_PATH="data/faces/"