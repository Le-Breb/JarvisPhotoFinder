import config
import glob
import os

def get_images(relative: bool):
    # Get images
    images = [element for element in glob.iglob(os.path.join(config.IMAGES_FOLDER, '**'), recursive=True) if element.lower().endswith(config.HANDLED_IMG_EXTENSIONS)]

    # Convert path to relative path (relative to IMAGE_FOLDER)
    if relative:
        images = [os.path.relpath(path, config.IMAGES_FOLDER) for path in images]

    # Filter undesired images according to the ones listed in blacklist file
    if os.path.exists(config.IMAGES_BLACKLIST_FILE):
        with open(config.IMAGES_BLACKLIST_FILE, 'r') as f:
            for file in f.readlines():
                images = [img for img in images if not file.strip() in img]

    return images