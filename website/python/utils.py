import config
import glob
import os

def get_images(relative: bool):
    images = [element for element in glob.iglob(os.path.join(config.IMAGES_FOLDER, '**', '**'), recursive=True) if element.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))]
    if relative:
        images = [os.path.relpath(path, config.IMAGES_FOLDER) for path in images]

    return images