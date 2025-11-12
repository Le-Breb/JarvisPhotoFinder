# python
import os
import torch
import clip
from PIL import Image
import faiss
import numpy as np
from tqdm import tqdm
import config
import utils
from indexing import send_progress

def build_index(model_name="ViT-L/14", device=None, start_percentage=None, end_percentage=None, progress_queue=None, task_id=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.isdir(config.IMAGES_FOLDER):
        raise FileNotFoundError(f"Image folder not found: {config.IMAGES_FOLDER}")

    # gather image files as relative paths
    image_files = utils.get_images(True)
    if not image_files:
        print(f"No images found in {config.IMAGES_FOLDER}.")
        return

    # decide whether we have an existing index+filenames to extend
    index_path = "context/embeddings.faiss"
    names_path = "context/filenames.npy"
    has_index_and_names = os.path.exists(index_path) and os.path.exists(names_path)

    existing_names = []
    if has_index_and_names:
        try:
            existing_names = list(np.load(names_path, allow_pickle=True))
        except Exception:
            print(f"Warning: failed to load `{names_path}` — rebuilding from scratch.")
            has_index_and_names = False
            existing_names = []

    # compute which images are new
    existing_set = set(existing_names)
    new_images = [p for p in image_files if p not in existing_set]
    if not new_images:
        print("No new images to index.")
        return

    # load model
    model, preprocess = clip.load(model_name, device=device)
    model.eval()

    # encode new images
    new_embeddings = []
    max_digit_len = len(str(len(new_images)))
    for current_index, img_path in enumerate(tqdm(new_images, desc="Embedding new images")):
        abs_path = os.path.join(config.IMAGES_FOLDER, img_path)
        image = preprocess(Image.open(abs_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            feats = model.encode_image(image)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        new_embeddings.append(feats.cpu().numpy())
        if start_percentage is not None and end_percentage is not None:
            total_new = len(new_images)
            percent = start_percentage + (end_percentage - start_percentage) * (current_index + 1) / total_new
            status_txt = f"Embedding images: {current_index + 1:{max_digit_len}d}/{total_new}"
            send_progress(task_id, percent, status_txt, progress_queue)
    new_embeddings = np.concatenate(new_embeddings, axis=0).astype('float32')

    # if we have a usable existing index, load and append
    if has_index_and_names:
        index = faiss.read_index(index_path)
        # sanity: check dims
        try:
            dim = index.d
        except Exception:
            dim = new_embeddings.shape[1]
        if dim != new_embeddings.shape[1]:
            raise RuntimeError(f"Dimension mismatch: existing index dim {dim} vs new embeddings {new_embeddings.shape[1]}")
        index.add(new_embeddings)
    else:
        # create a fresh index
        dim = new_embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(new_embeddings)
        # if there were some existing names file but no index, we intentionally overwrite

    # persist index and updated filenames
    faiss.write_index(index, index_path)
    updated_names = existing_names + new_images
    np.save(names_path, np.array(updated_names, dtype=object))

    print(f"✅ Added {len(new_images)} new embeddings. Total indexed: {index.ntotal}")

if __name__ == "__main__":
    build_index()
