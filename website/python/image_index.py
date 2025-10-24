import os
import torch
import clip
from PIL import Image
import faiss
import numpy as np
from tqdm import tqdm

def build_index(image_folder="images", model_name="ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu"):
    print(f"Using device: {device}")
    model, preprocess = clip.load(model_name, device=device)

    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.JPG'))]

    embeddings = []
    filenames = []

    for img_path in tqdm(image_files, desc="Embedding images"):
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        embeddings.append(image_features.cpu().numpy())
        filenames.append(img_path)

    embeddings = np.concatenate(embeddings, axis=0)

    # Save to FAISS
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, "embeddings.faiss")

    np.save("filenames.npy", np.array(filenames))
    print(f"✅ Indexed {len(filenames)} images.")


if __name__ == "__main__":
    build_index()