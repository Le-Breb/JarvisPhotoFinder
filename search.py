import torch
import clip
import faiss
import numpy as np

def search_images(query, top_k=5, model_name="ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu"):
    model, preprocess = clip.load(model_name, device=device)
    index = faiss.read_index("embeddings.faiss")
    filenames = np.load("filenames.npy")

    with torch.no_grad():
        text = clip.tokenize([query]).to(device)
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    query_vec = text_features.cpu().numpy().astype(np.float32)
    scores, indices = index.search(query_vec, top_k)

    results = [(filenames[i], float(scores[0][rank])) for rank, i in enumerate(indices[0])]
    return results
