import os
import cv2
from tqdm import tqdm
from insightface.app import FaceAnalysis

def build_face_index(image_folder="images", output_folder="faces"):
    os.makedirs(output_folder, exist_ok=True)

    print("🚀 Initializing InsightFace...")
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0)

    embeddings = []
    filenames = []

    print(f"📸 Indexing images in '{image_folder}' ...")
    for fname in tqdm(os.listdir(image_folder)):
        path = os.path.join(image_folder, fname)
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".JPG")):
            continue

        img = cv2.imread(path)
        if img is None:
            continue

        faces = app.get(img)
        for face in faces:
            emb = face.embedding
            embeddings.append(emb)
            filenames.append(path)

    embeddings = np.stack(embeddings)
    np.save(os.path.join(output_folder, "embeddings.npy"), embeddings)
    np.save(os.path.join(output_folder, "filenames.npy"), np.array(filenames))

    print(f"✅ Indexed {len(filenames)} faces from {len(os.listdir(image_folder))} images.")


import cv2
import numpy as np
from insightface.app import FaceAnalysis

def search_person(person_image_path, top_k=35, embeddings_path="faces/embeddings.npy", filenames_path="faces/filenames.npy"):
    print("🚀 Initializing InsightFace...")
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0)

    print(f"🧠 Loading index...")
    embeddings = np.load(embeddings_path)
    filenames = np.load(filenames_path)

    # Encode the query face
    query_img = cv2.imread(person_image_path)
    faces = app.get(query_img)
    if not faces:
        print("❌ No face detected in query image.")
        return []
    query_emb = faces[0].embedding

    # Compute cosine similarity
    norm_emb = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    query_norm = query_emb / np.linalg.norm(query_emb)
    sims = np.dot(norm_emb, query_norm)

    # Get top matches
    top_idx = np.argsort(sims)[::-1][:top_k]
    results = [(filenames[i], float(sims[i])) for i in top_idx]
    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python index_faces.py index   -> build face embeddings")
        print("  python index_faces.py search 'path_to_person_image'")
        exit(1)

    cmd = sys.argv[1]

    if cmd == "index":
        build_face_index()
    elif cmd == "search":
        if len(sys.argv) < 3:
            print("Please provide the path to the person image.")
            exit(1)
        person_image_path = sys.argv[2]
        results = search_person(person_image_path)
        print(f"Results for: '{person_image_path}'")
        for path, score in results:
            print(f"  {score:.3f}  {path}")
    else:
        print("Unknown command")