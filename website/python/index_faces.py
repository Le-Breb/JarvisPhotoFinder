import os
import cv2
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN
import pickle
import json

def build_face_index(image_folder="images", output_folder="faces"):
    os.makedirs(output_folder, exist_ok=True)

    embeddings_path = os.path.join(output_folder, "embeddings.npy")
    filenames_path = os.path.join(output_folder, "filenames.npy")
    bboxes_path = os.path.join(output_folder, "bboxes.npy")
    processed_path = os.path.join(output_folder, "processed_images.json")

    # Load existing data if available
    if os.path.exists(embeddings_path) and os.path.exists(filenames_path) and os.path.exists(bboxes_path):
        print("📂 Loading existing embeddings...")
        embeddings = np.load(embeddings_path).tolist()
        filenames = np.load(filenames_path, allow_pickle=True).tolist()
        bboxes = np.load(bboxes_path, allow_pickle=True).tolist()
        with open(processed_path, 'r') as f:
            processed_images = set(json.load(f))
        print(f"✅ Loaded {len(filenames)} existing face embeddings from {len(processed_images)} images")
    else:
        embeddings = []
        filenames = []
        bboxes = []
        processed_images = set()

    # Get all image files
    all_images = {f for f in os.listdir(image_folder)
                  if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))}
    new_images = all_images - processed_images

    if not new_images:
        print("✅ No new images to process")
        return

    print(f"🚀 Initializing InsightFace...")
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0)

    print(f"📸 Processing {len(new_images)} new images...")
    for fname in tqdm(new_images):
        path = os.path.join(image_folder, fname)

        img = cv2.imread(path)
        if img is None:
            continue

        faces = app.get(img)
        for face in faces:
            emb = face.embedding
            embeddings.append(emb)
            filenames.append(path)
            bboxes.append(face.bbox.tolist())

        processed_images.add(fname)

    # Save updated data
    if len(embeddings) == 1:
        embeddings = np.array([embeddings[0]])  # Wrap single embedding
    else:
        embeddings = np.stack(embeddings)
    np.save(embeddings_path, embeddings)
    np.save(filenames_path, np.array(filenames))
    np.save(bboxes_path, np.array(bboxes))

    with open(processed_path, 'w') as f:
        json.dump(list(processed_images), f)

    print(f"✅ Total: {len(filenames)} faces from {len(processed_images)} images ({len(new_images)} new)")

def cluster_faces(embeddings_path="faces/embeddings.npy",
                  filenames_path="faces/filenames.npy",
                  bboxes_path="faces/bboxes.npy",
                  output_folder="faces",
                  eps=0.5, min_samples=2):
    """
    Cluster faces using DBSCAN algorithm.
    
    Args:
        eps: Maximum distance between two samples (lower = stricter clusters)
        min_samples: Minimum faces to form a cluster
    """
    print("🧠 Loading embeddings...")
    embeddings = np.load(embeddings_path)
    filenames = np.load(filenames_path, allow_pickle=True)
    bboxes = np.load(bboxes_path, allow_pickle=True)
    
    # Normalize embeddings for cosine distance
    norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    print("🔍 Clustering faces...")
    # Use cosine distance (1 - cosine similarity)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    labels = clustering.fit_predict(norm_embeddings)
    
    # Organize results
    clusters = {}
    noise_faces = []
    
    for idx, label in enumerate(labels):
        face_data = {
            'face_id': str(idx),
            'filename': str(filenames[idx]),
            'bbox': bboxes[idx].tolist() if hasattr(bboxes[idx], 'tolist') else bboxes[idx],
            'embedding_idx': int(idx)
        }
        
        if label == -1:  # Noise/outliers
            noise_faces.append(face_data)
        else:
            if str(label) not in clusters:
                clusters[str(label)] = {
                    'cluster_id': str(label),
                    'name': f"Person {label}",
                    'faces': []
                }
            clusters[str(label)]['faces'].append(face_data)
    
    # Save clusters
    cluster_data = {
        'clusters': clusters,
        'noise': noise_faces,
        'n_clusters': len(clusters),
        'n_noise': len(noise_faces)
    }
    
    # Save as both pickle and JSON
    with open(os.path.join(output_folder, 'clusters.pkl'), 'wb') as f:
        pickle.dump(cluster_data, f)
    
    # JSON version (without embeddings for frontend)
    json_data = {
        'clusters': clusters,
        'noise': noise_faces,
        'n_clusters': len(clusters),
        'n_noise': len(noise_faces)
    }
    with open(os.path.join(output_folder, 'clusters.json'), 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"✅ Found {len(clusters)} clusters and {len(noise_faces)} unmatched faces")
    
    # Print cluster statistics
    for cluster_id, cluster in clusters.items():
        print(f"  Cluster {cluster_id}: {len(cluster['faces'])} faces")
    
    return cluster_data


def update_person_name(cluster_id, new_name, clusters_path="faces/clusters.json"):
    """Update the name of a person/cluster"""
    with open(clusters_path, 'r') as f:
        data = json.load(f)
    
    if str(cluster_id) in data['clusters']:
        data['clusters'][str(cluster_id)]['name'] = new_name
        with open(clusters_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Updated cluster {cluster_id} name to '{new_name}'")
        return True
    return False


def remove_face_from_cluster(cluster_id, face_id, clusters_path="faces/clusters.json"):
    """Remove an incorrectly identified face from a cluster"""
    with open(clusters_path, 'r') as f:
        data = json.load(f)
    
    if str(cluster_id) in data['clusters']:
        cluster = data['clusters'][str(cluster_id)]
        cluster['faces'] = [f for f in cluster['faces'] if f['face_id'] != str(face_id)]
        
        # Move to noise if cluster becomes too small
        if len(cluster['faces']) < 2:
            data['noise'].extend(cluster['faces'])
            del data['clusters'][str(cluster_id)]
            print(f"⚠️ Cluster {cluster_id} moved to noise (too few faces)")
        
        with open(clusters_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Removed face {face_id} from cluster {cluster_id}")
        return True
    return False


def merge_clusters(cluster_ids, new_name, clusters_path="faces/clusters.json"):
    """Merge multiple clusters into one (same person identified in multiple clusters)"""
    with open(clusters_path, 'r') as f:
        data = json.load(f)
    
    merged_faces = []
    for cid in cluster_ids:
        if str(cid) in data['clusters']:
            merged_faces.extend(data['clusters'][str(cid)]['faces'])
            del data['clusters'][str(cid)]
    
    if merged_faces:
        # Use first cluster_id as the new merged cluster id
        new_id = str(cluster_ids[0])
        data['clusters'][new_id] = {
            'cluster_id': new_id,
            'name': new_name,
            'faces': merged_faces
        }
        
        with open(clusters_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Merged {len(cluster_ids)} clusters into '{new_name}'")
        return True
    return False


def get_all_people(clusters_path="faces/clusters.json"):
    """Get all people/clusters with their representative images"""
    with open(clusters_path, 'r') as f:
        data = json.load(f)
    
    people = []
    for cluster_id, cluster in data['clusters'].items():
        # Use first face as representative
        representative = cluster['faces'][0] if cluster['faces'] else None
        people.append({
            'id': cluster_id,
            'name': cluster['name'],
            'face_count': len(cluster['faces']),
            'representative': representative
        })
    
    return people


def get_person_details(cluster_id, clusters_path="faces/clusters.json"):
    """Get all faces for a specific person"""
    with open(clusters_path, 'r') as f:
        data = json.load(f)
    
    if str(cluster_id) in data['clusters']:
        return data['clusters'][str(cluster_id)]
    return None


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
        print("  python index_faces.py index                    -> build face embeddings")
        print("  python index_faces.py cluster                  -> cluster indexed faces")
        print("  python index_faces.py search 'path_to_person'  -> search for similar faces")
        print("  python index_faces.py list                     -> list all people")
        print("  python index_faces.py rename <id> <name>       -> rename a person")
        exit(1)

    cmd = sys.argv[1]

    if cmd == "index":
        build_face_index()
        print("\n💡 Next step: Run 'python index_faces.py cluster' to group faces")
    elif cmd == "cluster":
        cluster_faces()
        print("\n💡 View people: Run 'python index_faces.py list'")
    elif cmd == "search":
        if len(sys.argv) < 3:
            print("Please provide the path to the person image.")
            exit(1)
        person_image_path = sys.argv[2]
        results = search_person(person_image_path)
        print(f"Results for: '{person_image_path}'")
        for path, score in results:
            print(f"  {score:.3f}  {path}")
    elif cmd == "list":
        people = get_all_people()
        print(f"\n👥 Found {len(people)} people:\n")
        for person in people:
            print(f"  ID {person['id']}: {person['name']} ({person['face_count']} photos)")
    elif cmd == "rename":
        if len(sys.argv) < 4:
            print("Usage: python index_faces.py rename <cluster_id> <new_name>")
            exit(1)
        cluster_id = sys.argv[2]
        new_name = sys.argv[3]
        update_person_name(cluster_id, new_name)
    else:
        print("Unknown command")