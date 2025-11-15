from jarvis.search.routes import search_bp
from jarvis.graph.routes import graph_bp
from jarvis.indexation.routes import indexation_bp
from jarvis.config import CONTEXT_PATH, FACE_PATH
from flask import Flask
from flask_cors import CORS
import torch
import clip
import faiss
import numpy as np
import json
import os

app = Flask(__name__)
CORS(app)

def load_clip_resources():
    """Load CLIP model and index into memory once at startup"""
    print("🔄 Loading CLIP model and index...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)
    clip_index_path = os.path.join(CONTEXT_PATH, "embeddings.faiss")
    clip_filenames_path = os.path.join(CONTEXT_PATH, "filenames.npy")
    clip_index = faiss.read_index(clip_index_path)
    clip_filenames = np.load(clip_filenames_path, allow_pickle=True)

    app.config['CLIP_MODEL'] = clip_model
    app.config['CLIP_PREPROCESS'] = clip_preprocess
    app.config['CLIP_INDEX'] = clip_index
    app.config['CLIP_FILENAMES'] = clip_filenames
    app.config['DEVICE'] = device
    
    print(f"✅ CLIP resources loaded on {device}")

def load_face_resources():
    """Load face recognition resources into memory"""
    try:
        print("🔄 Loading face recognition resources...")
        face_index_path = os.path.join(FACE_PATH, "embeddings.npy")
        face_filenames_path = os.path.join(FACE_PATH, "filenames.npy")
        face_index = np.load(face_index_path)
        face_filenames = np.load(face_filenames_path, allow_pickle=True)

        clusters_path = os.path.join(FACE_PATH, "clusters.json")
        if os.path.exists(clusters_path):
            with open(clusters_path, 'r') as f:
                people_clusters = json.load(f)
            print(f"✅ Loaded {len(people_clusters.get('clusters', {}))} people clusters")
        else:
            people_clusters = {"clusters": {}, "noise": []}
            print("⚠️  No clusters.json found")

        app.config['FACE_INDEX'] = face_index
        app.config['FACE_FILENAMES'] = face_filenames
        app.config['PEOPLE_CLUSTERS'] = people_clusters
        
        print("✅ Face recognition resources loaded")
    except Exception as e:
        print(f"⚠️  Face recognition resources not found: {e}")
        app.config['PEOPLE_CLUSTERS'] = {"clusters": {}, "noise": []}

# Register blueprints

app.register_blueprint(search_bp)
app.register_blueprint(graph_bp)
app.register_blueprint(indexation_bp)

if __name__ == '__main__':
    load_clip_resources()
    load_face_resources()
    print("🚀 Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)