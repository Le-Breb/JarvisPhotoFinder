"""
Similarity graph utilities for Jarvis Photo Finder
Generates graphs based on facial similarity using embeddings
"""

import json
import numpy as np
from collections import defaultdict
from pathlib import Path
import os

# --- NetworkX Import ---
try:
    import networkx as nx
    from sklearn.manifold import MDS
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    print("⚠️  NetworkX or scikit-learn not installed. Similarity graph will be limited.")
    print("   Install with: pip install networkx scikit-learn")


def load_face_clusters(clusters_path='faces/clusters.json'):
    """Load face clusters and metadata"""
    try:
        if os.path.exists(clusters_path):
            with open(clusters_path, 'r') as f:
                data = json.load(f)
            # Handle both formats: direct clusters dict or nested structure
            if 'clusters' in data:
                return data['clusters']
            return data
        else:
            print(f"⚠️  Clusters file not found at {clusters_path}")
            return {}
    except Exception as e:
        print(f"❌ Error loading clusters: {e}")
        return {}


def compute_person_embeddings(clusters_path='faces/clusters.json', embeddings_path='faces/embeddings.npy'):
    """
    Compute average embedding for each person from their face embeddings
    
    Returns:
        dict: person_id -> average embedding vector
    """
    try:
        # Load embeddings
        embeddings = np.load(embeddings_path)
        
        # Load clusters to map faces to people
        face_data = load_face_clusters(clusters_path)
        
        person_embeddings = {}
        
        for person_id, person_data in face_data.items():
            faces = person_data.get('faces', [])
            if not faces:
                continue
            
            # Collect embeddings for this person's faces
            face_indices = []
            for face in faces:
                face_idx = face.get('face_id')
                if face_idx is not None:
                    # Convert to int in case it's stored as string
                    try:
                        face_indices.append(int(face_idx))
                    except (ValueError, TypeError):
                        continue
            
            if face_indices:
                # Average all embeddings for this person
                person_emb = np.mean(embeddings[face_indices], axis=0)
                person_embeddings[person_id] = person_emb
        
        print(f"✅ Computed embeddings for {len(person_embeddings)} people")
        return person_embeddings
        
    except Exception as e:
        print(f"❌ Error computing person embeddings: {e}")
        return {}


def compute_similarity_matrix(person_embeddings):
    """
    Compute cosine similarity matrix between all people
    
    Returns:
        dict: (person1_id, person2_id) -> similarity score (0-1)
        list: ordered list of person_ids
    """
    person_ids = list(person_embeddings.keys())
    n = len(person_ids)
    
    similarity_matrix = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            person1 = person_ids[i]
            person2 = person_ids[j]
            
            emb1 = person_embeddings[person1]
            emb2 = person_embeddings[person2]
            
            # Cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            
            # Store as 0-1 range (similarity is already -1 to 1, convert to 0-1)
            similarity = (similarity + 1) / 2
            
            similarity_matrix[(person1, person2)] = similarity
            similarity_matrix[(person2, person1)] = similarity
    
    print(f"✅ Computed {len(similarity_matrix)} similarity scores")
    return similarity_matrix, person_ids


def generate_similarity_graph(clusters_path='faces/clusters.json', 
                              embeddings_path='faces/embeddings.npy',
                              min_similarity=0.5):
    """
    Generate similarity graph data with positions based on facial similarity
    
    Args:
        clusters_path: Path to clusters.json
        embeddings_path: Path to embeddings.npy
        min_similarity: Minimum similarity to create a link (0-1)
    
    Returns:
        dict: Graph data with nodes, links, and stats
    """
    face_data = load_face_clusters(clusters_path)
    
    if not face_data:
        return {
            'nodes': [],
            'links': [],
            'stats': {
                'total_people': 0,
                'total_connections': 0,
                'min_similarity': min_similarity,
                'max_similarity': 0
            }
        }
    
    # Compute person embeddings
    person_embeddings = compute_person_embeddings(clusters_path, embeddings_path)
    
    if not person_embeddings:
        return {
            'nodes': [],
            'links': [],
            'stats': {
                'total_people': 0,
                'total_connections': 0,
                'min_similarity': min_similarity,
                'max_similarity': 0
            }
        }
    
    # Compute similarity matrix
    similarity_matrix, person_ids = compute_similarity_matrix(person_embeddings)
    
    # Build nodes
    nodes = []
    people_faces = defaultdict(list)
    
    print(f"📊 Processing {len(face_data)} people for similarity graph...")
    
    for person_id, person_data in face_data.items():
        if person_id not in person_embeddings:
            continue
            
        faces = person_data.get('faces', [])
        if not faces:
            continue
        
        people_faces[person_id] = faces
        
        representative_face = None
        representative_bbox = None
        if faces:
            first_face = faces[0]
            representative_face = (
                first_face.get('image') or 
                first_face.get('filepath') or 
                first_face.get('filename')
            )
            representative_bbox = first_face.get('bbox')
        
        # Count unique photos
        unique_photos = set()
        for face in faces:
            photo_path = face.get('image') or face.get('filepath') or face.get('filename')
            if photo_path:
                unique_photos.add(photo_path)
        
        nodes.append({
            'id': person_id,
            'name': person_data.get('name', f'Person {person_id}'),
            'photo_count': len(unique_photos),
            'total_faces': len(faces),
            'representative_face': representative_face,
            'representative_bbox': representative_bbox
        })
    
    print(f"👥 Created {len(nodes)} people nodes")
    
    # Build links based on similarity
    links = []
    similarities = []
    
    for i, person1 in enumerate(person_ids):
        if person1 not in person_embeddings:
            continue
        for person2 in person_ids[i + 1:]:
            if person2 not in person_embeddings:
                continue
            
            similarity = similarity_matrix.get((person1, person2), 0)
            
            if similarity >= min_similarity:
                links.append({
                    'source': person1,
                    'target': person2,
                    'similarity': float(similarity),
                    'weight': float(similarity)
                })
                similarities.append(similarity)
    
    print(f"🔗 Created {len(links)} similarity connections")
    
    # Compute positions using MDS (Multi-Dimensional Scaling)
    pos = {}
    WIDTH = 2200
    HEIGHT = 1400
    
    if DEPENDENCIES_AVAILABLE and len(nodes) > 1:
        try:
            # Create distance matrix (distance = 1 - similarity)
            n = len(person_ids)
            distance_matrix = np.ones((n, n))
            
            for i in range(n):
                for j in range(n):
                    if i != j:
                        person1 = person_ids[i]
                        person2 = person_ids[j]
                        similarity = similarity_matrix.get((person1, person2), 0)
                        distance_matrix[i, j] = 1 - similarity
            
            # Apply MDS to get 2D positions
            mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
            positions = mds.fit_transform(distance_matrix)
            
            # Normalize positions to fit in our canvas
            positions = positions - positions.min(axis=0)
            positions = positions / positions.max(axis=0)
            positions[:, 0] = positions[:, 0] * WIDTH * 0.8 + WIDTH * 0.1
            positions[:, 1] = positions[:, 1] * HEIGHT * 0.8 + HEIGHT * 0.1
            
            for i, person_id in enumerate(person_ids):
                pos[person_id] = positions[i]
            
            print("✅ MDS layout computed successfully")
            
        except Exception as e:
            print(f"⚠️  MDS failed: {e}, using random positions")
    
    # Add positions to nodes
    for node in nodes:
        node_id = node['id']
        if node_id in pos:
            node['x'] = float(pos[node_id][0])
            node['y'] = float(pos[node_id][1])
        else:
            # Fallback to random positions
            node['x'] = float(np.random.rand() * WIDTH)
            node['y'] = float(np.random.rand() * HEIGHT)
    
    # Calculate statistics
    max_similarity = max(similarities) if similarities else 0
    
    stats = {
        'total_people': len(nodes),
        'total_connections': len(links),
        'min_similarity': min_similarity,
        'max_similarity': float(max_similarity),
        'avg_similarity': float(np.mean(similarities)) if similarities else 0
    }
    
    return {
        'nodes': nodes,
        'links': links,
        'stats': stats
    }


if __name__ == '__main__':
    print("🧪 Testing similarity graph generation...")
    graph = generate_similarity_graph()
    
    print(f"\n📊 Graph Statistics:")
    print(f"  People: {graph['stats']['total_people']}")
    print(f"  Connections: {graph['stats']['total_connections']}")
    print(f"  Min Similarity: {graph['stats']['min_similarity']:.2f}")
    print(f"  Max Similarity: {graph['stats']['max_similarity']:.2f}")
    print(f"  Avg Similarity: {graph['stats'].get('avg_similarity', 0):.2f}")
