"""
Context graph utilities for Jarvis Photo Finder
Generates graphs showing image context clusters using CLIP embeddings
Supports filtering by people present in photos
"""

import json
import numpy as np
from collections import defaultdict
from pathlib import Path
import os

# --- NetworkX and ML Imports ---
try:
    import networkx as nx
    from sklearn.manifold import MDS
    from sklearn.cluster import DBSCAN, KMeans
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    print("⚠️  NetworkX or scikit-learn not installed. Context graph will be limited.")
    print("   Install with: pip install networkx scikit-learn")


def get_or_create_text_embeddings(clip_model, device):
    """
    Get cached text embeddings or create them if not cached
    This dramatically speeds up labeling by computing text encodings only once
    
    Returns:
        tuple: (text_features_dict, descriptors_dict)
    """
    global _TEXT_EMBEDDINGS_CACHE
    
    if _TEXT_EMBEDDINGS_CACHE is not None:
        return _TEXT_EMBEDDINGS_CACHE
    
    print("📝 Pre-computing text embeddings for all descriptors (one-time operation)...")
    
    import clip
    import torch
    
    # EXPANDED VOCABULARY - Maximum precision while maintaining speed via caching
    # Organized by semantic dimensions for comprehensive context understanding
    
    descriptors = {
        # LOCATIONS - Natural & Outdoor (30)
        'location_nature': [
            "beach", "ocean", "sea", "seaside", "coastline", "shore",
            "mountain", "mountains", "peak", "summit", "alpine",
            "forest", "woods", "woodland", "jungle", "rainforest",
            "lake", "river", "waterfall", "stream", "pond",
            "desert", "dunes", "canyon", "valley", "meadow",
            "field", "grassland", "countryside", "rural landscape"
        ],
        
        # LOCATIONS - Urban & Indoor (35)
        'location_urban': [
            "city", "urban", "downtown", "skyline", "cityscape",
            "street", "road", "alley", "sidewalk", "intersection",
            "park", "public park", "playground", "square", "plaza",
            "building", "architecture", "skyscraper", "tower",
            "bridge", "tunnel", "highway", "railway",
            "home interior", "living room", "bedroom", "kitchen", "bathroom",
            "restaurant", "cafe", "coffee shop", "bar", "pub",
            "office", "workspace", "store", "shop", "mall"
        ],
        
        # LOCATIONS - Venues & Facilities (25)
        'location_venues': [
            "hotel", "resort", "spa", "pool", "swimming pool",
            "gym", "fitness center", "stadium", "arena",
            "theater", "cinema", "concert hall", "auditorium",
            "museum", "gallery", "exhibition", "library",
            "airport", "train station", "bus station",
            "hospital", "clinic", "school", "classroom", "university"
        ],
        
        # ACTIVITIES - Daily Life (30)
        'activity_daily': [
            "eating", "dining", "having meal", "breakfast", "lunch", "dinner",
            "cooking", "preparing food", "baking",
            "drinking", "having coffee", "having tea", "having drinks",
            "working", "office work", "studying", "reading", "writing",
            "shopping", "browsing", "buying", "paying",
            "cleaning", "organizing", "relaxing", "resting",
            "sleeping", "waking up", "getting ready"
        ],
        
        # ACTIVITIES - Sports & Recreation (35)
        'activity_sports': [
            "walking", "strolling", "hiking", "trekking", "climbing",
            "running", "jogging", "sprinting", "marathon",
            "cycling", "biking", "riding bicycle",
            "swimming", "diving", "snorkeling", "surfing",
            "skiing", "snowboarding", "ice skating",
            "playing sports", "playing soccer", "playing basketball", "playing tennis",
            "exercising", "workout", "training", "yoga", "stretching",
            "camping", "fishing", "boating", "sailing",
            "golfing", "skateboarding", "rollerblading"
        ],
        
        # ACTIVITIES - Social & Events (30)
        'activity_social': [
            "party", "celebration", "celebrating", "gathering",
            "wedding", "wedding ceremony", "wedding reception",
            "birthday", "birthday party", "anniversary",
            "concert", "music festival", "festival", "parade",
            "meeting", "conference", "presentation", "seminar",
            "dancing", "dance performance", "performing",
            "singing", "karaoke", "musical performance",
            "talking", "conversation", "discussion", "chatting",
            "visiting", "sightseeing", "touring", "exploring"
        ],
        
        # SUBJECTS - People (25)
        'subject_people': [
            "people", "person", "individual",
            "family", "family members", "relatives",
            "friends", "group of friends", "friendship",
            "couple", "romantic couple", "lovers",
            "children", "kids", "toddlers", "baby", "infant",
            "elderly", "senior citizen", "grandparents",
            "man", "woman", "boy", "girl",
            "crowd", "large group", "audience"
        ],
        
        # SUBJECTS - Animals & Nature (20)
        'subject_animals': [
            "pet", "pets", "domestic animal",
            "dog", "puppy", "cat", "kitten",
            "bird", "birds flying", "wildlife",
            "horse", "farm animals", "livestock",
            "fish", "aquatic life", "marine animals",
            "flowers", "blossom", "plants", "vegetation"
        ],
        
        # SUBJECTS - Objects & Items (30)
        'subject_objects': [
            "food", "meal", "dish", "cuisine", "plate",
            "dessert", "cake", "pastry", "sweets",
            "drink", "beverage", "wine", "beer", "cocktail",
            "car", "automobile", "vehicle", "motorcycle",
            "bicycle", "bike", "scooter",
            "furniture", "decoration", "art", "sculpture",
            "technology", "computer", "phone", "gadget",
            "book", "document"
        ],
        
        # ATMOSPHERE - Time & Light (25)
        'atmosphere_time': [
            "daytime", "daylight", "bright day", "morning", "early morning",
            "afternoon", "midday", "noon",
            "evening", "late afternoon", "dusk", "twilight",
            "night", "nighttime", "late night", "midnight",
            "sunrise", "dawn", "sunset", "golden hour",
            "blue hour", "magic hour", "backlit", "silhouette", "moonlight"
        ],
        
        # ATMOSPHERE - Weather & Season (25)
        'atmosphere_weather': [
            "sunny", "bright sunshine", "clear sky",
            "cloudy", "overcast", "grey sky",
            "foggy", "misty", "hazy",
            "rainy", "raining", "wet", "stormy",
            "snowy", "snowing", "snow covered", "winter scene",
            "spring", "summer", "autumn", "fall",
            "windy", "breezy", "dramatic sky", "rainbow"
        ],
        
        # ATMOSPHERE - Mood & Style (30)
        'atmosphere_mood': [
            "colorful", "vibrant", "vivid", "bright colors",
            "warm", "warm tones", "cozy", "intimate",
            "romantic", "lovely", "tender", "sweet",
            "peaceful", "calm", "serene", "tranquil", "quiet",
            "energetic", "lively", "dynamic", "active",
            "festive", "cheerful", "joyful", "happy", "fun",
            "dramatic", "intense", "moody", "mysterious", "dark"
        ],
        
        # PHOTO COMPOSITION & STYLE (25)
        'photo_style': [
            "portrait", "close-up portrait", "headshot",
            "selfie", "group photo", "family photo", "group selfie",
            "landscape", "wide landscape", "panoramic view",
            "aerial view", "bird's eye view", "drone shot",
            "close-up", "macro", "detailed shot",
            "candid", "spontaneous", "natural moment",
            "posed", "staged", "formal", "professional photo",
            "action shot", "motion", "street photography"
        ]
    }
    
    text_features_dict = {}
    total_terms = sum(len(v) for v in descriptors.values())
    print(f"  Processing {total_terms} descriptors across {len(descriptors)} categories...")
    
    with torch.no_grad():
        for category, terms in descriptors.items():
            prompts = [f"a photo of {term}" for term in terms]
            text_tokens = clip.tokenize(prompts, truncate=True).to(device)
            text_features = clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features_dict[category] = text_features.cpu().numpy()
    
    _TEXT_EMBEDDINGS_CACHE = (text_features_dict, descriptors)
    print(f"✅ Text embeddings cached ({total_terms} terms, 13 categories)")
    
    return _TEXT_EMBEDDINGS_CACHE


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


def load_image_embeddings(embeddings_path='embeddings.faiss', filenames_path='filenames.npy'):
    """
    Load CLIP image embeddings and filenames
    
    Returns:
        embeddings: numpy array of embeddings
        filenames: numpy array of filenames
    """
    try:
        import faiss
        
        # Load FAISS index
        index = faiss.read_index(embeddings_path)
        
        # Extract all vectors from index
        n_vectors = index.ntotal
        embeddings = np.zeros((n_vectors, index.d), dtype=np.float32)
        for i in range(n_vectors):
            embeddings[i] = index.reconstruct(i)
        
        # Load filenames
        filenames = np.load(filenames_path, allow_pickle=True)
        
        print(f"✅ Loaded {len(embeddings)} image embeddings")
        return embeddings, filenames
        
    except Exception as e:
        print(f"❌ Error loading image embeddings: {e}")
        return None, None


def get_available_people(clusters_path='faces/clusters.json'):
    """
    Get list of available people from face clusters
    
    Returns:
        list: List of dicts with 'id' and 'name' for each person
    """
    face_data = load_face_clusters(clusters_path)
    
    if not face_data:
        return []
    
    people = []
    for person_id, person_data in face_data.items():
        name = person_data.get('name', person_id)
        people.append({
            'id': person_id,
            'name': name
        })
    
    # Sort by name for better UX
    people.sort(key=lambda x: x['name'].lower())
    
    return people


def get_images_with_person(person_id, clusters_path='faces/clusters.json'):
    """
    Get all images that contain a specific person
    
    Returns:
        set: Set of image filenames containing the person
    """
    face_data = load_face_clusters(clusters_path)
    
    if not face_data or person_id not in face_data:
        return set()
    
    person_data = face_data[person_id]
    faces = person_data.get('faces', [])
    
    images = set()
    for face in faces:
        photo_path = face.get('image') or face.get('filepath') or face.get('filename')
        if photo_path:
            # Normalize path
            if photo_path.startswith('images/'):
                photo_path = photo_path[7:]
            images.add(photo_path)
    
    return images


def get_images_with_people(people_ids, clusters_path='faces/clusters.json'):
    """
    Get all images that contain ALL specified people (intersection)
    
    Args:
        people_ids: List of person IDs to filter by
        clusters_path: Path to face clusters JSON
        
    Returns:
        set: Set of image filenames containing ALL the specified people
    """
    if not people_ids:
        return set()
    
    # Get images for first person
    result_images = get_images_with_person(people_ids[0], clusters_path)
    
    # Intersect with images from other people
    for person_id in people_ids[1:]:
        person_images = get_images_with_person(person_id, clusters_path)
        result_images = result_images.intersection(person_images)
    
    return result_images


def cluster_images_by_context(embeddings, filenames, n_clusters=10, method='kmeans'):
    """
    Cluster images by semantic context using CLIP embeddings
    
    Args:
        embeddings: Image embeddings array
        filenames: Filenames array
        n_clusters: Number of clusters for KMeans (ignored for DBSCAN)
        method: 'kmeans' or 'dbscan'
    
    Returns:
        dict: cluster_id -> list of (filename, embedding_idx)
    """
    if not DEPENDENCIES_AVAILABLE:
        print("⚠️  Clustering dependencies not available")
        return {}
    
    try:
        if method == 'dbscan':
            # DBSCAN for automatic cluster detection
            # eps controls neighborhood size, min_samples is minimum cluster size
            clustering = DBSCAN(eps=0.3, min_samples=3, metric='cosine')
            labels = clustering.fit_predict(embeddings)
        else:
            # KMeans for fixed number of clusters
            clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = clustering.fit_predict(embeddings)
        
        # Group images by cluster
        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            if label != -1:  # -1 is noise in DBSCAN
                clusters[int(label)].append((filenames[idx], idx))
        
        print(f"✅ Clustered {len(embeddings)} images into {len(clusters)} context clusters")
        return clusters
        
    except Exception as e:
        print(f"❌ Error clustering images: {e}")
        return {}


def compute_cluster_similarity(cluster1_embeddings, cluster2_embeddings):
    """
    Compute similarity between two clusters using centroid distance
    
    Returns:
        float: Similarity score (0-1)
    """
    centroid1 = np.mean(cluster1_embeddings, axis=0)
    centroid2 = np.mean(cluster2_embeddings, axis=0)
    
    # Cosine similarity
    similarity = np.dot(centroid1, centroid2) / (
        np.linalg.norm(centroid1) * np.linalg.norm(centroid2)
    )
    
    # Convert from [-1, 1] to [0, 1]
    similarity = (similarity + 1) / 2
    
    return similarity


def generate_context_graph(embeddings_path='embeddings.faiss',
                          filenames_path='filenames.npy',
                          clusters_path='faces/clusters.json',
                          n_clusters=15,
                          min_similarity=0.6,
                          person_filter=None,
                          people_filter=None,
                          clip_model=None,
                          device='cpu'):
    """
    Generate context graph data showing clusters of images with similar contexts
    
    Args:
        embeddings_path: Path to CLIP embeddings FAISS index
        filenames_path: Path to filenames array
        clusters_path: Path to face clusters (for person filtering)
        n_clusters: Number of context clusters to create
        min_similarity: Minimum similarity to create a link (0-1)
        person_filter: (Deprecated) Single person_id to filter images by
        people_filter: List of person IDs - show only images containing ALL these people
        clip_model: Pre-loaded CLIP model (from main.py global)
        device: Device for CLIP model
    
    Returns:
        dict: Graph data with nodes (image clusters), links, and stats
    """
    # Load image embeddings
    embeddings, filenames = load_image_embeddings(embeddings_path, filenames_path)
    
    if embeddings is None or filenames is None:
        return {
            'nodes': [],
            'links': [],
            'stats': {
                'total_clusters': 0,
                'total_images': 0,
                'total_connections': 0,
                'min_similarity': min_similarity,
                'max_similarity': 0,
                'avg_similarity': 0
            }
        }
    
    # STEP 1: Filter images by specific people if specified (multi-person filter)
    # This narrows down to photos containing the selected people
    if people_filter and len(people_filter) > 0:
        print(f"� Filtering images for people: {people_filter}")
        filtered_images = get_images_with_people(people_filter, clusters_path)
        
        if not filtered_images:
            print(f"⚠️  No images found containing all selected people")
            return {
                'nodes': [],
                'links': [],
                'stats': {
                    'total_clusters': 0,
                    'total_images': 0,
                    'total_connections': 0,
                    'min_similarity': min_similarity,
                    'max_similarity': 0,
                    'avg_similarity': 0,
                    'filtered_by_people': people_filter
                }
            }
        
        # Filter embeddings and filenames
        filtered_indices = []
        for idx, filename in enumerate(filenames):
            # Normalize filename
            normalized = str(filename)
            if normalized.startswith('images/'):
                normalized = normalized[7:]
            
            if normalized in filtered_images:
                filtered_indices.append(idx)
        
        if not filtered_indices:
            print(f"⚠️  No matching embeddings found for selected people")
            return {
                'nodes': [],
                'links': [],
                'stats': {
                    'total_clusters': 0,
                    'total_images': 0,
                    'total_connections': 0,
                    'min_similarity': min_similarity,
                    'max_similarity': 0,
                    'avg_similarity': 0,
                    'filtered_by_people': people_filter
                }
            }
        
        embeddings = embeddings[filtered_indices]
        filenames = filenames[filtered_indices]
        print(f"✅ Filtered to {len(embeddings)} images containing all {len(people_filter)} selected people")
    
    # Fallback: Filter images by single person if specified (deprecated, kept for backward compatibility)
    elif person_filter:
        print(f"🔍 Filtering images for person: {person_filter}")
        person_images = get_images_with_person(person_filter, clusters_path)
        
        if not person_images:
            print(f"⚠️  No images found for person {person_filter}")
            return {
                'nodes': [],
                'links': [],
                'stats': {
                    'total_clusters': 0,
                    'total_images': 0,
                    'total_connections': 0,
                    'min_similarity': min_similarity,
                    'max_similarity': 0,
                    'avg_similarity': 0,
                    'filtered_by_person': person_filter
                }
            }
        
        # Filter embeddings and filenames
        filtered_indices = []
        for idx, filename in enumerate(filenames):
            # Normalize filename
            normalized = str(filename)
            if normalized.startswith('images/'):
                normalized = normalized[7:]
            
            if normalized in person_images:
                filtered_indices.append(idx)
        
        if not filtered_indices:
            print(f"⚠️  No matching embeddings found for person {person_filter}")
            return {
                'nodes': [],
                'links': [],
                'stats': {
                    'total_clusters': 0,
                    'total_images': 0,
                    'total_connections': 0,
                    'min_similarity': min_similarity,
                    'max_similarity': 0,
                    'avg_similarity': 0,
                    'filtered_by_person': person_filter
                }
            }
        
        embeddings = embeddings[filtered_indices]
        filenames = filenames[filtered_indices]
        print(f"✅ Filtered to {len(embeddings)} images with person {person_filter}")
    
    # Cluster images by context
    clusters = cluster_images_by_context(embeddings, filenames, n_clusters=n_clusters)
    
    if not clusters:
        return {
            'nodes': [],
            'links': [],
            'stats': {
                'total_clusters': 0,
                'total_images': len(filenames),
                'total_connections': 0,
                'min_similarity': min_similarity,
                'max_similarity': 0,
                'avg_similarity': 0
            }
        }
    
    # Use provided CLIP model or load it (kept for fallback labeling)
    if clip_model is None:
        print("🔄 Loading CLIP model for cluster labeling...")
        try:
            import clip
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            clip_model, _ = clip.load("ViT-L/14", device=device)
            print(f"✅ CLIP model loaded on {device}")
        except Exception as e:
            print(f"⚠️  Could not load CLIP model: {e}, using fallback labels")
    else:
        print("✅ Using pre-loaded CLIP model from main.py")
    
    # Build nodes (one per cluster)
    print(f"📦 Building {len(clusters)} cluster nodes...")
    nodes = []
    cluster_embeddings_map = {}
    
    for cluster_id, cluster_images in clusters.items():
        # Get representative image (first one)
        representative_image = None
        if cluster_images:
            representative_filename = cluster_images[0][0]
            if str(representative_filename).startswith('images/'):
                representative_filename = str(representative_filename)[7:]
            representative_image = representative_filename
        
        # Store cluster embeddings for similarity computation
        cluster_indices = [idx for _, idx in cluster_images]
        cluster_embeddings_map[cluster_id] = embeddings[cluster_indices]
        
        nodes.append({
            'id': f'cluster_{cluster_id}',
            'cluster_id': int(cluster_id),
            'image_count': len(cluster_images),
            'size': len(cluster_images),  # Alias for frontend compatibility
            'representative_image': representative_image,
            'images': [str(fname) for fname, _ in cluster_images[:10]]  # First 10 for preview
        })
    
    print(f"📦 Created {len(nodes)} cluster nodes")
    
    # Build links based on cluster similarity
    links = []
    similarities = []
    cluster_ids = list(clusters.keys())
    
    for i, cluster1_id in enumerate(cluster_ids):
        for cluster2_id in cluster_ids[i + 1:]:
            similarity = compute_cluster_similarity(
                cluster_embeddings_map[cluster1_id],
                cluster_embeddings_map[cluster2_id]
            )
            
            if similarity >= min_similarity:
                links.append({
                    'source': f'cluster_{cluster1_id}',
                    'target': f'cluster_{cluster2_id}',
                    'similarity': float(similarity),
                    'weight': float(similarity)
                })
                similarities.append(similarity)
    
    print(f"🔗 Created {len(links)} similarity connections")
    
    # Compute positions using MDS if available
    pos = {}
    WIDTH = 2200
    HEIGHT = 1400
    
    if DEPENDENCIES_AVAILABLE and len(nodes) > 1:
        try:
            # Create distance matrix
            n = len(cluster_ids)
            distance_matrix = np.ones((n, n))
            
            for i in range(n):
                for j in range(n):
                    if i != j:
                        cluster1_id = cluster_ids[i]
                        cluster2_id = cluster_ids[j]
                        similarity = compute_cluster_similarity(
                            cluster_embeddings_map[cluster1_id],
                            cluster_embeddings_map[cluster2_id]
                        )
                        distance_matrix[i, j] = 1 - similarity
            
            # Apply MDS
            mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
            positions = mds.fit_transform(distance_matrix)
            
            # Normalize positions
            positions = positions - positions.min(axis=0)
            max_pos = positions.max(axis=0)
            if max_pos[0] > 0 and max_pos[1] > 0:
                positions = positions / max_pos
                positions[:, 0] = positions[:, 0] * WIDTH * 0.8 + WIDTH * 0.1
                positions[:, 1] = positions[:, 1] * HEIGHT * 0.8 + HEIGHT * 0.1
            
            for i, cluster_id in enumerate(cluster_ids):
                pos[f'cluster_{cluster_id}'] = positions[i]
            
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
    total_images = sum(len(cluster_images) for cluster_images in clusters.values())
    
    stats = {
        'total_clusters': len(nodes),
        'total_images': total_images,
        'total_connections': len(links),
        'min_similarity': min_similarity,
        'max_similarity': float(max_similarity),
        'avg_similarity': float(np.mean(similarities)) if similarities else 0
    }
    
    if person_filter:
        stats['filtered_by_person'] = person_filter
    
    # Get available people for filtering
    available_people = get_available_people(clusters_path)
    
    return {
        'nodes': nodes,
        'links': links,
        'stats': stats,
        'available_people': available_people
    }


if __name__ == '__main__':
    print("🧪 Testing context graph generation...")
    graph = generate_context_graph(n_clusters=10)
    
    print(f"\n📊 Graph Statistics:")
    print(f"  Clusters: {graph['stats']['total_clusters']}")
    print(f"  Images: {graph['stats']['total_images']}")
    print(f"  Connections: {graph['stats']['total_connections']}")
    print(f"  Min Similarity: {graph['stats']['min_similarity']:.2f}")
    print(f"  Max Similarity: {graph['stats']['max_similarity']:.2f}")
    print(f"  Avg Similarity: {graph['stats'].get('avg_similarity', 0):.2f}")
    
    if graph['nodes']:
        print(f"\n📦 Sample clusters:")
        for node in graph['nodes'][:5]:
            print(f"  - {node['label']}: {node['image_count']} images")
