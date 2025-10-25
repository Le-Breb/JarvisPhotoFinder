from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from datetime import datetime
import traceback
import torch
import clip
import faiss
import numpy as np
import json
import graph_utils
import similarity_graph_utils

app = Flask(__name__)
CORS(app)

# Global variables to store loaded models and indexes
clip_model = None
clip_preprocess = None
clip_index = None
clip_filenames = None
face_index = None
face_filenames = None
device = None
people_clusters = None

def load_clip_resources():
    """Load CLIP model and index into memory once at startup"""
    global clip_model, clip_preprocess, clip_index, clip_filenames, device
    
    print("🔄 Loading CLIP model and index...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)
    clip_index = faiss.read_index("embeddings.faiss")
    clip_filenames = np.load("filenames.npy")
    print(f"✅ CLIP resources loaded on {device}")

def load_face_resources():
    """Load face recognition resources into memory"""
    global face_index, face_filenames, people_clusters
    
    try:
        print("🔄 Loading face recognition resources...")
        face_index = np.load("faces/embeddings.npy")
        face_filenames = np.load("faces/filenames.npy", allow_pickle=True)
        
        # Load people clusters
        clusters_path = "faces/clusters.json"
        if os.path.exists(clusters_path):
            with open(clusters_path, 'r') as f:
                people_clusters = json.load(f)
            print(f"✅ Loaded {len(people_clusters.get('clusters', {}))} people clusters")
        else:
            people_clusters = {"clusters": {}, "noise": []}
            print("⚠️  No clusters.json found")
            
        print("✅ Face recognition resources loaded")
    except Exception as e:
        print(f"⚠️  Face recognition resources not found: {e}")
        people_clusters = {"clusters": {}, "noise": []}

def search_images_fast(query, top_k=5):
    """Fast search using pre-loaded resources"""
    with torch.no_grad():
        text = clip.tokenize([query]).to(device)
        text_features = clip_model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    query_vec = text_features.cpu().numpy().astype(np.float32)
    scores, indices = clip_index.search(query_vec, top_k)

    results = []
    for rank, i in enumerate(indices[0]):
        filename = clip_filenames[i]
        if filename.startswith('images/'):
            filename = filename[7:]
        
        image_path = f'/api/images/{filename}'
        results.append({
            'filepath': image_path,
            'thumbnail': image_path,
            'score': float(scores[0][rank])
        })

    return results

def search_person_fast(reference_image_path, top_k=5):
    """Fast face search using pre-loaded resources"""
    # Import face recognition here to avoid loading if not needed
    import face_recognition
    
    # Load and encode the reference image
    ref_image = face_recognition.load_image_file(reference_image_path)
    ref_encoding = face_recognition.face_encodings(ref_image)
    
    if not ref_encoding:
        return []
    
    ref_encoding = ref_encoding[0]
    
    # Calculate distances to all faces in the index
    distances = np.linalg.norm(face_index - ref_encoding, axis=1)
    
    # Get top k results
    top_indices = np.argsort(distances)[:top_k]
    
    results = []
    for rank, idx in enumerate(top_indices):
        filename = face_filenames[idx]
        if filename.startswith('images/'):
            filename = filename[7:]
        
        image_path = f'/api/images/{filename}'
        # Use negative distance as score (closer to 0 = better match)
        # Negative so that better matches have higher scores
        score = float(-distances[idx])
        results.append({
            'filepath': image_path,
            'thumbnail': image_path,
            'score': score
        })
    
    return results

def search_person_by_name(person_name, top_k=50):
    """Search for images by person name from clusters"""
    if not people_clusters or not people_clusters.get('clusters'):
        return []
    
    # Search for person by name (case-insensitive)
    matched_cluster = None
    for cluster_id, cluster_data in people_clusters['clusters'].items():
        if cluster_data.get('name', '').lower() == person_name.lower():
            matched_cluster = cluster_data
            break
    
    if not matched_cluster:
        return []
    
    # Get all faces from this cluster
    faces = matched_cluster.get('faces', [])
    if not faces:
        return []
    
    # Collect cluster embeddings
    cluster_embeddings = []
    cluster_embedding_indices = []
    for face in faces:
        face_idx = face.get('embedding_idx')
        if face_idx is not None and face_idx < len(face_index):
            cluster_embeddings.append(face_index[face_idx])
            cluster_embedding_indices.append(face_idx)
    
    if not cluster_embeddings:
        return []
    
    # Use the medoid (most representative actual face) as reference
    centroid_embedding = np.mean(cluster_embeddings, axis=0)
    distances_to_center = [np.linalg.norm(emb - centroid_embedding) for emb in cluster_embeddings]
    medoid_local_idx = np.argmin(distances_to_center)
    medoid_global_idx = cluster_embedding_indices[medoid_local_idx]
    reference_embedding = cluster_embeddings[medoid_local_idx]
    
    # Find similar faces using the medoid as reference
    distances = np.linalg.norm(face_index - reference_embedding, axis=1)
    
    # Only deprioritize the medoid itself (set to max of other values)
    max_other_distance = np.max([distances[i] for i in range(len(distances)) 
                                  if i != medoid_global_idx])
    distances[medoid_global_idx] = max_other_distance
    
    # Get top k results
    top_indices = np.argsort(distances)[:top_k * 2]
    
    results = []
    seen_files = set()
    
    for rank, idx in enumerate(top_indices):
        if idx >= len(face_filenames):
            continue
        
        if len(results) >= top_k:
            break
            
        filename = str(face_filenames[idx])
        
        # Avoid duplicates
        if filename in seen_files:
            continue
        seen_files.add(filename)
        
        if filename.startswith('images/'):
            filename = filename[7:]
        
        image_path = f'/api/images/{filename}'
        score = float(-distances[idx])
        
        results.append({
            'filepath': image_path,
            'thumbnail': image_path,
            'score': score,
            'matched_person': matched_cluster.get('name'),
            'face_match': True
        })
    
    return results

def check_person_name_match(query):
    """Check if query matches a person name in clusters"""
    if not people_clusters or not people_clusters.get('clusters'):
        return None, query
    
    query_lower = query.lower().strip()
    
    for cluster_id, cluster_data in people_clusters['clusters'].items():
        person_name = cluster_data.get('name', '').strip()
        person_name_lower = person_name.lower()
        
        # Check for exact match or if query contains person name
        if person_name_lower == query_lower:
            return person_name, ""  # Exact match, no additional context
        elif person_name_lower in query_lower:
            # Person name found in query - keep full query as context
            return person_name, query
        elif query_lower in person_name_lower:
            return person_name, query  # Query is part of name, use full query
    
    return None, query

def combine_search_results(face_results, text_results, threshold=0.01):
    """Combine face and text search results using multiplicative scoring with threshold
    
    Face scores are negative distances. Convert to similarity by inverting:
    face_similarity = 1 / (1 + abs(face_score))
    Then: combined_score = (face_similarity + threshold) * (text_score + threshold)
    """
    # Create a dict to track combined scores
    combined = {}
    
    # Add face results with threshold
    for result in face_results:
        filepath = result['filepath']
        face_score = result['score']  # Negative distance
        # Convert negative distance to similarity in range (0, 1]: use inverse distance
        # This maps distance 0 -> 1.0, distance 10 -> ~0.091, distance 20 -> ~0.047
        # We use 1 / (1 + distance) which is stable for a wide range of distances.
        face_similarity = 1.0 / (1.0 + abs(face_score))
        # Add threshold, use threshold for missing text score
        combined_score = (face_similarity + threshold) * threshold
        combined[filepath] = {
            'result': result,
            'face_score': face_score,
            'face_similarity': face_similarity,
            'text_score': 0,  # No text match yet
            'combined_score': combined_score
        }
        # Print with higher precision so very small similarities are visible
        print(f"📊 Face only: {filepath.split('/')[-1]} | Face dist: {face_score:.3f} → sim: {face_similarity:.6f} | ({face_similarity:.6f} + {threshold:.6f}) × {threshold:.6f} = {combined_score:.6f}")
    
    # Add or update with text results
    for result in text_results:
        filepath = result['filepath']
        text_score = result['score']
        
        if filepath in combined:
            # File found in both searches - multiply with thresholds added!
            face_similarity = combined[filepath]['face_similarity']
            combined_score = (face_similarity + threshold) * (text_score + threshold)
            combined[filepath]['text_score'] = text_score
            combined[filepath]['combined_score'] = combined_score
            # Preserve face match metadata
            combined[filepath]['result']['score'] = combined_score
            print(f"📊 BOTH: {filepath.split('/')[-1]} | ({face_similarity:.3f} + {threshold:.3f}) × ({text_score:.3f} + {threshold:.3f}) = {combined_score:.3f} ⭐")
        else:
            # File only in text search - use threshold for missing face match
            combined_score = threshold * (text_score + threshold)
            combined[filepath] = {
                'result': result,
                'face_score': 0,
                'face_similarity': 0,
                'text_score': text_score,
                'combined_score': combined_score
            }
            combined[filepath]['result']['score'] = combined_score
            print(f"📊 Text only: {filepath.split('/')[-1]} | {threshold:.3f} × ({text_score:.3f} + {threshold:.3f}) = {combined_score:.3f}")
    
    # Sort by combined score and return results
    sorted_results = sorted(combined.values(), key=lambda x: x['combined_score'], reverse=True)
    print(f"\n🏆 Top 5 results after combining:")
    for i, item in enumerate(sorted_results[:5]):
        print(f"  {i+1}. {item['result']['filepath'].split('/')[-1]} | Score: {item['combined_score']:.3f}")
    
    return [item['result'] for item in sorted_results]

@app.route('/api/search', methods=['POST'])
def search():
    try:
        data = request.json
        query = data.get('query', '')
        date_from = data.get('dateFrom')
        date_to = data.get('dateTo')
        search_type = data.get('type', 'text')

        if not query:
            return jsonify({'results': []})

        results = []
        matched_person = None
        search_context = None
        
        # Check if query matches a person name and extract context
        if search_type == 'text':
            matched_person, search_context = check_person_name_match(query)
        
        if matched_person and search_type == 'text':
            # Person name detected - combine face and text search
            print(f"🔍 Found person name match: '{matched_person}'")
            
            # Get face-based results
            face_results = search_person_by_name(matched_person, top_k=50)
            
            if search_context:
                # If there's additional context, also do text search
                print(f"🔍 Additional context: '{search_context}' - combining with semantic search")
                text_results = search_images_fast(search_context, top_k=50)
                
                # Combine both results with weighted scoring
                results = combine_search_results(face_results, text_results)
            else:
                # No additional context, just use face results
                results = face_results
                
        elif search_type == 'face':
            # Direct face search with reference image
            results = search_person_fast(query, top_k=50)
        else:
            # Standard CLIP text search
            results = search_images_fast(query, top_k=50)

        # Convert results to the format expected by the frontend
        formatted_results = []
        for idx, result in enumerate(results):
            filepath = result['filepath']
            score = result['score']
            thumbnail = result['thumbnail']
            matched_person_name = result.get('matched_person', None)
            face_match = result.get('face_match', False)
            
            filename = filepath.split('/')[-1]
            actual_file_path = os.path.join(os.getcwd(), '../private/images', filename)

            # Get file modification time
            try:
                file_stat = os.stat(actual_file_path)
                file_date = datetime.fromtimestamp(file_stat.st_mtime).isoformat()
            except:
                file_date = datetime.now().isoformat()

            # Filter by date range if provided
            if date_from or date_to:
                file_datetime = datetime.fromisoformat(file_date.replace('Z', '+00:00').split('+')[0])
                if date_from:
                    from_datetime = datetime.fromisoformat(date_from.replace('Z', '+00:00').split('+')[0])
                    if file_datetime < from_datetime:
                        continue
                if date_to:
                    to_datetime = datetime.fromisoformat(date_to.replace('Z', '+00:00').split('+')[0])
                    if file_datetime > to_datetime:
                        continue

            file_type = 'image' if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')) else 'pdf'

            result_item = {
                'id': str(idx),
                'filename': filename,
                'filepath': filepath,
                'type': file_type,
                'thumbnail': thumbnail,
                'date': file_date,
                'score': score
            }
            
            # Add matched person info if available
            if matched_person_name:
                result_item['matched_person'] = matched_person_name
            
            formatted_results.append(result_item)

        # Add metadata about the search
        response_data = {'results': formatted_results}
        if matched_person:
            response_data['search_type'] = 'combined_face_text' if search_context else 'face_by_name'
            response_data['matched_person'] = matched_person
            if search_context:
                response_data['search_context'] = search_context
        
        return jsonify(response_data)

    except Exception as e:
        print(f"Error during search: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e), 'results': []}), 500

@app.route('/api/index/status', methods=['GET'])
def index_status():
    """Check if index files exist"""
    try:
        embeddings_exist = os.path.exists('embeddings.faiss')
        filenames_exist = os.path.exists('filenames.npy')
        face_embeddings_exist = os.path.exists('faces/embeddings.npy')
        face_filenames_exist = os.path.exists('faces/filenames.npy')

        return jsonify({
            'clip_index': embeddings_exist and filenames_exist,
            'face_index': face_embeddings_exist and face_filenames_exist
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/people/graph', methods=['GET'])
def get_social_graph():
    """
    Get social graph data showing people connections based on photo co-occurrence
    
    Returns JSON with:
    - nodes: Array of people with their stats
    - links: Array of connections between people
    - stats: Overall graph statistics
    """
    try:
        clusters_path = 'faces/clusters.json'
        print("📊 Generating social graph...")
        graph_data = graph_utils.generate_social_graph(clusters_path)
        return jsonify(graph_data), 200
    except Exception as e:
        print(f"❌ Error generating social graph: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'message': 'Failed to generate social graph'
        }), 500


@app.route('/api/people/graph/connections/<person_id>', methods=['GET'])
def get_person_graph_connections(person_id):
    """
    Get all connections for a specific person
    
    Args:
        person_id: ID of the person
        
    Returns JSON with array of connections
    """
    try:
        clusters_path = 'faces/clusters.json'
        connections = graph_utils.get_person_connections(person_id, clusters_path)
        return jsonify({
            'person_id': person_id,
            'connections': connections
        }), 200
    except Exception as e:
        print(f"❌ Error getting person connections: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/people/graph/top-connections', methods=['GET'])
def get_top_connections():
    """
    Get the strongest connections in the graph
    
    Query params:
        limit: Number of top connections (default: 10)
    """
    try:
        limit = int(request.args.get('limit', 10))
        clusters_path = 'faces/clusters.json'
        top_connections = graph_utils.get_strongest_connections(top_n=limit, clusters_path=clusters_path)
        return jsonify({
            'top_connections': top_connections,
            'count': len(top_connections)
        }), 200
    except Exception as e:
        print(f"❌ Error getting top connections: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/people/similarity-graph', methods=['GET'])
def get_similarity_graph():
    """
    Get similarity graph data showing people positioned by facial similarity
    Uses face embeddings to compute similarity and MDS for positioning
    
    Query params:
        min_similarity: Minimum similarity threshold (0-1, default: 0.5)
        
    Returns JSON with:
    - nodes: Array of people with their stats and positions based on similarity
    - links: Array of connections between similar people
    - stats: Overall graph statistics
    """
    try:
        min_similarity = float(request.args.get('min_similarity', 0.5))
        clusters_path = 'faces/clusters.json'
        embeddings_path = 'faces/embeddings.npy'
        
        print(f"📊 Generating similarity graph (min_similarity={min_similarity})...")
        graph_data = similarity_graph_utils.generate_similarity_graph(
            clusters_path=clusters_path,
            embeddings_path=embeddings_path,
            min_similarity=min_similarity
        )
        return jsonify(graph_data), 200
    except Exception as e:
        print(f"❌ Error generating similarity graph: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'message': 'Failed to generate similarity graph'
        }), 500


@app.route('/api/people/similar-clusters', methods=['GET'])
def get_similar_clusters():
    """
    Find pairs of clusters that are very similar and might need to be merged
    Uses face embeddings to compute similarity between cluster centroids
    
    Query params:
        min_similarity: Minimum similarity threshold (0-1, default: 0.7)
        max_pairs: Maximum number of pairs to return (default: 50)
        
    Returns JSON with:
    - pairs: Array of similar cluster pairs with their similarity scores
    """
    try:
        min_similarity = float(request.args.get('min_similarity', 0.7))
        max_pairs = int(request.args.get('max_pairs', 50))
        
        clusters_path = 'faces/clusters.json'
        embeddings_path = 'faces/embeddings.npy'
        
        print(f"🔍 Finding similar clusters (min_similarity={min_similarity}, max_pairs={max_pairs})...")
        
        # Compute person embeddings (average embedding per person)
        person_embeddings = similarity_graph_utils.compute_person_embeddings(clusters_path, embeddings_path)
        
        if not person_embeddings:
            return jsonify({'pairs': []}), 200
        
        # Compute similarity matrix
        similarity_matrix, person_ids = similarity_graph_utils.compute_similarity_matrix(person_embeddings)
        
        # Load cluster data for metadata
        face_data = similarity_graph_utils.load_face_clusters(clusters_path)
        
        # Find similar pairs
        similar_pairs = []
        for i in range(len(person_ids)):
            for j in range(i + 1, len(person_ids)):
                person1 = person_ids[i]
                person2 = person_ids[j]
                
                similarity = similarity_matrix.get((person1, person2), 0)
                
                if similarity >= min_similarity:
                    # Get cluster info
                    cluster1 = face_data.get(person1, {})
                    cluster2 = face_data.get(person2, {})
                    
                    faces1 = cluster1.get('faces', [])
                    faces2 = cluster2.get('faces', [])
                    
                    if not faces1 or not faces2:
                        continue
                    
                    # Get representative faces
                    rep1 = faces1[0] if faces1 else None
                    rep2 = faces2[0] if faces2 else None
                    
                    similar_pairs.append({
                        'cluster1_id': person1,
                        'cluster1_name': cluster1.get('name', f'Person {person1}'),
                        'cluster1_face_count': len(faces1),
                        'cluster1_representative': {
                            'filename': rep1.get('filename') or rep1.get('image'),
                            'bbox': rep1.get('bbox')
                        } if rep1 else None,
                        'cluster2_id': person2,
                        'cluster2_name': cluster2.get('name', f'Person {person2}'),
                        'cluster2_face_count': len(faces2),
                        'cluster2_representative': {
                            'filename': rep2.get('filename') or rep2.get('image'),
                            'bbox': rep2.get('bbox')
                        } if rep2 else None,
                        'similarity': float(similarity)
                    })
        
        # Sort by similarity (highest first) and limit
        similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)
        similar_pairs = similar_pairs[:max_pairs]
        
        print(f"✅ Found {len(similar_pairs)} similar cluster pairs")
        
        return jsonify({'pairs': similar_pairs}), 200
        
    except Exception as e:
        print(f"❌ Error finding similar clusters: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'message': 'Failed to find similar clusters'
        }), 500


if __name__ == '__main__':
    # Load resources into memory at startup
    load_clip_resources()
    load_face_resources()
    
    print("🚀 Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)