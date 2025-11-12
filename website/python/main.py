from flask import Flask
from flask_cors import CORS
from indexing import send_progress
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
import context_graph_utils
import threading

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
    if not clip_model: # Load only if not already loaded
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)
    clip_index = faiss.read_index("./context/embeddings.faiss")
    clip_filenames = np.load("./context/filenames.npy", allow_pickle=True)
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

def rescore_with_clip(face_results, context_query, top_k=50):
    """Re-score face search results using CLIP for context matching"""
    if not context_query or not face_results:
        return face_results

    print(f"🔄 Re-scoring {len(face_results)} face results with CLIP context: '{context_query}'")

    # Extract filenames from face results
    filenames_to_check = []
    filepath_to_face_result = {}
    for result in face_results:
        filepath = result['filepath']
        filename = filepath.split('/')[-1]
        # Look up in clip_filenames to get the index
        for idx, clip_fname in enumerate(clip_filenames):
            clip_fname_clean = clip_fname[7:] if clip_fname.startswith('images/') else clip_fname
            if clip_fname_clean == filename:
                filenames_to_check.append((idx, filepath))
                filepath_to_face_result[filepath] = result
                break

    if not filenames_to_check:
        return face_results

    # Get CLIP scores for these specific images
    with torch.no_grad():
        text = clip.tokenize([context_query]).to(device)
        text_features = clip_model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    text_vec = text_features.cpu().numpy().astype(np.float32)

    # Calculate similarity for each image
    rescored_results = []
    for clip_idx, filepath in filenames_to_check:
        image_embedding = np.array([clip_index.reconstruct(int(clip_idx))], dtype=np.float32)
        # Normalize
        image_embedding = image_embedding / np.linalg.norm(image_embedding)

        # Calculate cosine similarity
        clip_score = float(np.dot(text_vec, image_embedding.T)[0][0])

        face_result = filepath_to_face_result[filepath]
        rescored_results.append({
            **face_result,
            'clip_score': clip_score,
            'face_score_original': float(face_result['score'])
        })

    return rescored_results


def rescore_with_face(text_results, person_name, top_k=50):
    """Re-score CLIP results using face recognition for person matching"""
    if not person_name or not text_results:
        return text_results

    print(f"🔄 Re-scoring {len(text_results)} CLIP results with face recognition for: '{person_name}'")

    # Find the person's cluster
    matched_cluster = None
    for cluster_id, cluster_data in people_clusters['clusters'].items():
        if cluster_data.get('name', '').lower() == person_name.lower():
            matched_cluster = cluster_data
            break

    if not matched_cluster:
        return text_results

    # Get cluster embeddings and compute medoid
    faces = matched_cluster.get('faces', [])
    if not faces:
        return text_results

    cluster_embeddings = []
    for face in faces:
        face_idx = face.get('embedding_idx')
        if face_idx is not None and face_idx < len(face_index):
            cluster_embeddings.append(face_index[face_idx])

    if not cluster_embeddings:
        return text_results

    centroid_embedding = np.mean(cluster_embeddings, axis=0)
    distances_to_center = [np.linalg.norm(emb - centroid_embedding) for emb in cluster_embeddings]
    medoid_idx = np.argmin(distances_to_center)
    reference_embedding = cluster_embeddings[medoid_idx]

    # Re-score each text result with face similarity
    rescored_results = []
    for result in text_results:
        filepath = result['filepath']
        filename = filepath.split('/')[-1]

        # Find face embeddings for this image
        face_scores_for_image = []
        for face_idx, face_fname in enumerate(face_filenames):
            face_fname_clean = face_fname[7:] if face_fname.startswith('images/') else str(face_fname)
            if face_fname_clean == filename:
                distance = np.linalg.norm(face_index[face_idx] - reference_embedding)
                face_scores_for_image.append(float(-distance))

        # Use best face match for this image
        if face_scores_for_image:
            best_face_score = float(max(face_scores_for_image))
        else:
            best_face_score = -999.0  # Very low score if no face found

        rescored_results.append({
            **result,
            'face_score': best_face_score,
            'clip_score_original': float(result['score']),
            'matched_person': person_name,
            'face_match': True if face_scores_for_image else False
        })

    return rescored_results


def combine_search_results(face_results, text_results, threshold_faces=0.0, threshold_text=0.1):
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
        face_score = float(result.get('face_score_original', result['score']))  # Use original if available
        clip_score = float(result.get('clip_score', 0))  # Get CLIP score if rescored

        # Convert negative distance to similarity in range (0, 1]
        face_similarity = 1.0 / (1.0 + abs(face_score))

        # If we have a CLIP score from rescoring, use it
        if clip_score > 0:
            combined_score = float((face_similarity + threshold_faces) * (clip_score + threshold_text))
            combined[filepath] = {
                'result': result,
                'face_score': face_score,
                'face_similarity': face_similarity,
                'text_score': clip_score,
                'combined_score': combined_score
            }
            print(f"📊 Face→CLIP: {filepath.split('/')[-1]} | Face: {face_similarity:.3f}, CLIP: {clip_score:.3f} = {combined_score:.3f} ⭐⭐")
        else:
            combined_score = float((face_similarity + threshold_faces) * threshold_text)
            combined[filepath] = {
                'result': result,
                'face_score': face_score,
                'face_similarity': face_similarity,
                'text_score': 0.0,
                'combined_score': combined_score
            }
            print(f"📊 Face only: {filepath.split('/')[-1]} | Face: {face_similarity:.6f} = {combined_score:.6f}")
    
    # Add or update with text results
    for result in text_results:
        filepath = result['filepath']
        text_score = float(result.get('clip_score_original', result['score']))  # Use original if available
        face_score = float(result.get('face_score', 0))  # Get face score if rescored
        
        if filepath in combined:
            # File found in both original searches
            # Use the rescored values if available
            existing = combined[filepath]
            if face_score != 0:  # Text was rescored with face
                face_similarity = 1.0 / (1.0 + abs(face_score))
                combined_score = float((face_similarity + threshold_faces) * (text_score + threshold_text))
                combined[filepath]['face_score'] = face_score
                combined[filepath]['face_similarity'] = face_similarity
                combined[filepath]['text_score'] = text_score
                combined[filepath]['combined_score'] = combined_score
                combined[filepath]['result']['score'] = combined_score
                print(f"📊 CLIP→Face: {filepath.split('/')[-1]} | Face: {face_similarity:.3f}, CLIP: {text_score:.3f} = {combined_score:.3f} ⭐⭐")
            else:
                # Already combined in face results section
                pass
        else:
            # File only in text search
            if face_score != 0:  # Was rescored with face
                face_similarity = 1.0 / (1.0 + abs(face_score))
                combined_score = float((face_similarity + threshold_faces) * (text_score + threshold_text))
                combined[filepath] = {
                    'result': result,
                    'face_score': face_score,
                    'face_similarity': face_similarity,
                    'text_score': text_score,
                    'combined_score': combined_score
                }
                combined[filepath]['result']['score'] = combined_score
                print(f"📊 CLIP→Face: {filepath.split('/')[-1]} | Face: {face_similarity:.3f}, CLIP: {text_score:.3f} = {combined_score:.3f} ⭐")
            else:
                combined_score = float(threshold_faces * (text_score + threshold_text))
                combined[filepath] = {
                    'result': result,
                    'face_score': 0.0,
                    'face_similarity': 0.0,
                    'text_score': text_score,
                    'combined_score': combined_score
                }
                combined[filepath]['result']['score'] = combined_score
                print(f"📊 Text only: {filepath.split('/')[-1]} | CLIP: {text_score:.3f} = {combined_score:.3f}")
    
    # Sort by combined score and return results
    sorted_results = sorted(combined.values(), key=lambda x: x['combined_score'], reverse=True)
    print(f"\n🏆 Top 5 results after combining:")
    for i, item in enumerate(sorted_results[:5]):
        print(f"  {i + 1}. {item['result']['filepath'].split('/')[-1]} | Score: {item['combined_score']:.3f}")

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
            # Person name detected - combine face and text search with bidirectional rescoring
            print(f"🔍 Found person name match: '{matched_person}'")
            
            # Get face-based results
            face_results = search_person_by_name(matched_person, top_k=100)
            
            if search_context:
                # BIDIRECTIONAL RESCORING:
                # 1. Re-score face results with CLIP context
                # 2. Do text search and re-score with face recognition
                # 3. Combine all results

                print(f"🔍 Additional context: '{search_context}' - using bidirectional rescoring")

                # Get CLIP text search results
                text_results = search_images_fast(search_context, top_k=100)

                # Re-score face results with CLIP context
                face_results_rescored = rescore_with_clip(face_results, search_context, top_k=100)

                # Re-score text results with face recognition
                text_results_rescored = rescore_with_face(text_results, matched_person, top_k=100)

                # Combine both rescored results
                results = combine_search_results(face_results_rescored, text_results_rescored)
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
                'score': float(score)
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

from flask import Flask, request, jsonify, Response
import queue

# Add this near the top with other global variables
indexing_progress_queues = {}

def run_indexing_background(task_id, progress_queue):
    """Run indexing in a separate process with progress tracking"""
    try:
        print(f"🔄 Starting background indexing for task {task_id}...")
        send_progress(task_id, 0, "Starting indexing...", progress_queue)

        # Import heavy modules INSIDE child process
        import image_index

        send_progress(task_id, 10, "Indexing images...", progress_queue)
        image_index.build_index(start_percentage=10, end_percentage=50, progress_queue=progress_queue, task_id=task_id)
        send_progress(task_id, 50, "Images indexed", progress_queue)

        import index_faces

        send_progress(task_id, 50, "Detecting faces...", progress_queue)
        index_faces.build_face_index(start_percentage=50, end_percentage=90, progress_queue=progress_queue, task_id=task_id)

        send_progress(task_id, 90, "Clustering faces...", progress_queue)
        index_faces.cluster_faces()
        send_progress(task_id, 100, "Indexing complete", progress_queue)

        print(f"✅ Background indexing completed for task {task_id}")

    except Exception as e:
        print(f"❌ Background indexing failed: {e}")
        traceback.print_exc()
        send_progress(task_id, -1, f"Error: {str(e)}", progress_queue)

import uuid
import threading
import queue
from multiprocessing import get_context
import traceback

# Store process references to ensure cleanup
indexing_processes = {}

@app.route('/api/index/trigger', methods=['POST'])
def trigger_indexing():
    """Trigger indexing in separate process and return task ID"""
    try:
        task_id = str(uuid.uuid4())

        # Use spawn context
        ctx = get_context('spawn')

        # Create manager and queue
        manager = ctx.Manager()
        progress_queue = manager.Queue()
        indexing_progress_queues[task_id] = {
            'queue': progress_queue,
            'manager': manager  # Store manager for cleanup
        }

        # Create process
        proc = ctx.Process(
            target=run_indexing_background,
            args=(task_id, progress_queue),
            daemon=False
        )
        proc.start()

        # Store process reference
        indexing_processes[task_id] = proc

        # Start monitoring thread
        monitor_thread = threading.Thread(
            target=monitor_indexing_completion,
            args=(task_id, progress_queue, manager, proc),
            daemon=True
        )
        monitor_thread.start()

        return jsonify({
            'status': 'started',
            'message': 'Indexing started in background process',
            'task_id': task_id,
            'pid': proc.pid
        }), 202

    except Exception as e:
        print(f"❌ Error triggering indexing: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def monitor_indexing_completion(task_id, progress_queue, manager, process):
    """Monitor for indexing completion and cleanup resources"""
    try:
        while True:
            try:
                progress_data = progress_queue.get(timeout=1)

                # Check for completion or error
                if progress_data.get('progress') == 100:
                    print(f"🔄 Reloading resources after indexing completion")
                    load_clip_resources()
                    load_face_resources()
                    print(f"✅ Resources reloaded")
                    break
                elif progress_data.get('progress', 0) < 0:
                    print(f"❌ Indexing failed for task {task_id}")
                    break

            except queue.Empty:
                # Check if process is still alive
                if not process.is_alive():
                    print(f"⚠️ Process died unexpectedly for task {task_id}")
                    break
                continue

    except Exception as e:
        print(f"❌ Error in completion monitor: {e}")
        traceback.print_exc()
    finally:
        # CRITICAL: Cleanup all resources
        cleanup_indexing_task(task_id, process, manager, progress_queue)

def cleanup_indexing_task(task_id, process, manager, progress_queue):
    """Clean up all resources associated with an indexing task"""
    try:
        print(f"🧹 Cleaning up indexing task {task_id}")

        # 1. Join the process (wait for it to finish)
        if process.is_alive():
            process.join(timeout=5)
            if process.is_alive():
                print(f"⚠️ Force terminating process {process.pid}")
                process.terminate()
                process.join(timeout=2)

        # 2. Clear the queue completely
        try:
            while not progress_queue.empty():
                progress_queue.get_nowait()
        except:
            pass

        # 3. **CRITICAL: Shutdown manager BEFORE closing process**
        try:
            manager.shutdown()
        except:
            pass

        # 4. Close the process object
        try:
            process.close()
        except:
            pass

        # 5. Remove from global dicts
        if task_id in indexing_progress_queues:
            del indexing_progress_queues[task_id]
        if task_id in indexing_processes:
            del indexing_processes[task_id]

        # 6. Force garbage collection
        import gc
        gc.collect()

        print(f"✅ Cleaned up task {task_id}")

    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        traceback.print_exc()

@app.route('/api/index/progress/<task_id>')
def indexing_progress(task_id):
    """SSE endpoint for indexing progress"""

    def generate():
        if task_id not in indexing_progress_queues:
            yield f"data: {json.dumps({'progress': -1, 'message': 'Task not found'})}\n\n"
            return

        progress_queue = indexing_progress_queues[task_id]['queue']

        while True:
            try:
                # Wait for progress update with timeout
                progress_data = progress_queue.get(timeout=30)
                yield f"data: {json.dumps(progress_data)}\n\n"

                # If complete or error, end stream
                if progress_data['progress'] >= 100 or progress_data['progress'] < 0:
                    break

            except queue.Empty:
                # Send keepalive
                yield f": keepalive\n\n"
            except:
                break

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )

@app.route('/api/images/context-graph', methods=['GET'])
def get_context_graph():
    """
    Get context clustering graph data showing images clustered by semantic content
    Uses CLIP embeddings to cluster images by context and displays context labels

    Query params:
        person_filter: (Deprecated) Single person name/ID to filter images by
        people_filter: Comma-separated list of person IDs - show only images with ALL these people
        min_similarity: Minimum similarity to create links (0-1, default: 0.6)
        num_clusters: Number of context clusters to create (default: 10)

    Returns JSON with:
    - nodes: Array of image clusters with context labels and positions
    - links: Array of connections between similar context clusters
    - stats: Overall graph statistics
    """
    try:
        person_filter = request.args.get('person_filter')  # Single person (deprecated)
        people_filter_str = request.args.get('people_filter')  # Multiple people (comma-separated)
        min_similarity = float(request.args.get('min_similarity', 0.6))
        num_clusters = int(request.args.get('num_clusters', 10))

        # Parse people_filter from comma-separated string to list
        people_filter = None
        if people_filter_str:
            people_filter = [p.strip() for p in people_filter_str.split(',') if p.strip()]

        embeddings_path = 'embeddings.faiss'
        filenames_path = 'filenames.npy'
        clusters_path = 'faces/clusters.json'

        print(f"📊 Generating context graph (people_filter={people_filter}, min_similarity={min_similarity}, num_clusters={num_clusters})...")

        # Pass the global CLIP model to avoid reloading
        graph_data = context_graph_utils.generate_context_graph(
            embeddings_path=embeddings_path,
            filenames_path=filenames_path,
            clusters_path=clusters_path,
            n_clusters=num_clusters,
            min_similarity=min_similarity,
            person_filter=person_filter,  # Backward compatibility
            people_filter=people_filter,  # New multi-person filter
            clip_model=clip_model,  # Use global pre-loaded model
            device=device  # Use global device
        )

        return jsonify(graph_data), 200

    except Exception as e:
        print(f"❌ Error generating context graph: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'message': 'Failed to generate context graph'
        }), 500


if __name__ == '__main__':
    # Load resources into memory at startup
    load_clip_resources()
    load_face_resources()

    print("🚀 Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)