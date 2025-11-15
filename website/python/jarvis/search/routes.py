from flask import Blueprint, app, request, jsonify, current_app
import face_recognition
import os
from datetime import datetime
import traceback
import torch
import clip
import numpy as np

search_bp = Blueprint('search', __name__)

def get_people_clusters():
    return current_app.config['PEOPLE_CLUSTERS']

def get_face_index():
    return current_app.config['FACE_INDEX']

def get_face_filenames():
    return current_app.config['FACE_FILENAMES']


def search_images_fast(query, clip_model, clip_preprocess, clip_index, clip_filenames, device, top_k=5):
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

def search_person_fast(reference_image_path, face_index, face_filenames, top_k=5):
    """Fast face search using pre-loaded resources"""

    ref_image = face_recognition.load_image_file(reference_image_path)
    ref_encoding = face_recognition.face_encodings(ref_image)

    if not ref_encoding:
        return []

    ref_encoding = ref_encoding[0]
    distances = np.linalg.norm(face_index - ref_encoding, axis=1)
    top_indices = np.argsort(distances)[:top_k]

    results = []
    for rank, idx in enumerate(top_indices):
        filename = face_filenames[idx]
        if filename.startswith('images/'):
            filename = filename[7:]

        image_path = f'/api/images/{filename}'
        score = float(-distances[idx])
        results.append({
            'filepath': image_path,
            'thumbnail': image_path,
            'score': score
        })

    return results

def search_person_by_name(person_name, top_k=50):
    """Search for images by person name from clusters"""
    people_clusters = get_people_clusters()
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
        if face_idx is not None and face_idx < len(get_face_index()):
            cluster_embeddings.append(get_face_index()[face_idx])
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
    distances = np.linalg.norm(get_face_index() - reference_embedding, axis=1)

    # Only deprioritize the medoid itself (set to max of other values)
    max_other_distance = np.max([distances[i] for i in range(len(distances))
                                 if i != medoid_global_idx])
    distances[medoid_global_idx] = max_other_distance

    # Get top k results
    top_indices = np.argsort(distances)[:top_k * 2]

    results = []
    seen_files = set()

    for rank, idx in enumerate(top_indices):
        if idx >= len(get_face_filenames()):
            continue

        if len(results) >= top_k:
            break

        filename = str(get_face_filenames()[idx])

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

def check_person_names_match(query):
    """Check if query matches one or more person names in clusters

    Returns:
        matched_persons: List of matched person names
        search_context: Remaining query text after removing person names
    """
    people_clusters = get_people_clusters()
    if not people_clusters or not people_clusters.get('clusters'):
        return [], query

    query_lower = query.lower().strip()
    matched_persons = []
    remaining_query = query_lower

    # Sort person names by length (longest first) to match longer names first
    person_names = []
    for cluster_id, cluster_data in people_clusters['clusters'].items():
        person_name = cluster_data.get('name', '').strip()
        if person_name:
            person_names.append(person_name)

    person_names.sort(key=len, reverse=True)

    # Find all matching person names
    for person_name in person_names:
        person_name_lower = person_name.lower()

        if person_name_lower in remaining_query:
            matched_persons.append(person_name)
            # Remove matched name from query
            remaining_query = remaining_query.replace(person_name_lower, '').strip()

    # Clean up remaining query (remove extra spaces)
    search_context = ' '.join(remaining_query.split())

    return matched_persons, search_context

def search_multiple_persons_by_name(person_names, top_k=50):
    """Search for images containing ALL specified persons

    Args:
        person_names: List of person names to search for
        top_k: Number of results to return

    Returns:
        List of images containing all specified persons
    """
    if not person_names:
        return []

    # Get images for each person
    person_image_sets = []
    person_embeddings = []

    for person_name in person_names:
        # Find the person's cluster
        matched_cluster = None
        people_clusters = get_people_clusters()
        for cluster_id, cluster_data in people_clusters['clusters'].items():
            if cluster_data.get('name', '').lower() == person_name.lower():
                matched_cluster = cluster_data
                break

        if not matched_cluster:
            print(f"⚠️ Person not found: {person_name}")
            return []  # If any person not found, return empty

        # Get all faces for this person
        faces = matched_cluster.get('faces', [])
        if not faces:
            return []

        # Collect images containing this person
        person_images = set()
        cluster_embeddings = []

        for face in faces:
            face_idx = face.get('embedding_idx')
            face_index = get_face_index()
            face_filenames = get_face_filenames()
            if face_idx is not None and face_idx < len(face_filenames):
                filename = str(face_filenames[face_idx])
                person_images.add(filename)
                cluster_embeddings.append(face_index[face_idx])

        person_image_sets.append(person_images)

        # Calculate medoid for this person
        if cluster_embeddings:
            centroid = np.mean(cluster_embeddings, axis=0)
            distances = [np.linalg.norm(emb - centroid) for emb in cluster_embeddings]
            medoid_idx = np.argmin(distances)
            person_embeddings.append(cluster_embeddings[medoid_idx])

    # Find intersection - images containing ALL persons
    common_images = set.intersection(*person_image_sets)

    if not common_images:
        print(f"⚠️ No images found containing all persons: {', '.join(person_names)}")
        return []

    print(f"✅ Found {len(common_images)} images containing all {len(person_names)} persons")

    # Score images based on average face match quality for all persons
    results = []

    for filename in common_images:
        # Calculate average match score across all persons
        total_score = 0.0

        for person_embedding in person_embeddings:
            # Find best face match in this image for this person
            best_distance = float('inf')

            for face_idx, face_fname in enumerate(face_filenames):
                face_fname_clean = face_fname[7:] if face_fname.startswith('images/') else str(face_fname)
                if face_fname_clean == filename or str(face_fname) == filename:
                    distance = np.linalg.norm(face_index[face_idx] - person_embedding)
                    best_distance = min(best_distance, distance)

            if best_distance != float('inf'):
                total_score += -best_distance

        # Average score across all persons
        avg_score = total_score / len(person_embeddings)

        # Format filename
        display_filename = filename[7:] if filename.startswith('images/') else filename
        image_path = f'/api/images/{display_filename}'

        results.append({
            'filepath': image_path,
            'thumbnail': image_path,
            'score': float(avg_score),
            'matched_persons': person_names,
            'face_match': True
        })

    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)

    return results[:top_k]

def rescore_with_clip(face_results, context_query, clip_model, clip_index, clip_filenames, device, top_k=50):
    """Re-score face search results using CLIP for context matching"""
    if not context_query or not face_results:
        return face_results

    print(f"🔄 Re-scoring {len(face_results)} face results with CLIP context: '{context_query}'")

    filenames_to_check = []
    filepath_to_face_result = {}
    for result in face_results:
        filepath = result['filepath']
        filename = filepath.split('/')[-1]
        for idx, clip_fname in enumerate(clip_filenames):
            clip_fname_clean = clip_fname[7:] if clip_fname.startswith('images/') else clip_fname
            if clip_fname_clean == filename:
                filenames_to_check.append((idx, filepath))
                filepath_to_face_result[filepath] = result
                break

    if not filenames_to_check:
        return face_results

    with torch.no_grad():
        text = clip.tokenize([context_query]).to(device)
        text_features = clip_model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    text_vec = text_features.cpu().numpy().astype(np.float32)

    rescored_results = []
    for clip_idx, filepath in filenames_to_check:
        image_embedding = np.array([clip_index.reconstruct(int(clip_idx))], dtype=np.float32)
        image_embedding = image_embedding / np.linalg.norm(image_embedding)
        clip_score = float(np.dot(text_vec, image_embedding.T)[0][0])

        face_result = filepath_to_face_result[filepath]
        rescored_results.append({
            **face_result,
            'clip_score': clip_score,
            'face_score_original': float(face_result['score'])
        })

    return rescored_results

def rescore_with_face(text_results, person_name, people_clusters, face_index, face_filenames, top_k=50):
    """Re-score CLIP results using face recognition for person matching"""
    if not person_name or not text_results:
        return text_results

    print(f"🔄 Re-scoring {len(text_results)} CLIP results with face recognition for: '{person_name}'")

    matched_cluster = None
    for cluster_id, cluster_data in people_clusters['clusters'].items():
        if cluster_data.get('name', '').lower() == person_name.lower():
            matched_cluster = cluster_data
            break

    if not matched_cluster:
        return text_results

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

    rescored_results = []
    for result in text_results:
        filepath = result['filepath']
        filename = filepath.split('/')[-1]

        face_scores_for_image = []
        for face_idx, face_fname in enumerate(face_filenames):
            face_fname_clean = face_fname[7:] if face_fname.startswith('images/') else str(face_fname)
            if face_fname_clean == filename:
                distance = np.linalg.norm(face_index[face_idx] - reference_embedding)
                face_scores_for_image.append(float(-distance))

        if face_scores_for_image:
            best_face_score = float(max(face_scores_for_image))
        else:
            best_face_score = -999.0

        rescored_results.append({
            **result,
            'face_score': best_face_score,
            'clip_score_original': float(result['score']),
            'matched_person': person_name,
            'face_match': True if face_scores_for_image else False
        })

    return rescored_results

def combine_search_results(face_results, text_results, threshold_faces=0.0, threshold_text=0.1):
    """Combine face and text search results using multiplicative scoring with threshold"""
    combined = {}

    for result in face_results:
        filepath = result['filepath']
        face_score = float(result.get('face_score_original', result['score']))
        clip_score = float(result.get('clip_score', 0))

        face_similarity = 1.0 / (1.0 + abs(face_score))

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
    
    for result in text_results:
        filepath = result['filepath']
        text_score = float(result.get('clip_score_original', result['score']))
        face_score = float(result.get('face_score', 0))
        
        if filepath in combined:
            existing = combined[filepath]
            if face_score != 0:
                face_similarity = 1.0 / (1.0 + abs(face_score))
                combined_score = float((face_similarity + threshold_faces) * (text_score + threshold_text))
                combined[filepath]['face_score'] = face_score
                combined[filepath]['face_similarity'] = face_similarity
                combined[filepath]['text_score'] = text_score
                combined[filepath]['combined_score'] = combined_score
                combined[filepath]['result']['score'] = combined_score
                print(f"📊 CLIP→Face: {filepath.split('/')[-1]} | Face: {face_similarity:.3f}, CLIP: {text_score:.3f} = {combined_score:.3f} ⭐⭐")
        else:
            if face_score != 0:
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
    
    sorted_results = sorted(combined.values(), key=lambda x: x['combined_score'], reverse=True)
    print(f"\n🏆 Top 5 results after combining:")
    for i, item in enumerate(sorted_results[:5]):
        print(f"  {i + 1}. {item['result']['filepath'].split('/')[-1]} | Score: {item['combined_score']:.3f}")

    return [item['result'] for item in sorted_results]

@search_bp.route('/api/search', methods=['POST'])
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
        matched_persons = []
        search_context = None

        # Check if query matches person names and extract context
        if search_type == 'text':
            matched_persons, search_context = check_person_names_match(query)

        if matched_persons and search_type == 'text':
            # Person names detected
            print(f"🔍 Found {len(matched_persons)} person name(s): {', '.join(matched_persons)}")

            # Get face-based results for all matched persons
            face_results = search_multiple_persons_by_name(matched_persons, top_k=100)

            if search_context and face_results:
                # Re-score face results with CLIP context
                print(f"🔍 Additional context: '{search_context}' - rescoring with CLIP")

                # Get CLIP text search results
                text_results = search_images_fast(search_context, top_k=100)

                # Re-score face results with CLIP
                face_results_rescored = rescore_with_clip(face_results, search_context, top_k=100)

                # Re-score text results with face recognition (using first person as reference)
                text_results_rescored = rescore_with_face(text_results, matched_persons[0], top_k=100)

                # Combine results
                results = combine_search_results(face_results_rescored, text_results_rescored)
            else:
                # No additional context or no face results
                results = face_results

        elif search_type == 'face':
            # Direct face search with reference image
            results = search_person_fast(query, get_face_index(), get_face_filenames(), top_k=50)
        else:
            # Standard CLIP text search
            clip_model = current_app.config['CLIP_MODEL']
            clip_preprocess = current_app.config['CLIP_PREPROCESS']
            clip_index = current_app.config['CLIP_INDEX']
            clip_filenames = current_app.config['CLIP_FILENAMES']

            results = search_images_fast(query, clip_model=clip_model, clip_preprocess=clip_preprocess,
                                         clip_index=clip_index, clip_filenames=clip_filenames, device='cpu', top_k=50)

        # Convert results to frontend format
        formatted_results = []
        for idx, result in enumerate(results):
            filepath = result['filepath']
            score = result['score']
            thumbnail = result['thumbnail']
            matched_person_name = result.get('matched_person', None)
            matched_persons_list = result.get('matched_persons', None)
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

            # Add matched person info
            if matched_persons_list:
                result_item['matched_persons'] = matched_persons_list
            elif matched_person_name:
                result_item['matched_person'] = matched_person_name

            formatted_results.append(result_item)

        # Add metadata about the search
        response_data = {'results': formatted_results}
        if matched_persons:
            response_data['search_type'] = 'combined_face_text' if search_context else 'multi_person_face'
            response_data['matched_persons'] = matched_persons
            if search_context:
                response_data['search_context'] = search_context

        return jsonify(response_data)

    except Exception as e:
        print(f"Error during search: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e), 'results': []}), 500

@search_bp.route('/api/index/status', methods=['GET'])
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