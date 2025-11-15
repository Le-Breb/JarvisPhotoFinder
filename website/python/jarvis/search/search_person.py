import torch
import clip
import numpy as np
from jarvis.config import DEFAULT_SEARCH_COUNT_FACES
import face_recognition
from flask import current_app


def search_person_with_reference(reference_image_path, top_k=DEFAULT_SEARCH_COUNT_FACES):
    """Fast face search using pre-loaded resources"""

    ref_image = face_recognition.load_image_file(reference_image_path)
    ref_encoding = face_recognition.face_encodings(ref_image)
    face_filenames = current_app.config['FACE_FILENAMES']
    face_index = current_app.config['FACE_INDEX']

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
    people_clusters = current_app.config.get('PEOPLE_CLUSTERS')
    get_face_index = current_app.config.get('GET_FACE_INDEX')
    get_face_filenames = current_app.config.get('GET_FACE_FILENAMES')
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
        people_clusters = current_app.config.get('PEOPLE_CLUSTERS')
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
            face_index = current_app.config.get('GET_FACE_INDEX')()
            face_filenames = current_app.config.get('GET_FACE_FILENAMES')()
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