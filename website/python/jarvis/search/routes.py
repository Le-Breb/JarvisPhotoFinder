from flask import Blueprint, app, request, jsonify, current_app
import face_recognition
import os
from datetime import datetime
import traceback
import torch
import clip
import numpy as np
from jarvis.search.search_combined import combine_search_results
from jarvis.search.search_context import rescore_with_clip, search_images_fast
from jarvis.search.search_person import rescore_with_face, search_multiple_persons_by_name, search_person_with_reference

search_bp = Blueprint('search', __name__)


def check_person_names_match(query):
    """Check if query matches one or more person names in clusters

    Returns:
        matched_persons: List of matched person names
        search_context: Remaining query text after removing person names
    """
    people_clusters = current_app.config.get('PEOPLE_CLUSTERS')
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
            results = search_person_with_reference(query, top_k=50)
        else:
            # Standard CLIP text search
            clip_model = current_app.config['CLIP_MODEL']
            clip_preprocess = current_app.config['CLIP_PREPROCESS']
            clip_index = current_app.config['CLIP_INDEX']
            clip_filenames = current_app.config['CLIP_FILENAMES']

            results = search_images_fast(query)

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