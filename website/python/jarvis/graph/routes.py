from flask import Blueprint, request, jsonify
import traceback
import jarvis.graph.graph_utils as graph_utils
import jarvis.graph.similarity_graph_utils as similarity_graph_utils
import jarvis.graph.context_graph_utils as context_graph_utils

graph_bp = Blueprint('graph', __name__)

@graph_bp.route('/api/people/graph', methods=['GET'])
def get_social_graph():
    """Get social graph data showing people connections"""
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

@graph_bp.route('/api/people/graph/connections/<person_id>', methods=['GET'])
def get_person_graph_connections(person_id):
    """Get all connections for a specific person"""
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

@graph_bp.route('/api/people/graph/top-connections', methods=['GET'])
def get_top_connections():
    """Get the strongest connections in the graph"""
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

@graph_bp.route('/api/people/similarity-graph', methods=['GET'])
def get_similarity_graph():
    """Get similarity graph data showing people positioned by facial similarity"""
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

@graph_bp.route('/api/people/similar-clusters', methods=['GET'])
def get_similar_clusters():
    """Find pairs of clusters that are very similar and might need to be merged"""
    try:
        min_similarity = float(request.args.get('min_similarity', 0.7))
        max_pairs = int(request.args.get('max_pairs', 50))

        clusters_path = 'faces/clusters.json'
        embeddings_path = 'faces/embeddings.npy'

        print(f"🔍 Finding similar clusters (min_similarity={min_similarity}, max_pairs={max_pairs})...")

        person_embeddings = similarity_graph_utils.compute_person_embeddings(clusters_path, embeddings_path)

        if not person_embeddings:
            return jsonify({'pairs': []}), 200

        similarity_matrix, person_ids = similarity_graph_utils.compute_similarity_matrix(person_embeddings)
        face_data = similarity_graph_utils.load_face_clusters(clusters_path)

        similar_pairs = []
        for i in range(len(person_ids)):
            for j in range(i + 1, len(person_ids)):
                person1 = person_ids[i]
                person2 = person_ids[j]

                similarity = similarity_matrix.get((person1, person2), 0)

                if similarity >= min_similarity:
                    cluster1 = face_data.get(person1, {})
                    cluster2 = face_data.get(person2, {})

                    faces1 = cluster1.get('faces', [])
                    faces2 = cluster2.get('faces', [])

                    if not faces1 or not faces2:
                        continue

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

@graph_bp.route('/api/images/context-graph', methods=['GET'])
def get_context_graph():
    """Get context clustering graph data showing images clustered by semantic content"""
    from flask import current_app
    
    try:
        clip_model = current_app.config['CLIP_MODEL']
        device = current_app.config['DEVICE']
        
        person_filter = request.args.get('person_filter')
        people_filter_str = request.args.get('people_filter')
        min_similarity = float(request.args.get('min_similarity', 0.6))
        num_clusters = int(request.args.get('num_clusters', 10))

        people_filter = None
        if people_filter_str:
            people_filter = [p.strip() for p in people_filter_str.split(',') if p.strip()]

        embeddings_path = 'embeddings.faiss'
        filenames_path = 'filenames.npy'
        clusters_path = 'faces/clusters.json'

        print(f"📊 Generating context graph (people_filter={people_filter}, min_similarity={min_similarity}, num_clusters={num_clusters})...")

        graph_data = context_graph_utils.generate_context_graph(
            embeddings_path=embeddings_path,
            filenames_path=filenames_path,
            clusters_path=clusters_path,
            n_clusters=num_clusters,
            min_similarity=min_similarity,
            person_filter=person_filter,
            people_filter=people_filter,
            clip_model=clip_model,
            device=device
        )

        return jsonify(graph_data), 200

    except Exception as e:
        print(f"❌ Error generating context graph: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'message': 'Failed to generate context graph'
        }), 500