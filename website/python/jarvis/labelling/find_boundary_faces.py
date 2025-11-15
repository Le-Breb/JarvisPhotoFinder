#!/usr/bin/env python3
"""
Find boundary faces - faces that are uncertain in their cluster assignment.
These are faces that are either:
1. Far from their own cluster centroid
2. Close to another cluster's centroid
"""

import numpy as np
import json
import sys
from pathlib import Path

def find_boundary_faces(distance_threshold=0.4, max_results=50):
    """
    Find boundary faces that might be misclassified.
    
    Args:
        distance_threshold: Distance threshold for determining boundary faces
        max_results: Maximum number of boundary faces to return
    
    Returns:
        List of boundary face objects with uncertainty metrics
    """
    # Load embeddings
    embeddings = np.load('faces/embeddings.npy')
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Load clusters
    with open('faces/clusters.json', 'r') as f:
        clusters_data = json.load(f)
    
    # Calculate cluster centroids
    cluster_centroids = {}
    for cluster_id, cluster_data in clusters_data['clusters'].items():
        faces = cluster_data.get('faces', [])
        if not faces:
            continue
        
        embedding_indices = [f['embedding_idx'] for f in faces]
        cluster_embeddings = embeddings_norm[embedding_indices]
        
        # Calculate centroid (normalized)
        centroid = np.mean(cluster_embeddings, axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        
        cluster_centroids[cluster_id] = centroid
    
    # Find boundary faces
    boundary_faces = []
    
    for cluster_id, cluster_data in clusters_data['clusters'].items():
        cluster_name = cluster_data.get('name', f'Person {cluster_id}')
        faces = cluster_data.get('faces', [])
        
        if not faces or cluster_id not in cluster_centroids:
            continue
        
        centroid = cluster_centroids[cluster_id]
        
        for face in faces:
            embedding_idx = face['embedding_idx']
            face_embedding = embeddings_norm[embedding_idx]
            
            # Distance to own centroid (L2 distance)
            distance_to_centroid = np.linalg.norm(face_embedding - centroid)
            
            # Find distance to closest other cluster
            min_distance_to_other = float('inf')
            closest_other_cluster = None
            closest_other_name = None
            
            for other_id, other_centroid in cluster_centroids.items():
                if other_id == cluster_id:
                    continue
                
                dist = np.linalg.norm(face_embedding - other_centroid)
                if dist < min_distance_to_other:
                    min_distance_to_other = dist
                    closest_other_cluster = other_id
                    closest_other_name = clusters_data['clusters'][other_id].get('name', f'Person {other_id}')
            
            # Calculate uncertainty score (higher = more uncertain)
            # Uncertain if far from own cluster OR close to another cluster
            uncertainty_score = distance_to_centroid - min_distance_to_other
            
            # Check if face is uncertain
            is_uncertain = (
                distance_to_centroid > distance_threshold or
                min_distance_to_other < distance_threshold
            )
            
            if is_uncertain:
                # Get up to 3 other faces from the same cluster for context
                context_faces = [f for f in faces if f['face_id'] != face['face_id']]
                if len(context_faces) > 3:
                    # Sample evenly distributed faces
                    step = len(context_faces) // 3
                    context_faces = [context_faces[i * step] for i in range(3)]
                else:
                    context_faces = context_faces[:3]
                
                boundary_faces.append({
                    'face': face,
                    'cluster_id': cluster_id,
                    'cluster_name': cluster_name,
                    'distance_to_centroid': float(distance_to_centroid),
                    'closest_other_cluster': closest_other_cluster,
                    'closest_other_name': closest_other_name,
                    'distance_to_other': float(min_distance_to_other),
                    'uncertainty_score': float(uncertainty_score),
                    'context_faces': context_faces
                })
    
    # Sort by uncertainty score (most uncertain first)
    boundary_faces.sort(key=lambda x: x['uncertainty_score'], reverse=True)
    
    # Limit results
    boundary_faces = boundary_faces[:max_results]
    
    return boundary_faces

if __name__ == '__main__':
    try:
        # Parse command line arguments
        distance_threshold = float(sys.argv[1]) if len(sys.argv) > 1 else 0.4
        max_results = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        
        # Find boundary faces
        boundary_faces = find_boundary_faces(distance_threshold, max_results)
        
        # Output as JSON
        print(json.dumps({
            'boundary_faces': boundary_faces,
            'count': len(boundary_faces)
        }))
        
    except Exception as e:
        print(json.dumps({
            'error': str(e),
            'boundary_faces': []
        }), file=sys.stderr)
        sys.exit(1)
