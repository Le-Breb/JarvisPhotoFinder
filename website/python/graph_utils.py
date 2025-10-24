"""
Social graph utilities for Jarvis Photo Finder
Generates co-occurrence graphs showing people connections
"""

import json
import numpy as np
from collections import defaultdict
from pathlib import Path
import os


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


def detect_communities(nodes, links):
    """
    Detect communities/clusters in the social graph using Louvain algorithm
    Returns a dict mapping node_id to community_id
    """
    try:
        import networkx as nx
        from networkx.algorithms import community
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        for node in nodes:
            G.add_node(node['id'])
        
        # Add edges with weights
        for link in links:
            G.add_edge(link['source'], link['target'], weight=link['weight'])
        
        # Detect communities using Louvain algorithm
        communities = community.louvain_communities(G, seed=42, weight='weight')
        
        # Create mapping from node_id to community_id
        node_to_community = {}
        for community_id, community_nodes in enumerate(communities):
            for node_id in community_nodes:
                node_to_community[node_id] = community_id
        
        print(f"🔍 Detected {len(communities)} communities")
        for i, comm in enumerate(communities):
            print(f"  Community {i}: {len(comm)} people")
        
        return node_to_community
        
    except ImportError:
        print("⚠️  NetworkX not installed. Install with: pip install networkx")
        # Fallback: assign everyone to community 0
        return {node['id']: 0 for node in nodes}
    except Exception as e:
        print(f"❌ Error detecting communities: {e}")
        # Fallback: assign everyone to community 0
        return {node['id']: 0 for node in nodes}


def generate_social_graph(clusters_path='faces/clusters.json'):
    """
    Generate social graph data from face clusters with community detection
    
    Returns:
        dict: Graph data with nodes, links, communities, and statistics
    """
    face_data = load_face_clusters(clusters_path)
    
    if not face_data:
        return {
            'nodes': [],
            'links': [],
            'communities': {},
            'stats': {
                'total_people': 0,
                'total_connections': 0,
                'total_photos': 0,
                'total_communities': 0
            }
        }
    
    # Build photo -> people mapping
    photo_to_people = defaultdict(set)
    people_photos = defaultdict(list)
    people_faces = defaultdict(list)
    
    print(f"📊 Processing {len(face_data)} people clusters...")
    
    for person_id, person_data in face_data.items():
        faces = person_data.get('faces', [])
        
        for face in faces:
            # Handle different possible keys for image path
            photo_path = face.get('image') or face.get('filepath') or face.get('filename')
            if not photo_path:
                continue
                
            photo_to_people[photo_path].add(person_id)
            
            # Track photos for this person (deduplicate)
            if photo_path not in people_photos[person_id]:
                people_photos[person_id].append(photo_path)
            
            people_faces[person_id].append(face)
    
    print(f"📸 Found {len(photo_to_people)} unique photos with people")
    
    # Calculate co-occurrence matrix
    co_occurrence = defaultdict(lambda: defaultdict(int))
    
    for photo, people in photo_to_people.items():
        people_list = list(people)
        # For each pair of people in the same photo
        for i, person1 in enumerate(people_list):
            for person2 in people_list[i+1:]:
                co_occurrence[person1][person2] += 1
                co_occurrence[person2][person1] += 1
    
    # Build nodes
    nodes = []
    for person_id, person_data in face_data.items():
        faces = people_faces.get(person_id, [])
        if not faces:
            continue
        
        # Get representative face
        representative_face = None
        if faces:
            first_face = faces[0]
            representative_face = (
                first_face.get('image') or 
                first_face.get('filepath') or 
                first_face.get('filename')
            )
        
        nodes.append({
            'id': person_id,
            'name': person_data.get('name', f'Person {person_id}'),
            'photo_count': len(people_photos[person_id]),
            'total_faces': len(faces),
            'representative_face': representative_face
        })
    
    print(f"👥 Created {len(nodes)} people nodes")
    
    # Build links (edges)
    links = []
    edge_set = set()
    
    for person1, connections in co_occurrence.items():
        for person2, weight in connections.items():
            # Avoid duplicate edges
            edge_key = tuple(sorted([person1, person2]))
            if edge_key not in edge_set and weight > 0:
                edge_set.add(edge_key)
                links.append({
                    'source': person1,
                    'target': person2,
                    'weight': weight,
                    'value': weight
                })
    
    print(f"🔗 Created {len(links)} connections")
    
    # Detect communities
    node_to_community = detect_communities(nodes, links)
    
    # Add community info to nodes
    for node in nodes:
        node['community'] = node_to_community.get(node['id'], 0)
    
    # Calculate statistics
    num_communities = len(set(node_to_community.values()))
    
    stats = {
        'total_people': len(nodes),
        'total_connections': len(links),
        'total_photos': len(photo_to_people),
        'total_communities': num_communities,
        'avg_photos_per_person': np.mean([len(photos) for photos in people_photos.values()]) if people_photos else 0,
        'avg_connections_per_person': (2 * len(links) / len(nodes)) if nodes else 0
    }
    
    return {
        'nodes': nodes,
        'links': links,
        'communities': node_to_community,
        'stats': stats
    }


def get_person_connections(person_id, clusters_path='faces/clusters.json'):
    """
    Get all connections (co-occurrences) for a specific person
    
    Args:
        person_id: ID of the person
        clusters_path: Path to clusters JSON file
        
    Returns:
        list: List of connections with weight
    """
    graph = generate_social_graph(clusters_path)
    
    connections = []
    for link in graph['links']:
        if link['source'] == person_id:
            connections.append({
                'person_id': link['target'],
                'weight': link['weight']
            })
        elif link['target'] == person_id:
            connections.append({
                'person_id': link['source'],
                'weight': link['weight']
            })
    
    # Sort by weight (most connections first)
    connections.sort(key=lambda x: x['weight'], reverse=True)
    
    return connections


def get_strongest_connections(top_n=10, clusters_path='faces/clusters.json'):
    """
    Get the strongest connections in the social graph
    
    Args:
        top_n: Number of top connections to return
        clusters_path: Path to clusters JSON file
        
    Returns:
        list: Top N strongest connections
    """
    graph = generate_social_graph(clusters_path)
    
    # Sort links by weight
    sorted_links = sorted(graph['links'], key=lambda x: x['weight'], reverse=True)
    
    # Get person names
    person_names = {node['id']: node['name'] for node in graph['nodes']}
    
    # Format results
    top_connections = []
    for link in sorted_links[:top_n]:
        top_connections.append({
            'person1_id': link['source'],
            'person1_name': person_names.get(link['source'], 'Unknown'),
            'person2_id': link['target'],
            'person2_name': person_names.get(link['target'], 'Unknown'),
            'shared_photos': link['weight']
        })
    
    return top_connections


if __name__ == '__main__':
    # Test the graph generation
    print("🧪 Testing social graph generation...")
    graph = generate_social_graph()
    
    print(f"\n📊 Graph Statistics:")
    print(f"  People: {graph['stats']['total_people']}")
    print(f"  Connections: {graph['stats']['total_connections']}")
    print(f"  Photos: {graph['stats']['total_photos']}")
    print(f"  Communities: {graph['stats']['total_communities']}")
    
    if graph['stats']['total_connections'] > 0:
        print(f"\n🔗 Top 5 Strongest Connections:")
        top = get_strongest_connections(top_n=5)
        for i, conn in enumerate(top, 1):
            print(f"  {i}. {conn['person1_name']} ↔ {conn['person2_name']}: {conn['shared_photos']} photos")