"""
Social graph utilities for Jarvis Photo Finder
Generates co-occurrence graphs showing people connections
"""

import json
import numpy as np
from collections import defaultdict
from pathlib import Path
import os

# --- NetworkX Import ---
# Import at the top level and set a flag
try:
    import networkx as nx
    from networkx.algorithms import community
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("⚠️  NetworkX not installed. Layout and community detection will be skipped.")
    print("   Install with: pip install networkx")


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


def detect_communities(G):
    """
    Detect communities/clusters in a given NetworkX graph using Louvain algorithm
    Returns a dict mapping node_id to community_id
    """
    if not NETWORKX_AVAILABLE:
        return {}
        
    try:
        # Detect communities using Louvain algorithm
        communities_set = community.louvain_communities(G, seed=42, weight='weight')
        
        # Create mapping from node_id to community_id
        node_to_community = {}
        for community_id, community_nodes in enumerate(communities_set):
            for node_id in community_nodes:
                node_to_community[node_id] = community_id
        
        print(f"🔍 Detected {len(communities_set)} communities")
        for i, comm in enumerate(communities_set):
            print(f"  Community {i}: {len(comm)} people")
        
        return node_to_community
        
    except Exception as e:
        print(f"❌ Error detecting communities: {e}")
        # Fallback: return empty map
        return {}


def generate_social_graph(clusters_path='faces/clusters.json'):
    """
    Generate social graph data from face clusters with community detection
    AND pre-computed physics layout.
    
    Returns:
        dict: Graph data with nodes (inc. x, y coords), links, communities, and stats
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
            photo_path = face.get('image') or face.get('filepath') or face.get('filename')
            if not photo_path:
                continue
            photo_to_people[photo_path].add(person_id)
            if photo_path not in people_photos[person_id]:
                people_photos[person_id].append(photo_path)
            people_faces[person_id].append(face)
    
    print(f"📸 Found {len(photo_to_people)} unique photos with people")
    
    # Calculate co-occurrence matrix
    co_occurrence = defaultdict(lambda: defaultdict(int))
    
    for photo, people in photo_to_people.items():
        people_list = list(people)
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
    
    # --- NetworkX Graph Creation, Community Detection & Layout ---
    node_to_community = {}
    pos = {}
    num_communities = 1 # Default
    
    if NETWORKX_AVAILABLE:
        # Create graph
        G = nx.Graph()
        for node in nodes:
            G.add_node(node['id'])
        for link in links:
            G.add_edge(link['source'], link['target'], weight=link['weight'])
        
        # 1. Detect communities
        node_to_community = detect_communities(G)
        num_communities = len(set(node_to_community.values()))
        if num_communities == 0: num_communities = 1
        
        # 2. Pre-compute layout
        print("Starting layout simulation... (This may take a moment)")
        # k controls spacing, iterations for stability. Tune k as needed.
        pos = nx.spring_layout(G, k=0.8, iterations=75, weight='weight', seed=42)
        print("✅ Layout simulation finished.")
    
    else:
        # Fallback if networkx is not installed
        node_to_community = {node['id']: 0 for node in nodes}
    
    # --- Add community and position data to nodes ---
    # These dimensions match your React component's viewBox
    WIDTH = 2200
    HEIGHT = 1400
    
    for node in nodes:
        node_id = node['id']
        node['community'] = node_to_community.get(node_id, 0)
        
        if node_id in pos:
            # Scale positions from [0, 1] to [0, WIDTH/HEIGHT]
            node['x'] = pos[node_id][0] * WIDTH
            node['y'] = pos[node_id][1] * HEIGHT
        else:
            # Fallback for nodes (e.g., if networkx failed or node had no pos)
            # Give it a random position so it doesn't stack at [0,0]
            node['x'] = np.random.rand() * WIDTH
            node['y'] = np.random.rand() * HEIGHT
    
    # --- Calculate statistics ---
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


def get_person_connections(person_id, clusters_path='faces/clusters.json', graph_data=None):
    """
    Get all connections (co-occurrences) for a specific person
    """
    if graph_data:
        graph = graph_data
    else:
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
    
    connections.sort(key=lambda x: x['weight'], reverse=True)
    return connections


def get_strongest_connections(top_n=10, clusters_path='faces/clusters.json', graph_data=None):
    """
    Get the strongest connections in the social graph
    """
    if graph_data:
        graph = graph_data
    else:
        graph = generate_social_graph(clusters_path)
    
    sorted_links = sorted(graph['links'], key=lambda x: x['weight'], reverse=True)
    person_names = {node['id']: node['name'] for node in graph['nodes']}
    
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
    print("🧪 Testing social graph generation...")
    # Generate graph ONCE
    graph = generate_social_graph()
    
    print(f"\n📊 Graph Statistics:")
    print(f"  People: {graph['stats']['total_people']}")
    print(f"  Connections: {graph['stats']['total_connections']}")
    print(f"  Photos: {graph['stats']['total_photos']}")
    print(f"  Communities: {graph['stats']['total_communities']}")
    
    if graph['stats']['total_connections'] > 0:
        print(f"\n🔗 Top 5 Strongest Connections:")
        # Pass the already-computed graph to avoid re-computing
        top = get_strongest_connections(top_n=5, graph_data=graph)
        for i, conn in enumerate(top, 1):
            print(f"  {i}. {conn['person1_name']} ↔ {conn['person2_name']}: {conn['shared_photos']} photos")