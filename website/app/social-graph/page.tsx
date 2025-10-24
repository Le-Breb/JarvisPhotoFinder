'use client';

import { useEffect, useState, useRef, useMemo } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog';
import { Slider } from '@/components/ui/slider';
import { ZoomIn, ZoomOut, Maximize2, Filter, Maximize, Minimize } from 'lucide-react';
import * as d3 from 'd3';

interface Node {
  id: string;
  name: string;
  photo_count: number;
  representative_face: string;
  total_faces: number;
  community: number;
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
  fx?: number | null;
  fy?: number | null;
}

interface Link {
  source: string | Node;
  target: string | Node;
  weight: number;
  value: number;
}

interface GraphData {
  nodes: Node[];
  links: Link[];
  communities: Record<string, number>;
  stats: {
    total_people: number;
    total_connections: number;
    total_photos: number;
    total_communities: number;
  };
}

interface PersonFace {
  filename: string;
  bbox: number[];
  face_id?: string;
}

const PYTHON_API_URL = process.env.NEXT_PUBLIC_PYTHON_API_URL || 'http://localhost:5000';

const COMMUNITY_COLORS = [
  'hsl(221.2, 83.2%, 53.3%)', 'hsl(142, 76%, 36%)', 'hsl(262, 83%, 58%)',
  'hsl(24, 95%, 53%)', 'hsl(346, 77%, 50%)', 'hsl(199, 89%, 48%)',
  'hsl(48, 96%, 53%)', 'hsl(280, 65%, 60%)', 'hsl(168, 76%, 42%)',
  'hsl(32, 95%, 44%)',
];

export default function SocialGraphPage() {
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [personFaces, setPersonFaces] = useState<PersonFace[]>([]);
  const [loadingPhotos, setLoadingPhotos] = useState(false);
  const [showPhotoDialog, setShowPhotoDialog] = useState(false);
  const [linkThreshold, setLinkThreshold] = useState(1);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [simulationProgress, setSimulationProgress] = useState(0);
  const svgRef = useRef<SVGSVGElement>(null);
  const zoomRef = useRef<any>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch(`${PYTHON_API_URL}/api/people/graph`)
      .then(res => {
        if (!res.ok) throw new Error('Failed to fetch graph data');
        return res.json();
      })
      .then(data => {
        setGraphData(data);
        setLoading(false);
      })
      .catch(error => {
        console.error('Error loading graph:', error);
        setError(error.message);
        setLoading(false);
      });
  }, []);

  const fetchPersonPhotos = async (personId: string) => {
    setLoadingPhotos(true);
    
    try {
      const response = await fetch(`/api/people/${personId}`);
      
      if (!response.ok) {
        throw new Error('Failed to fetch person data');
      }
      
      const data = await response.json();
      const faces = data.person?.faces || [];
      
      setPersonFaces(faces);
      setShowPhotoDialog(true);
    } catch (error) {
      console.error('❌ Error fetching person photos:', error);
      alert('Impossible de charger les photos pour cette personne');
      setPersonFaces([]);
    } finally {
      setLoadingPhotos(false);
    }
  };

  const getUniquePhotos = () => {
    const photoMap = new Map<string, PersonFace[]>();
    
    personFaces.forEach(face => {
      if (!face.filename) return;
      
      const filename = face.filename.replace('images/', '');
      
      if (!photoMap.has(filename)) {
        photoMap.set(filename, []);
      }
      photoMap.get(filename)!.push(face);
    });
    
    return Array.from(photoMap.entries());
  };

  const getCommunityColor = (community: number) => {
    return COMMUNITY_COLORS[community % COMMUNITY_COLORS.length];
  };

  const filteredLinks = useMemo(() => {
    if (!graphData) return [];
    return graphData.links.filter(link => link.weight >= linkThreshold);
  }, [graphData, linkThreshold]);

  // Toggle fullscreen
  const toggleFullscreen = () => {
    if (!containerRef.current) return;

    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen().then(() => {
        setIsFullscreen(true);
      }).catch(err => {
        console.error('Error attempting to enable fullscreen:', err);
      });
    } else {
      document.exitFullscreen().then(() => {
        setIsFullscreen(false);
      });
    }
  };

  // Listen for fullscreen changes
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
    };
  }, []);

  useEffect(() => {
    if (!svgRef.current || !graphData || graphData.nodes.length === 0) return;

    // Larger canvas - especially in fullscreen
    const width = isFullscreen ? window.innerWidth : 2200;
    const height = isFullscreen ? window.innerHeight : 1400;

    d3.select(svgRef.current).selectAll('*').remove();

    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', `0 0 ${width} ${height}`);

    const defs = svg.append('defs');
    
    Array.from(new Set(graphData.nodes.map(n => n.community))).forEach(communityId => {
      const color = getCommunityColor(communityId);
      
      const gradient = defs.append('radialGradient')
        .attr('id', `gradient-${communityId}`);
      gradient.append('stop')
        .attr('offset', '0%')
        .attr('stop-color', d3.color(color)!.brighter(0.8).toString());
      gradient.append('stop')
        .attr('offset', '100%')
        .attr('stop-color', color);
    });

    const selectedGradient = defs.append('radialGradient')
      .attr('id', 'selected-gradient');
    selectedGradient.append('stop')
      .attr('offset', '0%')
      .attr('stop-color', 'hsl(221.2, 83.2%, 73.3%)');
    selectedGradient.append('stop')
      .attr('offset', '100%')
      .attr('stop-color', 'hsl(221.2, 83.2%, 53.3%)');

    const filter = defs.append('filter')
      .attr('id', 'drop-shadow')
      .attr('height', '130%');
    filter.append('feGaussianBlur')
      .attr('in', 'SourceAlpha')
      .attr('stdDeviation', 2);
    filter.append('feOffset')
      .attr('dx', 1)
      .attr('dy', 1)
      .attr('result', 'offsetblur');
    const feMerge = filter.append('feMerge');
    feMerge.append('feMergeNode');
    feMerge.append('feMergeNode')
      .attr('in', 'SourceGraphic');

    const container = svg.append('g');

    const zoom = d3.zoom()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        container.attr('transform', event.transform);
      });

    svg.call(zoom as any);
    zoomRef.current = zoom;

    // Prepare nodes and links
    const nodes = graphData.nodes.map(d => ({ ...d }));
    const links = filteredLinks.map(d => ({ ...d }));

    const maxWeight = Math.max(...links.map(l => l.weight), 1);

    // Create force simulation with MUCH stronger repulsion
    const simulation = d3.forceSimulation(nodes as any)
      .force('link', d3.forceLink(links)
        .id((d: any) => d.id)
        .distance(d => {
          const weight = (d as any).weight;
          const normalized = weight / maxWeight;
          // Much longer distances
          return 250 * (1 - normalized * 0.5);
        })
        .strength(d => {
          const weight = (d as any).weight;
          const normalized = weight / maxWeight;
          return 0.2 + normalized * 0.4;
        })
      )
      .force('charge', d3.forceManyBody()
        .strength(-1200) // Much stronger repulsion
        .distanceMax(600)
      )
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide()
        .radius(d => Math.sqrt((d as any).photo_count) * 0.5 + 40) // Larger collision radius
      )
      // Community clustering
      .force('x', d3.forceX(width / 2).strength(0.02))
      .force('y', d3.forceY(height / 2).strength(0.02))
      .alphaDecay(0.005) // Slower decay for better settling
      .velocityDecay(0.4); // More friction

    // Draw community backgrounds
    const communities = d3.group(nodes, d => d.community);
    
    const hullGroup = container.append('g')
      .attr('class', 'hulls');

    const hulls = hullGroup.selectAll('path')
      .data(Array.from(communities.entries()))
      .join('path')
      .attr('fill', ([communityId]) => {
        const color = getCommunityColor(communityId as number);
        return d3.color(color)!.copy({opacity: 0.05}).toString();
      })
      .attr('stroke', ([communityId]) => getCommunityColor(communityId as number))
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '8,4')
      .attr('stroke-opacity', 0.3);

    // Draw links
    const linkGroup = container.append('g')
      .attr('class', 'links');

    const intraCommunityLinks = links.filter(l => {
      const sourceNode = nodes.find(n => n.id === (typeof l.source === 'string' ? l.source : (l.source as any).id));
      const targetNode = nodes.find(n => n.id === (typeof l.target === 'string' ? l.target : (l.target as any).id));
      return sourceNode && targetNode && sourceNode.community === targetNode.community;
    });
    
    const interCommunityLinks = links.filter(l => {
      const sourceNode = nodes.find(n => n.id === (typeof l.source === 'string' ? l.source : (l.source as any).id));
      const targetNode = nodes.find(n => n.id === (typeof l.target === 'string' ? l.target : (l.target as any).id));
      return sourceNode && targetNode && sourceNode.community !== targetNode.community;
    });

    const interLinks = linkGroup.selectAll('line.inter')
      .data(interCommunityLinks)
      .join('line')
      .attr('class', 'inter')
      .attr('stroke', 'hsl(215.4, 16.3%, 46.9%)')
      .attr('stroke-opacity', d => 0.08 + (d.weight / maxWeight) * 0.12)
      .attr('stroke-width', d => 0.5 + Math.sqrt(d.weight) * 0.4)
      .attr('stroke-linecap', 'round');

    const intraLinks = linkGroup.selectAll('line.intra')
      .data(intraCommunityLinks)
      .join('line')
      .attr('class', 'intra')
      .attr('stroke-opacity', d => 0.15 + (d.weight / maxWeight) * 0.25)
      .attr('stroke-width', d => 0.8 + Math.sqrt(d.weight) * 0.6)
      .attr('stroke-linecap', 'round');

    // Draw nodes - EVEN SMALLER
    const nodeGroup = container.append('g')
      .attr('class', 'nodes');

    const node = nodeGroup.selectAll('g')
      .data(nodes)
      .join('g')
      .attr('class', 'node')
      .style('cursor', 'pointer');

    node.append('circle')
      .attr('r', d => Math.sqrt(d.photo_count) * 0.5 + 6)
      .attr('fill', d => {
        const color = getCommunityColor(d.community);
        return d.id === selectedNode?.id 
          ? 'hsla(221.2, 83.2%, 53.3%, 0.2)' 
          : d3.color(color)!.copy({opacity: 0.12}).toString();
      })
      .attr('class', 'glow');

    node.append('circle')
      .attr('r', d => Math.sqrt(d.photo_count) * 0.5 + 4)
      .attr('fill', d => d.id === selectedNode?.id 
        ? 'url(#selected-gradient)' 
        : `url(#gradient-${d.community})`)
      .attr('stroke', 'hsl(0, 0%, 100%)')
      .attr('stroke-width', 1.5)
      .attr('filter', 'url(#drop-shadow)')
      .attr('class', 'main-circle');

    node.append('circle')
      .attr('r', 8)
      .attr('cx', d => Math.sqrt(d.photo_count) * 0.5 + 2)
      .attr('cy', d => -(Math.sqrt(d.photo_count) * 0.5 + 2))
      .attr('fill', 'hsl(0, 84.2%, 60.2%)')
      .attr('stroke', 'hsl(0, 0%, 100%)')
      .attr('stroke-width', 1.5);

    node.append('text')
      .attr('x', d => Math.sqrt(d.photo_count) * 0.5 + 2)
      .attr('y', d => -(Math.sqrt(d.photo_count) * 0.5) + 2)
      .attr('text-anchor', 'middle')
      .attr('font-size', 8)
      .attr('font-weight', 'bold')
      .attr('fill', 'hsl(0, 0%, 100%)')
      .text(d => d.photo_count);

    node.append('rect')
      .attr('x', d => -d.name.length * 2.5)
      .attr('y', d => Math.sqrt(d.photo_count) * 0.5 + 10)
      .attr('width', d => d.name.length * 5)
      .attr('height', 14)
      .attr('rx', 3)
      .attr('fill', 'hsla(0, 0%, 100%, 0.95)')
      .attr('stroke', 'hsl(214.3, 31.8%, 91.4%)')
      .attr('stroke-width', 0.5);

    node.append('text')
      .text(d => d.name)
      .attr('y', d => Math.sqrt(d.photo_count) * 0.5 + 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', 10)
      .attr('font-weight', '600')
      .attr('fill', 'hsl(222.2, 84%, 4.9%)')
      .style('pointer-events', 'none')
      .style('user-select', 'none');

    node.append('title')
      .text(d => `${d.name}\nGroupe ${d.community + 1}\n${d.photo_count} photos\n${d.total_faces} visages`);

    // Update link colors based on community
    function updateLinkColors() {
      intraLinks.attr('stroke', (d: any) => {
        const source = d.source as Node;
        return getCommunityColor(source.community);
      });
    }

    // Update positions on tick
    simulation.on('tick', () => {
      // Update progress
      const alpha = simulation.alpha();
      const progress = Math.max(0, Math.min(100, (1 - alpha) * 100));
      setSimulationProgress(Math.round(progress));

      // Update link positions
      interLinks
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      intraLinks
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      // Update node positions
      node.attr('transform', (d: any) => `translate(${d.x},${d.y})`);

      // Update community hulls
      hulls.attr('d', ([communityId, communityNodes]) => {
        if (communityNodes.length < 3) return null;
        
        const points = communityNodes.map(n => [n.x!, n.y!] as [number, number]);
        const hull = d3.polygonHull(points);
        
        if (!hull || hull.length < 3) return null;
        
        const centroid = d3.polygonCentroid(hull);
        const expansion = 70;
        const expanded = hull.map(point => {
          const dx = point[0] - centroid[0];
          const dy = point[1] - centroid[1];
          const distance = Math.sqrt(dx * dx + dy * dy);
          if (distance === 0) return point;
          return [
            point[0] + (dx / distance) * expansion,
            point[1] + (dy / distance) * expansion
          ];
        });
        
        return `M${expanded.map(p => p.join(',')).join('L')}Z`;
      });

      updateLinkColors();
    });

    // Stop simulation when stabilized
    simulation.on('end', () => {
      console.log('✅ Simulation stabilisée');
      setSimulationProgress(100);
    });

    // Auto-stop after alpha threshold
    const checkStabilization = setInterval(() => {
      if (simulation.alpha() < 0.01) {
        simulation.stop();
        clearInterval(checkStabilization);
      }
    }, 100);

    // Interactions
    node.on('mouseenter', function(event, d) {
      const connectedIds = new Set<string>();
      links.forEach(link => {
        const sourceId = (link.source as any).id || link.source;
        const targetId = (link.target as any).id || link.target;
        if (sourceId === d.id) connectedIds.add(targetId);
        if (targetId === d.id) connectedIds.add(sourceId);
      });

      node.style('opacity', (n: Node) => 
        n.id === d.id || connectedIds.has(n.id) ? 1 : 0.2
      );

      linkGroup.selectAll('line')
        .style('opacity', (l: any) => {
          const sourceId = l.source.id;
          const targetId = l.target.id;
          return (sourceId === d.id || targetId === d.id) ? 1 : 0.05;
        })
        .style('stroke-width', (l: any) => {
          const sourceId = l.source.id;
          const targetId = l.target.id;
          if (sourceId === d.id || targetId === d.id) {
            return (2 + Math.sqrt(l.weight) * 1.2) + 'px';
          }
          return null;
        });

      d3.select(this).select('.main-circle')
        .transition()
        .duration(200)
        .attr('r', Math.sqrt(d.photo_count) * 0.5 + 7);
      
      d3.select(this).select('.glow')
        .transition()
        .duration(200)
        .attr('r', Math.sqrt(d.photo_count) * 0.5 + 12)
        .attr('fill-opacity', 0.25);
    })
    .on('mouseleave', function(event, d) {
      node.style('opacity', 1);
      linkGroup.selectAll('line')
        .style('opacity', null)
        .style('stroke-width', null);

      if (d.id !== selectedNode?.id) {
        d3.select(this).select('.main-circle')
          .transition()
          .duration(200)
          .attr('r', Math.sqrt(d.photo_count) * 0.5 + 4);
        
        d3.select(this).select('.glow')
          .transition()
          .duration(200)
          .attr('r', Math.sqrt(d.photo_count) * 0.5 + 6)
          .attr('fill-opacity', 0.12);
      }
    });

    node.on('click', (event, d) => {
      event.stopPropagation();
      setSelectedNode(d);
      fetchPersonPhotos(d.id);
      
      node.select('.main-circle')
        .transition()
        .duration(300)
        .attr('fill', (n: any) => n.id === d.id 
          ? 'url(#selected-gradient)' 
          : `url(#gradient-${n.community})`)
        .attr('r', (n: any) => Math.sqrt(n.photo_count) * 0.5 + (n.id === d.id ? 7 : 4));
      
      node.select('.glow')
        .transition()
        .duration(300)
        .attr('fill', (n: any) => {
          if (n.id === d.id) return 'hsla(221.2, 83.2%, 53.3%, 0.25)';
          const color = getCommunityColor(n.community);
          return d3.color(color)!.copy({opacity: 0.12}).toString();
        })
        .attr('r', (n: any) => Math.sqrt(n.photo_count) * 0.5 + (n.id === d.id ? 12 : 6));
    });

    return () => {
      simulation.stop();
      clearInterval(checkStabilization);
    };

  }, [graphData, selectedNode?.id, filteredLinks, isFullscreen]);

  const handleZoomIn = () => {
    if (svgRef.current && zoomRef.current) {
      d3.select(svgRef.current)
        .transition()
        .duration(300)
        .call(zoomRef.current.scaleBy, 1.5);
    }
  };

  const handleZoomOut = () => {
    if (svgRef.current && zoomRef.current) {
      d3.select(svgRef.current)
        .transition()
        .duration(300)
        .call(zoomRef.current.scaleBy, 0.67);
    }
  };

  const handleResetZoom = () => {
    if (svgRef.current && zoomRef.current) {
      d3.select(svgRef.current)
        .transition()
        .duration(500)
        .call(zoomRef.current.transform, d3.zoomIdentity);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-lg">Chargement du graphe social...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-screen">
        <Card className="p-6 max-w-md">
          <h2 className="text-xl font-bold text-destructive mb-2">Erreur</h2>
          <p className="mb-2">{error}</p>
          <p className="text-sm text-muted-foreground">
            Assurez-vous que l'API Python tourne sur {PYTHON_API_URL}
          </p>
        </Card>
      </div>
    );
  }

  if (!graphData || graphData.nodes.length === 0) {
    return (
      <div className="flex items-center justify-center h-screen">
        <Card className="p-6 max-w-md">
          <h2 className="text-xl font-bold mb-2">Aucune donnée</h2>
          <p>Aucune personne trouvée dans votre collection de photos.</p>
        </Card>
      </div>
    );
  }

  const uniquePhotos = getUniquePhotos();

  const communitySizes = graphData.nodes.reduce((acc, node) => {
    acc[node.community] = (acc[node.community] || 0) + 1;
    return acc;
  }, {} as Record<number, number>);

  const maxLinkWeight = Math.max(...graphData.links.map(l => l.weight));

  return (
    <div className="container mx-auto p-6 max-w-[2400px]">
      {!isFullscreen && (
        <>
          <div className="mb-6">
            <h1 className="text-3xl font-bold mb-2">Graphe Social</h1>
            <p className="text-muted-foreground">
              Visualisation des relations sociales détectées automatiquement. 
              Le graphe se stabilise automatiquement grâce à une simulation physique.
            </p>
            
            <div className="flex gap-4 mt-4 flex-wrap">
              <Card className="p-4 bg-card">
                <div className="text-2xl font-bold text-primary">{graphData.stats.total_people}</div>
                <div className="text-sm text-muted-foreground">Personnes</div>
              </Card>
              <Card className="p-4 bg-card">
                <div className="text-2xl font-bold text-primary">{filteredLinks.length}</div>
                <div className="text-sm text-muted-foreground">Connexions visibles</div>
              </Card>
              <Card className="p-4 bg-card">
                <div className="text-2xl font-bold text-primary">{graphData.stats.total_photos}</div>
                <div className="text-sm text-muted-foreground">Photos</div>
              </Card>
              <Card className="p-4 bg-card">
                <div className="text-2xl font-bold text-primary">{graphData.stats.total_communities}</div>
                <div className="text-sm text-muted-foreground">Communautés</div>
              </Card>
            </div>

            <Card className="p-4 mt-4 bg-card">
              <div className="flex items-center gap-4">
                <Filter className="h-5 w-5 text-muted-foreground" />
                <div className="flex-1">
                  <div className="flex justify-between mb-2">
                    <label className="text-sm font-medium">
                      Filtrer les connexions faibles
                    </label>
                    <span className="text-sm text-muted-foreground">
                      Min: {linkThreshold} photo{linkThreshold > 1 ? 's' : ''} partagée{linkThreshold > 1 ? 's' : ''}
                    </span>
                  </div>
                  <Slider
                    value={[linkThreshold]}
                    onValueChange={(values) => setLinkThreshold(values[0])}
                    min={1}
                    max={Math.max(10, Math.floor(maxLinkWeight / 2))}
                    step={1}
                    className="w-full"
                  />
                  <div className="text-xs text-muted-foreground mt-2">
                    Augmentez pour réduire le bruit visuel ({graphData.stats.total_connections} connexions au total)
                  </div>
                </div>
              </div>
            </Card>

            <Card className="p-4 mt-4 bg-card">
              <h3 className="font-semibold mb-3">Communautés détectées</h3>
              <div className="flex gap-3 flex-wrap">
                {Object.entries(communitySizes).map(([communityId, size]) => (
                  <div key={communityId} className="flex items-center gap-2 bg-secondary px-3 py-2 rounded-lg">
                    <div 
                      className="w-4 h-4 rounded-full"
                      style={{ backgroundColor: getCommunityColor(parseInt(communityId)) }}
                    />
                    <span className="text-sm font-medium">
                      Groupe {parseInt(communityId) + 1}
                    </span>
                    <span className="text-xs text-muted-foreground">
                      ({size})
                    </span>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        </>
      )}

      <Card 
        ref={containerRef}
        className={`p-4 relative bg-card ${isFullscreen ? 'h-screen' : ''}`}
      >
        <div className="absolute top-6 right-6 z-10 flex gap-2">
          <Button variant="outline" size="icon" onClick={handleZoomIn} title="Zoom avant">
            <ZoomIn className="h-4 w-4" />
          </Button>
          <Button variant="outline" size="icon" onClick={handleZoomOut} title="Zoom arrière">
            <ZoomOut className="h-4 w-4" />
          </Button>
          <Button variant="outline" size="icon" onClick={handleResetZoom} title="Réinitialiser">
            <Maximize2 className="h-4 w-4" />
          </Button>
          <Button variant="outline" size="icon" onClick={toggleFullscreen} title={isFullscreen ? "Quitter plein écran" : "Plein écran"}>
            {isFullscreen ? <Minimize className="h-4 w-4" /> : <Maximize className="h-4 w-4" />}
          </Button>
        </div>

        {simulationProgress < 100 && (
          <div className="absolute top-6 left-6 z-10 bg-card/90 backdrop-blur-sm px-4 py-2 rounded-lg border shadow-lg">
            <div className="flex items-center gap-3">
              <div className="text-sm font-medium text-muted-foreground">
                Stabilisation: {simulationProgress}%
              </div>
              <div className="w-32 h-2 bg-secondary rounded-full overflow-hidden">
                <div 
                  className="h-full bg-primary transition-all duration-300"
                  style={{ width: `${simulationProgress}%` }}
                />
              </div>
            </div>
          </div>
        )}
        
        <div className={`w-full overflow-auto rounded-lg border bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 ${isFullscreen ? 'h-full' : ''}`}>
          <svg 
            ref={svgRef} 
            className="w-full h-auto"
          />
        </div>
        
        {!isFullscreen && (
          <div className="mt-4 text-sm text-muted-foreground">
            <p className="font-medium mb-2">Légende:</p>
            <div className="grid grid-cols-2 gap-2">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: getCommunityColor(0) }}></div>
                <span>Connexion dans la communauté</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-muted-foreground"></div>
                <span>Connexion entre communautés</span>
              </div>
            </div>
          </div>
        )}
      </Card>

      <Dialog open={showPhotoDialog} onOpenChange={setShowPhotoDialog}>
        <DialogContent className="max-w-6xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="text-2xl flex items-center gap-2">
              {selectedNode?.name}
              <div 
                className="w-4 h-4 rounded-full"
                style={{ backgroundColor: selectedNode ? getCommunityColor(selectedNode.community) : '' }}
              />
            </DialogTitle>
            <DialogDescription>
              Groupe {selectedNode ? selectedNode.community + 1 : ''} • {uniquePhotos.length} photo{uniquePhotos.length > 1 ? 's' : ''} • {personFaces.length} visage{personFaces.length > 1 ? 's' : ''}
            </DialogDescription>
          </DialogHeader>
          
          {loadingPhotos ? (
            <div className="flex items-center justify-center py-12">
              <div className="text-lg">Chargement des photos...</div>
            </div>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 mt-4">
              {uniquePhotos.map(([filename, faces], index) => (
                <div key={index} className="space-y-2">
                  <div className="relative aspect-square overflow-hidden rounded-lg border-2 border-border hover:border-primary transition-colors group">
                    <img
                      src={`/api/images/${filename}`}
                      alt={`Photo ${index + 1}`}
                      className="w-full h-full object-cover"
                      loading="lazy"
                    />
                    <div className="absolute top-2 right-2 bg-black/70 text-white text-xs px-2 py-1 rounded-full">
                      {faces.length} visage{faces.length > 1 ? 's' : ''}
                    </div>
                  </div>
                  
                  {faces.length > 0 && (
                    <div className={`grid gap-1 ${faces.length === 1 ? 'grid-cols-1' : 'grid-cols-2'}`}>
                      {faces.slice(0, 4).map((face, faceIdx) => (
                        <div key={faceIdx} className="relative aspect-square overflow-hidden rounded border border-border hover:border-primary transition-colors">
                          <img
                            src={`/api/face/${filename}?bbox=${face.bbox.join(',')}&padding=50`}
                            alt={`Visage ${faceIdx + 1}`}
                            className="w-full h-full object-cover"
                            loading="lazy"
                          />
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
          
          {!loadingPhotos && personFaces.length === 0 && (
            <div className="text-center py-12 text-muted-foreground">
              Aucune photo trouvée pour cette personne
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}