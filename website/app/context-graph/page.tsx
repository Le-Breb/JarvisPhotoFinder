"use client";

import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import * as d3 from 'd3';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Header } from '@/components/header';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog';
import {
  ZoomIn,
  ZoomOut,
  Maximize,
  Minimize,
  Maximize2,
  Filter,
  User,
} from 'lucide-react';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';

interface ContextNode {
  id: string;
  cluster_id: number;
  size: number;
  x: number;
  y: number;
  images: string[];
  representative_image: string;
}

interface ContextLink {
  source: string | ContextNode;
  target: string | ContextNode;
  similarity: number;
}

interface ContextGraphData {
  nodes: ContextNode[];
  links: ContextLink[];
  stats: {
    total_clusters: number;
    total_images: number;
    total_connections: number;
    min_similarity: number;
    max_similarity: number;
    avg_similarity: number;
  };
  available_people?: Array<{ id: string; name: string }>;
}

interface ClusterImages {
  images: string[];
}

const PYTHON_API_URL = process.env.NEXT_PUBLIC_PYTHON_API_URL || 'http://localhost:5000';

export default function ContextGraphPage() {
  const [graphData, setGraphData] = useState<ContextGraphData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [clusterImages, setClusterImages] = useState<string[]>([]);
  const [loadingImages, setLoadingImages] = useState(false);
  const [showImageDialog, setShowImageDialog] = useState(false);
  const [minSimilarity, setMinSimilarity] = useState(0.6);
  const [numClusters, setNumClusters] = useState(10);
  const [selectedPeople, setSelectedPeople] = useState<string[]>([]);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const svgRef = useRef<SVGSVGElement>(null);
  const zoomRef = useRef<any>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const nodesRef = useRef<any>(null);
  const linksGroupRef = useRef<any>(null);

  const selectedNode = useMemo(() => {
    if (!selectedNodeId || !graphData) return null;
    return graphData.nodes.find(n => n.id === selectedNodeId) || null;
  }, [selectedNodeId, graphData]);
  
  const togglePersonSelection = useCallback((personId: string) => {
    setSelectedPeople(prev => {
      if (prev.includes(personId)) {
        return prev.filter(id => id !== personId);
      } else {
        return [...prev, personId];
      }
    });
  }, []);

  const handleZoomIn = useCallback(() => {
    if (svgRef.current && zoomRef.current) {
      d3.select(svgRef.current)
        .transition()
        .duration(300)
        .call(zoomRef.current.scaleBy, 1.5);
    }
  }, []);

  const handleZoomOut = useCallback(() => {
    if (svgRef.current && zoomRef.current) {
      d3.select(svgRef.current)
        .transition()
        .duration(300)
        .call(zoomRef.current.scaleBy, 0.67);
    }
  }, []);

  const handleResetZoom = useCallback(() => {
    if (svgRef.current && zoomRef.current) {
      d3.select(svgRef.current)
        .transition()
        .duration(500)
        .call(zoomRef.current.transform, d3.zoomIdentity);
    }
  }, []);

  const toggleFullscreen = useCallback(() => {
    setIsFullscreen(prev => !prev);
  }, []);

  const handleNodeClick = useCallback((clusterId: string, images: string[]) => {
    setSelectedNodeId(clusterId);
    setClusterImages(images);
    setShowImageDialog(true);
  }, []);

  // Fetch graph data
  useEffect(() => {
    setLoading(true);
    const params = new URLSearchParams({
      min_similarity: minSimilarity.toString(),
      num_clusters: numClusters.toString(),
    });
    
    if (selectedPeople.length > 0) {
      params.append('people_filter', selectedPeople.join(','));
    }

    fetch(`${PYTHON_API_URL}/api/images/context-graph?${params}`)
      .then(res => {
        if (!res.ok) throw new Error('Failed to fetch context graph data');
        return res.json();
      })
      .then(data => {
        setGraphData(data);
        setLoading(false);
      })
      .catch(error => {
        console.error('Error loading context graph:', error);
        setError(error.message);
        setLoading(false);
      });
  }, [minSimilarity, numClusters, selectedPeople]);

  // Update selection highlights
  useEffect(() => {
    if (!nodesRef.current || !selectedNodeId) return;

    const nodes = nodesRef.current;
    
    nodes.select('.main-circle')
      .transition()
      .duration(300)
      .attr('fill', (n: any) => n.id === selectedNodeId 
        ? 'url(#selected-gradient)' 
        : 'url(#default-gradient)')
      .attr('r', (n: any) => Math.sqrt(n.size) * 1.5 + (n.id === selectedNodeId ? 7 : 4));
    
    nodes.select('.glow')
      .transition()
      .duration(300)
      .attr('fill', (n: any) => {
        if (n.id === selectedNodeId) return 'hsla(142, 76%, 36%, 0.25)';
        return 'hsla(200, 70%, 50%, 0.12)';
      })
      .attr('r', (n: any) => Math.sqrt(n.size) * 1.5 + (n.id === selectedNodeId ? 12 : 6));

  }, [selectedNodeId]);

  // Main D3 rendering effect
  useEffect(() => {
    if (!svgRef.current || !graphData || graphData.nodes.length === 0) return;

    const width = isFullscreen ? window.innerWidth : 2200;
    const height = isFullscreen ? window.innerHeight : 1400;

    d3.select(svgRef.current).selectAll('*').remove();

    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', `0 0 ${width} ${height}`);

    const defs = svg.append('defs');
    
    // Gradients
    const defaultGradient = defs.append('radialGradient').attr('id', 'default-gradient');
    defaultGradient.append('stop').attr('offset', '0%').attr('stop-color', 'hsl(142, 76%, 46%)');
    defaultGradient.append('stop').attr('offset', '100%').attr('stop-color', 'hsl(142, 76%, 36%)');
    
    const selectedGradient = defs.append('radialGradient').attr('id', 'selected-gradient');
    selectedGradient.append('stop').attr('offset', '0%').attr('stop-color', 'hsl(142, 76%, 66%)');
    selectedGradient.append('stop').attr('offset', '100%').attr('stop-color', 'hsl(142, 76%, 46%)');
    
    const filter = defs.append('filter').attr('id', 'drop-shadow').attr('height', '130%');
    filter.append('feGaussianBlur').attr('in', 'SourceAlpha').attr('stdDeviation', 2);
    filter.append('feOffset').attr('dx', 1).attr('dy', 1).attr('result', 'offsetblur');
    const feMerge = filter.append('feMerge');
    feMerge.append('feMergeNode');
    feMerge.append('feMergeNode').attr('in', 'SourceGraphic');

    const container = svg.append('g');

    // Zoom
    const zoom = d3.zoom()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        container.attr('transform', event.transform);
      });
    svg.call(zoom as any);
    zoomRef.current = zoom;

    // Data preparation
    const nodes = graphData.nodes;
    const nodeMap = new Map(nodes.map(n => [n.id, n]));

    const hydratedLinks = graphData.links.map(link => ({
      ...link,
      source: nodeMap.get(typeof link.source === 'string' ? link.source : (link.source as any).id) as ContextNode,
      target: nodeMap.get(typeof link.target === 'string' ? link.target : (link.target as any).id) as ContextNode
    })).filter(l => l.source && l.target);
    
    const maxSimilarity = Math.max(...hydratedLinks.map(l => l.similarity), 1);

    // Links
    const linkGroup = container.append('g').attr('class', 'links');
    linksGroupRef.current = linkGroup;

    linkGroup.selectAll('line')
      .data(hydratedLinks)
      .join('line')
      .attr('stroke', 'hsl(142, 76%, 46%)')
      .attr('stroke-opacity', d => 0.1 + (d.similarity / maxSimilarity) * 0.3)
      .attr('stroke-width', d => 0.8 + Math.sqrt(d.similarity) * 2)
      .attr('stroke-linecap', 'round')
      .attr('x1', (d: any) => d.source.x)
      .attr('y1', (d: any) => d.source.y)
      .attr('x2', (d: any) => d.target.x)
      .attr('y2', (d: any) => d.target.y);

    // Nodes
    nodes.forEach(node => {
      defs.append('clipPath')
        .attr('id', `clip-${node.id}`)
        .append('circle')
        .attr('r', Math.sqrt(node.size) * 1.5 + 4);
    });
    
    const nodeGroup = container.append('g').attr('class', 'nodes');

    const node = nodeGroup.selectAll('g')
      .data(nodes)
      .join('g')
      .attr('class', 'node')
      .style('cursor', 'pointer')
      .attr('transform', (d: any) => `translate(${d.x},${d.y})`);

    nodesRef.current = node;

    // Glow effect
    node.append('circle')
      .attr('r', d => Math.sqrt(d.size) * 1.5 + 6)
      .attr('fill', d => {
        return d.id === selectedNodeId 
          ? 'hsla(142, 76%, 36%, 0.2)' 
          : 'hsla(200, 70%, 50%, 0.12)';
      })
      .attr('class', 'glow');

    // Main circle
    node.append('circle')
      .attr('r', d => Math.sqrt(d.size) * 1.5 + 4)
      .attr('fill', d => d.id === selectedNodeId 
        ? 'url(#selected-gradient)' 
        : 'url(#default-gradient)')
      .attr('stroke', 'hsl(0, 0%, 100%)')
      .attr('stroke-width', 1.5)
      .attr('filter', 'url(#drop-shadow)')
      .attr('class', 'main-circle');
    
    // Add representative image inside the circle
    node.filter(d => !!d.representative_image)
      .append('image')
      .attr('href', d => {
        const filename = d.representative_image.replace('images/', '');
        return `/api/images/${filename}`;
      })
      .attr('x', d => -(Math.sqrt(d.size) * 1.5 + 4))
      .attr('y', d => -(Math.sqrt(d.size) * 1.5 + 4))
      .attr('width', d => (Math.sqrt(d.size) * 1.5 + 4) * 2)
      .attr('height', d => (Math.sqrt(d.size) * 1.5 + 4) * 2)
      .attr('clip-path', d => `url(#clip-${d.id})`)
      .style('opacity', 0.7)
      .style('pointer-events', 'none');
    
    // Image count badge
    node.append('circle')
      .attr('r', d => {
        const numDigits = d.size.toString().length;
        return Math.min(4 + (numDigits - 1) * 1, 6);
      })
      .attr('cx', d => Math.sqrt(d.size) * 1.5 + 2)
      .attr('cy', d => -(Math.sqrt(d.size) * 1.5 + 2))
      .attr('fill', 'hsl(0, 0%, 40%)')
      .attr('fill-opacity', 0.8)
      .attr('stroke', 'hsl(0, 0%, 60%)')
      .attr('stroke-width', 1)
      .attr('stroke-opacity', 0.7);

    node.append('text')
      .attr('x', d => Math.sqrt(d.size) * 1.5 + 2)
      .attr('y', d => -(Math.sqrt(d.size) * 1.5 + 2) + 1)
      .attr('dominant-baseline', 'middle')
      .attr('text-anchor', 'middle')
      .attr('font-size', d => {
        const numDigits = d.size.toString().length;
        return Math.max(6 - numDigits, 4);
      })
      .attr('font-weight', 'bold')
      .attr('fill', 'hsl(0, 0%, 100%)')
      .text(d => d.size);

    // Tooltip with cluster info
    node.append('title')
      .text(d => `Cluster ${d.cluster_id}\n${d.size} images`);

    // Event handlers
    node.on('mouseenter', function(event, d) {
      const connectedIds = new Set<string>();
      hydratedLinks.forEach(link => {
        const sourceId = (link.source as ContextNode).id;
        const targetId = (link.target as ContextNode).id;
        if (sourceId === d.id) connectedIds.add(targetId);
        if (targetId === d.id) connectedIds.add(sourceId);
      });

      node.style('opacity', (n: ContextNode) => 
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
            return (2 + Math.sqrt(l.similarity) * 2) + 'px';
          }
          return null;
        });

      d3.select(this).select('.main-circle')
        .transition()
        .duration(200)
        .attr('r', Math.sqrt(d.size) * 1.5 + 7);
      
      d3.select(this).select('.glow')
        .transition()
        .duration(200)
        .attr('r', Math.sqrt(d.size) * 1.5 + 12)
        .attr('fill-opacity', 0.25);
    })
    .on('mouseleave', function(event, d) {
      node.style('opacity', 1);
      linkGroup.selectAll('line')
        .style('opacity', null)
        .style('stroke-width', null);

      if (d.id !== selectedNodeId) {
        d3.select(this).select('.main-circle')
          .transition()
          .duration(200)
          .attr('r', Math.sqrt(d.size) * 1.5 + 4);
        
        d3.select(this).select('.glow')
          .transition()
          .duration(200)
          .attr('r', Math.sqrt(d.size) * 1.5 + 6)
          .attr('fill-opacity', 0.12);
      }
    });

    node.on('click', (event, d) => {
      event.stopPropagation();
      handleNodeClick(d.id, d.images);
    });

  }, [graphData, isFullscreen, selectedNodeId, handleNodeClick]);

  if (loading) {
    return (
      <>
        <Header />
        <div className="flex items-center justify-center h-screen">
          <div className="text-lg">Chargement du graphe contextuel...</div>
        </div>
      </>
    );
  }

  if (error) {
    return (
      <>
        <Header />
        <div className="flex items-center justify-center h-screen">
          <Card className="p-6 max-w-md">
            <h2 className="text-xl font-bold text-destructive mb-2">Erreur</h2>
            <p className="mb-2">{error}</p>
            <p className="text-sm text-muted-foreground">
              Assurez-vous que l'API Python tourne sur {PYTHON_API_URL}
            </p>
          </Card>
        </div>
      </>
    );
  }

  if (!graphData || graphData.nodes.length === 0) {
    return (
      <>
        <Header />
        <div className="flex items-center justify-center h-screen">
          <Card className="p-6 max-w-md">
            <h2 className="text-xl font-bold mb-2">Aucune donnée</h2>
            <p>Aucun cluster contextuel trouvé dans votre collection de photos.</p>
          </Card>
        </div>
      </>
    );
  }

  return (
    <>
      {!isFullscreen && <Header />}
      <div className={`${isFullscreen ? 'fixed inset-0 z-50 bg-background' : 'container mx-auto p-6 max-w-[2400px]'}`}>
        {!isFullscreen && (
        <>
          <div className="mb-6">
            <h1 className="text-3xl font-bold mb-2">Graphe Contextuel</h1>
            <p className="text-muted-foreground">
              Visualisation des images regroupées par contexte sémantique similaire.
            </p>
            
            <div className="flex gap-4 mt-4 flex-wrap">
              <Card className="p-4 bg-card">
                <div className="text-2xl font-bold text-primary">{graphData.stats.total_clusters}</div>
                <div className="text-sm text-muted-foreground">Clusters</div>
              </Card>
              <Card className="p-4 bg-card">
                <div className="text-2xl font-bold text-primary">{graphData.stats.total_images}</div>
                <div className="text-sm text-muted-foreground">Images</div>
              </Card>
              <Card className="p-4 bg-card">
                <div className="text-2xl font-bold text-primary">{graphData.stats.total_connections}</div>
                <div className="text-sm text-muted-foreground">Connexions</div>
              </Card>
              <Card className="p-4 bg-card">
                <div className="text-2xl font-bold text-primary">
                  {graphData.stats.total_clusters > 0 
                    ? (graphData.stats.total_images / graphData.stats.total_clusters).toFixed(1) 
                    : '0'}
                </div>
                <div className="text-sm text-muted-foreground">Taille moy. cluster</div>
              </Card>
            </div>

            {/* Filters */}
            <div className="grid grid-cols-1 gap-4 mt-4">
              {/* People filter (multi-select) */}
              {graphData.available_people && graphData.available_people.length > 0 && (
                <Card className="p-4 bg-card">
                  <div className="space-y-3">
                    <div className="flex items-center gap-2">
                      <User className="h-5 w-5 text-muted-foreground" />
                      <label className="text-sm font-medium">
                        Filtrer par personnes présentes
                      </label>
                      {selectedPeople.length > 0 && (
                        <span className="text-xs text-muted-foreground">
                          ({selectedPeople.length} sélectionnée{selectedPeople.length > 1 ? 's' : ''})
                        </span>
                      )}
                    </div>
                    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2 max-h-48 overflow-y-auto">
                      {graphData.available_people.map(person => (
                        <label
                          key={person.id}
                          className="flex items-center gap-2 p-2 rounded-md hover:bg-accent cursor-pointer transition-colors"
                        >
                          <input
                            type="checkbox"
                            checked={selectedPeople.includes(person.id)}
                            onChange={() => togglePersonSelection(person.id)}
                            className="rounded border-gray-300"
                          />
                          <span className="text-sm">{person.name}</span>
                        </label>
                      ))}
                    </div>
                  </div>
                </Card>
              )}

              {/* Cluster parameters */}
              <Card className="p-4 bg-card">
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between mb-2">
                      <label className="text-sm font-medium">
                        Nombre de clusters
                      </label>
                      <span className="text-sm text-muted-foreground">
                        {numClusters}
                      </span>
                    </div>
                    <Slider
                      value={[numClusters]}
                      onValueChange={(values) => setNumClusters(values[0])}
                      min={5}
                      max={30}
                      step={1}
                      className="w-full"
                    />
                  </div>
                  
                  <div>
                    <div className="flex justify-between mb-2">
                      <label className="text-sm font-medium">
                        Similarité minimale
                      </label>
                      <span className="text-sm text-muted-foreground">
                        {(minSimilarity * 100).toFixed(0)}%
                      </span>
                    </div>
                    <Slider
                      value={[minSimilarity * 100]}
                      onValueChange={(values) => setMinSimilarity(values[0] / 100)}
                      min={0}
                      max={100}
                      step={5}
                      className="w-full"
                    />
                  </div>
                </div>
              </Card>
            </div>
          </div>
        </>
      )}

      <Card 
        ref={containerRef}
        className={`p-4 relative bg-card ${isFullscreen ? 'h-screen border-0 rounded-none' : ''}`}
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
        
        <div className={`w-full overflow-auto rounded-lg border bg-gradient-to-br from-emerald-50 to-teal-100 dark:from-emerald-950 dark:to-teal-900 ${isFullscreen ? 'h-full' : ''}`}>
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
                <div className="w-3 h-3 rounded-full bg-emerald-600"></div>
                <span>Cluster contextuel</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-emerald-400"></div>
                <span>Connexion de similarité</span>
              </div>
            </div>
          </div>
        )}
      </Card>

      </div>
      
      {/* Image Dialog */}
      <Dialog open={showImageDialog} onOpenChange={setShowImageDialog}>
        <DialogContent className="max-w-6xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="text-2xl">
              Cluster {selectedNode?.cluster_id}
            </DialogTitle>
            <DialogDescription>
              {clusterImages.length} image{clusterImages.length > 1 ? 's' : ''}
            </DialogDescription>
          </DialogHeader>
          
          {loadingImages ? (
            <div className="flex items-center justify-center py-12">
              <div className="text-lg">Chargement des images...</div>
            </div>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 mt-4">
              {clusterImages.map((filename, index) => (
                <div key={index} className="space-y-2">
                  <div className="relative aspect-square overflow-hidden rounded-lg border-2 border-border hover:border-primary transition-colors group">
                    <img
                      src={`/api/images/${filename.replace('images/', '')}`}
                      alt={`Image ${index + 1}`}
                      className="w-full h-full object-cover"
                      loading="lazy"
                    />
                  </div>
                </div>
              ))}
            </div>
          )}
          
          {!loadingImages && clusterImages.length === 0 && (
            <div className="text-center py-12 text-muted-foreground">
              Aucune image trouvée pour ce cluster
            </div>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}
