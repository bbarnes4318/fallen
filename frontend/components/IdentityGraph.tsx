'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import dynamic from 'next/dynamic';

const ForceGraph2D = dynamic(() => import('react-force-graph-2d'), { ssr: false });

// ─── Types ───────────────────────────────────────────────
interface GraphNode {
  id: string;
  name: string;
  group: number;
  thumbnail?: string;
  x?: number;
  y?: number;
}

interface GraphLink {
  source: string | GraphNode;
  target: string | GraphNode;
  value: number;
}

interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}

// ─── Dummy Constellation Data ────────────────────────────
const DUMMY_GRAPH: GraphData = {
  nodes: [
    { id: "angel_01", name: "VOSS_K", group: 2 },
    { id: "angel_02", name: "REYES_M", group: 1 },
    { id: "angel_03", name: "IVANOV_D", group: 2 },
    { id: "angel_04", name: "NAKAMURA_S", group: 1 },
    { id: "angel_05", name: "PETROV_A", group: 1 },
    { id: "angel_06", name: "DESCHAMPS_L", group: 2 },
    { id: "angel_07", name: "OKAFOR_C", group: 1 },
    { id: "angel_08", name: "MUELLER_H", group: 1 },
    { id: "angel_09", name: "SANTOS_R", group: 1 },
    { id: "angel_10", name: "CHEN_W", group: 2 },
    { id: "angel_11", name: "KOWALSKI_J", group: 1 },
    { id: "angel_12", name: "ABADI_F", group: 1 },
    { id: "angel_13", name: "NOVAK_T", group: 1 },
    { id: "angel_14", name: "GRAVES_E", group: 2 },
    { id: "angel_15", name: "SHAH_P", group: 1 },
  ],
  links: [
    { source: "angel_01", target: "angel_03", value: 97.4 },
    { source: "angel_01", target: "angel_06", value: 88.2 },
    { source: "angel_02", target: "angel_05", value: 82.1 },
    { source: "angel_03", target: "angel_10", value: 96.8 },
    { source: "angel_04", target: "angel_08", value: 79.5 },
    { source: "angel_05", target: "angel_09", value: 85.3 },
    { source: "angel_06", target: "angel_14", value: 91.7 },
    { source: "angel_07", target: "angel_11", value: 78.9 },
    { source: "angel_08", target: "angel_13", value: 83.6 },
    { source: "angel_09", target: "angel_12", value: 76.2 },
    { source: "angel_10", target: "angel_14", value: 94.1 },
    { source: "angel_11", target: "angel_15", value: 80.4 },
    { source: "angel_12", target: "angel_02", value: 77.8 },
    { source: "angel_13", target: "angel_04", value: 86.9 },
    { source: "angel_14", target: "angel_01", value: 98.2 },
    { source: "angel_15", target: "angel_07", value: 81.3 },
    { source: "angel_01", target: "angel_10", value: 92.5 },
    { source: "angel_06", target: "angel_03", value: 89.7 },
    { source: "angel_05", target: "angel_13", value: 84.1 },
    { source: "angel_02", target: "angel_08", value: 76.9 },
  ],
};

// ─── Helpers ─────────────────────────────────────────────
function getApiUrl(): string {
  if (typeof window !== 'undefined' && window.location.hostname.includes('facial-frontend')) {
    return window.location.origin.replace('facial-frontend', 'facial-backend');
  }
  return process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
}

function getNodeConnections(nodeId: string, links: GraphLink[]) {
  return links.filter((l) => {
    const src = typeof l.source === 'object' ? l.source.id : l.source;
    const tgt = typeof l.target === 'object' ? l.target.id : l.target;
    return src === nodeId || tgt === nodeId;
  }).map((l) => {
    const src = typeof l.source === 'object' ? l.source.id : l.source;
    const tgt = typeof l.target === 'object' ? l.target.id : l.target;
    return {
      entity: src === nodeId ? tgt : src,
      score: l.value,
    };
  });
}

// ─── Props ───────────────────────────────────────────────
interface IdentityGraphProps {
  onCompare?: (galleryUrl: string, probeUrl: string, galleryName: string, probeName: string) => void;
}

// ─── Component ───────────────────────────────────────────
export default function IdentityGraph({ onCompare }: IdentityGraphProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], links: [] });
  const [loading, setLoading] = useState(true);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [showLegend, setShowLegend] = useState(true);

  // ── Target selection for 1:1 comparison ──
  const [targetA, setTargetA] = useState<GraphNode | null>(null);  // Gallery
  const [targetB, setTargetB] = useState<GraphNode | null>(null);  // Probe

  // Ref for the force graph instance (must be before any conditional returns)
  const fgRef = useRef<any>(null);

  // ── Image cache for thumbnail rendering on canvas ──
  const imageCache = useRef<Map<string, HTMLImageElement | null>>(new Map());

  const loadImage = useCallback((url: string): HTMLImageElement | null => {
    if (!url) return null;
    if (imageCache.current.has(url)) return imageCache.current.get(url) || null;

    // Mark as loading (null = pending)
    imageCache.current.set(url, null);
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      imageCache.current.set(url, img);
    };
    img.onerror = () => {
      imageCache.current.set(url, null);
    };
    img.src = url;
    return null;
  }, []);

  // ── Responsive sizing ──
  useEffect(() => {
    const updateSize = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight,
        });
      }
    };
    updateSize();
    window.addEventListener('resize', updateSize);
    return () => window.removeEventListener('resize', updateSize);
  }, []);

  // ── Fetch live graph data OR fall back to dummy ──
  useEffect(() => {
    const token = localStorage.getItem('operator_token');
    if (!token) {
      queueMicrotask(() => {
        setGraphData(DUMMY_GRAPH);
        setLoading(false);
      });
      return;
    }

    const fetchWithRetry = async (attempt = 0): Promise<void> => {
      const MAX_RETRIES = 3;
      try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 30000); // 30s for signed URL
        const res = await fetch(`${getApiUrl()}/vault/network`, {
          headers: { 'Authorization': `Bearer ${token}` },
          signal: controller.signal,
        });
        clearTimeout(timeout);
        if (res.ok) {
          const meta = await res.json();
          if (meta.graph_url) {
            // Two-step: fetch actual graph JSON directly from GCS
            const gcsRes = await fetch(meta.graph_url);
            if (gcsRes.ok) {
              const data = await gcsRes.json();
              if (data.nodes && data.nodes.length > 0) {
                setGraphData(data);
              } else {
                setGraphData(DUMMY_GRAPH);
              }
            } else {
              setGraphData(DUMMY_GRAPH);
            }
          } else if (meta.nodes && meta.nodes.length > 0) {
            // Legacy: direct inline data
            setGraphData(meta);
          } else {
            setGraphData(DUMMY_GRAPH);
          }
          setLoading(false);
        } else if (attempt < MAX_RETRIES) {
          await new Promise(r => setTimeout(r, 2000 * (attempt + 1)));
          return fetchWithRetry(attempt + 1);
        } else {
          setGraphData(DUMMY_GRAPH);
          setLoading(false);
        }
      } catch {
        if (attempt < MAX_RETRIES) {
          await new Promise(r => setTimeout(r, 3000 * (attempt + 1)));
          return fetchWithRetry(attempt + 1);
        }
        setGraphData(DUMMY_GRAPH);
        setLoading(false);
      }
    };
    fetchWithRetry();
  }, []);

  // Pre-load all node thumbnails when graph data arrives
  useEffect(() => {
    graphData.nodes.forEach((node) => {
      if (node.thumbnail) loadImage(node.thumbnail);
    });
  }, [graphData, loadImage]);


  const connections = selectedNode ? getNodeConnections(selectedNode.id, graphData.links) : [];

  // Build thumbnail lookup for dossier panel
  const nodeMap = useRef<Map<string, GraphNode>>(new Map());
  useEffect(() => {
    const m = new Map<string, GraphNode>();
    graphData.nodes.forEach((n) => m.set(n.id, n));
    nodeMap.current = m;
  }, [graphData]);

  // ── Loading State ──
  if (loading) {
    return (
      <div className="h-full w-full flex items-center justify-center bg-[#050505]">
        <div className="text-center">
          <div className="w-10 h-10 border-2 border-[#333] border-t-[#D4AF37] rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-[#D4AF37] text-xs tracking-[0.3em] animate-pulse font-mono">
            DECRYPTING VAULT TOPOLOGY...
          </p>
        </div>
      </div>
    );
  }



  return (
    <div ref={containerRef} className="h-full w-full relative bg-[#050505] overflow-hidden">
      {/* ── Force Graph Canvas ── */}
      <ForceGraph2D
        ref={fgRef}
        graphData={graphData}
        width={dimensions.width}
        height={dimensions.height}
        backgroundColor="#050505"
        nodeCanvasObject={(node: Record<string, unknown>, ctx: CanvasRenderingContext2D) => {
          const x = (node.x as number) ?? 0;
          const y = (node.y as number) ?? 0;
          const isGold = node.group === 2;
          const radius = 12;
          const thumbUrl = node.thumbnail as string;

          // Try to get cached image
          const img = thumbUrl ? loadImage(thumbUrl) : null;

          if (img) {
            // ── Render circular face thumbnail ──
            ctx.save();
            ctx.beginPath();
            ctx.arc(x, y, radius, 0, 2 * Math.PI);
            ctx.closePath();
            ctx.clip();
            ctx.drawImage(img, x - radius, y - radius, radius * 2, radius * 2);
            ctx.restore();

            // Ring border
            ctx.beginPath();
            ctx.arc(x, y, radius + 1, 0, 2 * Math.PI);
            ctx.strokeStyle = isGold ? '#D4AF37' : '#555555';
            ctx.lineWidth = isGold ? 2.5 : 1.5;
            if (isGold) {
              ctx.shadowColor = '#D4AF37';
              ctx.shadowBlur = 10;
            }
            ctx.stroke();
            ctx.shadowColor = 'transparent';
            ctx.shadowBlur = 0;
          } else {
            // ── Fallback: colored dot (when no thumbnail) ──
            const dotRadius = isGold ? 7 : 5;
            if (isGold) { ctx.shadowColor = '#D4AF37'; ctx.shadowBlur = 12; }
            ctx.beginPath();
            ctx.arc(x, y, dotRadius, 0, 2 * Math.PI);
            ctx.fillStyle = isGold ? '#D4AF37' : '#666666';
            ctx.fill();
            ctx.strokeStyle = isGold ? '#D4AF37' : '#444444';
            ctx.lineWidth = 1;
            ctx.stroke();
            ctx.shadowColor = 'transparent';
            ctx.shadowBlur = 0;
          }

          // Name label
          ctx.font = '3.5px Courier New';
          ctx.fillStyle = isGold ? '#D4AF37' : '#999999';
          ctx.textAlign = 'left';
          ctx.textBaseline = 'middle';
          ctx.fillText((node.name as string) ?? (node.id as string), x + radius + 4, y);
        }}
        nodePointerAreaPaint={(node: Record<string, unknown>, color: string, ctx: CanvasRenderingContext2D) => {
          const x = (node.x as number) ?? 0;
          const y = (node.y as number) ?? 0;
          ctx.beginPath();
          ctx.arc(x, y, 14, 0, 2 * Math.PI);
          ctx.fillStyle = color;
          ctx.fill();
        }}
        linkColor={(link: Record<string, unknown>) => ((link.value as number) > 95 ? '#881111' : '#2a2a2a')}
        linkWidth={(link: Record<string, unknown>) => ((link.value as number) > 95 ? 1.8 : 0.4)}
        linkDirectionalParticles={1}
        linkDirectionalParticleWidth={(link: Record<string, unknown>) => (link.value as number) > 95 ? 2 : 0}
        linkDirectionalParticleColor={() => '#881111'}
        onNodeClick={(node: Record<string, unknown>) => setSelectedNode(node as unknown as GraphNode)}
        cooldownTicks={150}
        d3AlphaDecay={0.015}
        d3VelocityDecay={0.25}
        onEngineStop={() => {
          if (fgRef.current) {
            fgRef.current.zoomToFit(400, 60);
          }
        }}
      />

      {/* ── Scanline Overlay ── */}
      <div className="absolute inset-0 pointer-events-none"
        style={{
          background: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(212,175,55,0.015) 2px, rgba(212,175,55,0.015) 4px)',
        }}
      />

      {/* ── HUD Corners ── */}
      <div className="absolute top-3 left-3 w-5 h-5 border-t border-l border-[#D4AF37]/30 pointer-events-none" />
      <div className="absolute top-3 right-3 w-5 h-5 border-t border-r border-[#D4AF37]/30 pointer-events-none" />
      <div className="absolute bottom-3 left-3 w-5 h-5 border-b border-l border-[#D4AF37]/30 pointer-events-none" />
      <div className="absolute bottom-3 right-3 w-5 h-5 border-b border-r border-[#D4AF37]/30 pointer-events-none" />

      {/* ── Contextual Legend Panel ── */}
      {showLegend && (
        <div className="absolute top-5 left-5 z-10 pointer-events-auto">
          <div className="bg-[#0a0a0b]/95 border border-[#1f1f1f] rounded-lg p-4 max-w-[280px] backdrop-blur-sm">
            <div className="flex justify-between items-start mb-3">
              <p className="text-[9px] text-[#D4AF37] tracking-[0.25em] font-mono font-bold">
                BIOMETRIC SIMILARITY NETWORK
              </p>
              <button
                onClick={() => setShowLegend(false)}
                className="text-gray-600 hover:text-gray-400 text-xs ml-2 transition-colors"
              >
                ✕
              </button>
            </div>
            <p className="text-[10px] text-gray-500 leading-relaxed mb-3 font-mono">
              Each node represents an identity in the encrypted vault. Links indicate
              &gt;90% ArcFace cosine similarity between 512-D biometric embeddings.
            </p>
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full border-2 border-[#D4AF37] bg-[#D4AF37]/20 shrink-0" />
                <span className="text-[9px] text-gray-400 font-mono">
                  ANOMALY — High connectivity (potential duplicate/alias)
                </span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full border border-[#555] bg-[#333]/30 shrink-0" />
                <span className="text-[9px] text-gray-400 font-mono">
                  STANDARD — Normal connectivity pattern
                </span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-6 h-0.5 bg-[#881111] shrink-0" />
                <span className="text-[9px] text-gray-400 font-mono">
                  CRITICAL — &gt;95% match (near-identical)
                </span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-6 h-px bg-[#2a2a2a] shrink-0" />
                <span className="text-[9px] text-gray-400 font-mono">
                  STRONG — 90–95% match
                </span>
              </div>
            </div>
            <div className="mt-3 pt-2 border-t border-[#1a1a1a]">
              <p className="text-[8px] text-gray-600 font-mono">
                Click any node to inspect identity dossier
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Show legend button if hidden */}
      {!showLegend && (
        <button
          onClick={() => setShowLegend(true)}
          className="absolute top-5 left-5 z-10 text-[9px] text-gray-600 hover:text-[#D4AF37] font-mono tracking-widest border border-[#1f1f1f] hover:border-[#D4AF37]/30 bg-[#0a0a0b]/80 px-3 py-1.5 rounded transition-colors"
        >
          ◈ LEGEND
        </button>
      )}

      {/* ── Status Bar ── */}
      <div className="absolute bottom-3 left-1/2 -translate-x-1/2 text-[9px] text-gray-600 font-mono tracking-widest pointer-events-none">
        NODES: {graphData.nodes.length} &nbsp;│&nbsp; EDGES: {graphData.links.length} &nbsp;│&nbsp; SOVEREIGN IDENTITY GRAPH
      </div>

      {/* ── Entity Dossier Side Panel ── */}
      <div
        className={`absolute top-0 right-0 h-full w-80 bg-[#0a0a0b] border-l-2 border-[#D4AF37]/40 transition-transform duration-300 ease-out z-20 flex flex-col font-mono ${
          selectedNode ? 'translate-x-0 pointer-events-auto' : 'translate-x-full pointer-events-none'
        }`}
      >
        {selectedNode && (
          <>
            {/* Header with face thumbnail */}
            <div className="shrink-0 p-4 border-b border-[#1a1a1a]">
              <div className="flex justify-between items-start">
                <div className="flex items-center gap-3">
                  {/* Large face thumbnail */}
                  {selectedNode.thumbnail ? (
                    <div className="w-16 h-16 rounded-lg overflow-hidden border-2 border-[#D4AF37]/50 shrink-0">
                      <img
                        src={selectedNode.thumbnail}
                        alt={selectedNode.name}
                        className="w-full h-full object-cover"
                        crossOrigin="anonymous"
                      />
                    </div>
                  ) : (
                    <div className="w-16 h-16 rounded-lg border border-[#333] bg-[#111] flex items-center justify-center shrink-0">
                      <span className="text-gray-600 text-lg">◆</span>
                    </div>
                  )}
                  <div>
                    <p className="text-[9px] text-gray-600 tracking-widest mb-1">ENTITY DOSSIER</p>
                    <h2 className="text-[#D4AF37] text-sm font-bold tracking-wider leading-tight">
                      {selectedNode.name}
                    </h2>
                    <p className="text-[8px] text-gray-600 mt-0.5 break-all">{selectedNode.id}</p>
                  </div>
                </div>
                <button
                  onClick={() => setSelectedNode(null)}
                  className="text-gray-600 hover:text-white text-xs border border-[#333] hover:border-gray-500 px-2 py-0.5 rounded transition-colors"
                >
                  ✕
                </button>
              </div>
              <div className="mt-3 flex gap-3">
                <div className="border border-[#1f1f1f] bg-[#0d0d0e] rounded px-3 py-1.5">
                  <p className="text-[8px] text-gray-600 tracking-wider">GROUP</p>
                  <p className={`text-sm font-bold ${selectedNode.group === 2 ? 'text-[#D4AF37]' : 'text-gray-300'}`}>
                    {selectedNode.group === 2 ? 'ANOMALY' : 'STANDARD'}
                  </p>
                </div>
                <div className="border border-[#1f1f1f] bg-[#0d0d0e] rounded px-3 py-1.5">
                  <p className="text-[8px] text-gray-600 tracking-wider">CONNECTIONS</p>
                  <p className="text-sm font-bold text-white">{connections.length}</p>
                </div>
              </div>

              {/* ── Set as Target Buttons ── */}
              {selectedNode.thumbnail && (
                <div className="mt-3 flex gap-2">
                  <button
                    onClick={() => { setTargetA(selectedNode); setSelectedNode(null); }}
                    className={`flex-1 py-2 rounded text-[9px] tracking-[0.15em] font-bold transition-all border ${
                      targetA?.id === selectedNode.id
                        ? 'border-[#D4AF37] bg-[#D4AF37]/20 text-[#D4AF37]'
                        : 'border-[#333] bg-[#111] text-gray-400 hover:border-[#D4AF37]/60 hover:text-[#D4AF37] hover:bg-[#D4AF37]/10'
                    }`}
                  >
                    ◆ SET AS TARGET A
                    <span className="block text-[7px] tracking-wider text-gray-600 mt-0.5">GALLERY</span>
                  </button>
                  <button
                    onClick={() => { setTargetB(selectedNode); setSelectedNode(null); }}
                    className={`flex-1 py-2 rounded text-[9px] tracking-[0.15em] font-bold transition-all border ${
                      targetB?.id === selectedNode.id
                        ? 'border-cyan-500 bg-cyan-500/20 text-cyan-400'
                        : 'border-[#333] bg-[#111] text-gray-400 hover:border-cyan-500/60 hover:text-cyan-400 hover:bg-cyan-500/10'
                    }`}
                  >
                    ◆ SET AS TARGET B
                    <span className="block text-[7px] tracking-wider text-gray-600 mt-0.5">PROBE</span>
                  </button>
                </div>
              )}
            </div>

            {/* Connections List with thumbnails */}
            <div className="flex-1 overflow-y-auto p-4">
              <p className="text-[9px] text-gray-600 tracking-widest mb-3">VERIFIED BIOMETRIC LINKS</p>
              <div className="space-y-2">
                {connections
                  .sort((a, b) => b.score - a.score)
                  .map((conn) => {
                    const connNode = nodeMap.current.get(conn.entity);
                    return (
                      <div
                        key={conn.entity}
                        className={`flex items-center gap-2.5 p-2 rounded border cursor-pointer hover:opacity-80 transition-opacity ${
                          conn.score > 95
                            ? 'border-red-900/60 bg-red-950/20'
                            : conn.score > 85
                            ? 'border-[#D4AF37]/30 bg-[#D4AF37]/5'
                            : 'border-[#1f1f1f] bg-[#0d0d0e]'
                        }`}
                        onClick={() => {
                          if (connNode) setSelectedNode(connNode);
                        }}
                      >
                        {/* Connected entity thumbnail */}
                        {connNode?.thumbnail ? (
                          <div className="w-8 h-8 rounded overflow-hidden border border-[#333] shrink-0">
                            <img
                              src={connNode.thumbnail}
                              alt={connNode.name}
                              className="w-full h-full object-cover"
                              crossOrigin="anonymous"
                            />
                          </div>
                        ) : (
                          <div className="w-8 h-8 rounded border border-[#222] bg-[#111] flex items-center justify-center shrink-0">
                            <span className="text-gray-700 text-[8px]">◆</span>
                          </div>
                        )}
                        <div className="flex-1 min-w-0">
                          <p className="text-[10px] text-gray-300 tracking-wider truncate">
                            {connNode?.name || conn.entity}
                          </p>
                        </div>
                        <span className={`text-xs font-bold tracking-wider shrink-0 ${
                          conn.score > 95
                            ? 'text-red-400'
                            : conn.score > 85
                            ? 'text-[#D4AF37]'
                            : 'text-gray-400'
                        }`}>
                          {conn.score.toFixed(1)}%
                        </span>
                      </div>
                    );
                  })}
              </div>
            </div>

            {/* Footer */}
            <div className="shrink-0 p-3 border-t border-[#1a1a1a]">
              <div className="text-[8px] text-gray-700 text-center tracking-widest">
                ENCRYPTED VAULT RECORD
              </div>
            </div>
          </>
        )}
      </div>

      {/* ── Floating Comparison Launcher ── */}
      {(targetA || targetB) && (
        <div className="absolute bottom-5 left-1/2 -translate-x-1/2 z-30 pointer-events-auto">
          <div className="bg-[#0a0a0b]/95 border border-[#D4AF37]/40 rounded-xl px-5 py-3 backdrop-blur-md shadow-[0_0_40px_rgba(212,175,55,0.15)] flex items-center gap-4 font-mono">

            {/* Target A Slot */}
            <div className="flex items-center gap-2">
              {targetA ? (
                <div className="relative group">
                  <div className="w-10 h-10 rounded-lg overflow-hidden border-2 border-[#D4AF37]">
                    <img src={targetA.thumbnail} alt={targetA.name} className="w-full h-full object-cover" crossOrigin="anonymous" />
                  </div>
                  <button
                    onClick={() => setTargetA(null)}
                    className="absolute -top-1.5 -right-1.5 w-4 h-4 bg-red-900 border border-red-700 rounded-full text-[8px] text-white flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                  >✕</button>
                  <p className="text-[7px] text-[#D4AF37] tracking-wider text-center mt-1 truncate max-w-[60px]">{targetA.name}</p>
                </div>
              ) : (
                <div className="w-10 h-10 rounded-lg border-2 border-dashed border-[#333] flex items-center justify-center">
                  <span className="text-gray-700 text-[8px]">A</span>
                </div>
              )}
            </div>

            {/* VS Divider */}
            <div className="text-[10px] text-gray-600 tracking-[0.3em] font-bold">VS</div>

            {/* Target B Slot */}
            <div className="flex items-center gap-2">
              {targetB ? (
                <div className="relative group">
                  <div className="w-10 h-10 rounded-lg overflow-hidden border-2 border-cyan-500">
                    <img src={targetB.thumbnail} alt={targetB.name} className="w-full h-full object-cover" crossOrigin="anonymous" />
                  </div>
                  <button
                    onClick={() => setTargetB(null)}
                    className="absolute -top-1.5 -right-1.5 w-4 h-4 bg-red-900 border border-red-700 rounded-full text-[8px] text-white flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                  >✕</button>
                  <p className="text-[7px] text-cyan-400 tracking-wider text-center mt-1 truncate max-w-[60px]">{targetB.name}</p>
                </div>
              ) : (
                <div className="w-10 h-10 rounded-lg border-2 border-dashed border-[#333] flex items-center justify-center">
                  <span className="text-gray-700 text-[8px]">B</span>
                </div>
              )}
            </div>

            {/* Launch Button */}
            <button
              disabled={!targetA || !targetB || !targetA.thumbnail || !targetB.thumbnail}
              onClick={() => {
                if (targetA?.thumbnail && targetB?.thumbnail && onCompare) {
                  onCompare(targetA.thumbnail, targetB.thumbnail, targetA.name, targetB.name);
                }
              }}
              className={`ml-2 px-5 py-2.5 rounded-lg text-[10px] tracking-[0.2em] font-bold transition-all ${
                targetA && targetB && targetA.thumbnail && targetB.thumbnail
                  ? 'bg-[#D4AF37] text-black hover:bg-[#e5c544] shadow-[0_0_20px_rgba(212,175,55,0.3)] cursor-pointer'
                  : 'bg-[#1a1a1a] text-gray-600 border border-[#222] cursor-not-allowed'
              }`}
            >
              RUN DEEP FORENSIC COMPARISON
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
