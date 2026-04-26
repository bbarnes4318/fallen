'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import dynamic from 'next/dynamic';

const ForceGraph2D = dynamic(() => import('react-force-graph-2d'), { ssr: false });

// ─── Types ───────────────────────────────────────────────
interface GraphNode {
  id: string;
  name: string;
  group: number;
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

// ─── Component ───────────────────────────────────────────
export default function IdentityGraph() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], links: [] });
  const [loading, setLoading] = useState(true);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);

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
      setGraphData(DUMMY_GRAPH);
      setLoading(false);
      return;
    }

    const fetchGraph = async () => {
      try {
        const res = await fetch(`${getApiUrl()}/vault/network`, {
          headers: { 'Authorization': `Bearer ${token}` },
        });
        if (res.ok) {
          const data = await res.json();
          if (data.nodes && data.nodes.length > 0) {
            setGraphData(data);
          } else {
            setGraphData(DUMMY_GRAPH);
          }
        } else {
          setGraphData(DUMMY_GRAPH);
        }
      } catch {
        setGraphData(DUMMY_GRAPH);
      } finally {
        setLoading(false);
      }
    };
    fetchGraph();
  }, []);

  // ── Custom Node Renderer ──
  const nodeCanvasObject = useCallback((node: GraphNode, ctx: CanvasRenderingContext2D) => {
    const x = node.x ?? 0;
    const y = node.y ?? 0;
    const isGold = node.group === 2;
    const radius = isGold ? 6 : 4;

    // Glow for high-profile anomalies
    if (isGold) {
      ctx.shadowColor = '#D4AF37';
      ctx.shadowBlur = 12;
    }

    // Node circle
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    ctx.fillStyle = isGold ? '#D4AF37' : '#666666';
    ctx.fill();
    ctx.strokeStyle = isGold ? '#D4AF37' : '#444444';
    ctx.lineWidth = 1;
    ctx.stroke();

    // Reset shadow
    ctx.shadowColor = 'transparent';
    ctx.shadowBlur = 0;

    // Label
    ctx.font = '3px Courier New';
    ctx.fillStyle = isGold ? '#D4AF37' : '#888888';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(node.name, x + radius + 3, y);
  }, []);

  // ── Custom Link Renderer ──
  const linkColor = useCallback((link: GraphLink) => {
    return link.value > 95 ? '#660000' : '#333333';
  }, []);

  const linkWidth = useCallback((link: GraphLink) => {
    return link.value > 95 ? 1.5 : 0.5;
  }, []);

  // ── Node Click ──
  const handleNodeClick = useCallback((node: GraphNode) => {
    setSelectedNode(node);
  }, []);

  const connections = selectedNode ? getNodeConnections(selectedNode.id, graphData.links) : [];

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
        graphData={graphData}
        width={dimensions.width}
        height={dimensions.height}
        backgroundColor="#050505"
        nodeCanvasObject={nodeCanvasObject as never}
        nodePointerAreaPaint={(node: GraphNode, color: string, ctx: CanvasRenderingContext2D) => {
          const x = node.x ?? 0;
          const y = node.y ?? 0;
          ctx.beginPath();
          ctx.arc(x, y, 8, 0, 2 * Math.PI);
          ctx.fillStyle = color;
          ctx.fill();
        }}
        linkColor={linkColor as never}
        linkWidth={linkWidth as never}
        linkDirectionalParticles={1}
        linkDirectionalParticleWidth={(link: GraphLink) => link.value > 95 ? 2 : 0}
        linkDirectionalParticleColor={() => '#660000'}
        onNodeClick={handleNodeClick as never}
        cooldownTicks={100}
        d3AlphaDecay={0.02}
        d3VelocityDecay={0.3}
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

      {/* ── Status Bar ── */}
      <div className="absolute bottom-3 left-1/2 -translate-x-1/2 text-[9px] text-gray-600 font-mono tracking-widest pointer-events-none">
        NODES: {graphData.nodes.length} &nbsp;│&nbsp; EDGES: {graphData.links.length} &nbsp;│&nbsp; SOVEREIGN IDENTITY GRAPH
      </div>

      {/* ── Entity Dossier Side Panel ── */}
      <div
        className={`absolute top-0 right-0 h-full w-72 bg-[#0a0a0b] border-l-2 border-[#D4AF37]/40 transition-transform duration-300 ease-out z-20 flex flex-col font-mono ${
          selectedNode ? 'translate-x-0' : 'translate-x-full'
        }`}
      >
        {selectedNode && (
          <>
            {/* Header */}
            <div className="shrink-0 p-4 border-b border-[#1a1a1a]">
              <div className="flex justify-between items-start">
                <div>
                  <p className="text-[9px] text-gray-600 tracking-widest mb-1">ENTITY DOSSIER</p>
                  <h2 className="text-[#D4AF37] text-lg font-bold tracking-wider">
                    {selectedNode.name}
                  </h2>
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
            </div>

            {/* Connections List */}
            <div className="flex-1 overflow-y-auto p-4">
              <p className="text-[9px] text-gray-600 tracking-widest mb-3">VERIFIED BIOMETRIC LINKS</p>
              <div className="space-y-2">
                {connections
                  .sort((a, b) => b.score - a.score)
                  .map((conn) => (
                  <div
                    key={conn.entity}
                    className={`flex justify-between items-center p-2.5 rounded border ${
                      conn.score > 95
                        ? 'border-red-900/60 bg-red-950/20'
                        : conn.score > 85
                        ? 'border-[#D4AF37]/30 bg-[#D4AF37]/5'
                        : 'border-[#1f1f1f] bg-[#0d0d0e]'
                    }`}
                  >
                    <span className="text-xs text-gray-300 tracking-wider">{conn.entity}</span>
                    <span className={`text-xs font-bold tracking-wider ${
                      conn.score > 95
                        ? 'text-red-400'
                        : conn.score > 85
                        ? 'text-[#D4AF37]'
                        : 'text-gray-400'
                    }`}>
                      {conn.score.toFixed(1)}%
                    </span>
                  </div>
                ))}
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
    </div>
  );
}
