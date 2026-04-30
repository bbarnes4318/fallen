'use client';

import React, { useRef, useState, useEffect, useCallback } from 'react';

interface SymmetryMergeProps {
  results: any | null;
  isXrayMode?: boolean;
}

type ViewMode = 'aligned' | 'mesh' | 'delta' | 'overlap';

/* ── Tooltip descriptions for each view mode ── */
const VIEW_TOOLTIPS: Record<ViewMode, string> = {
  aligned: 'Procrustes normalized planar view. Scales and centers faces to eliminate distance and angle bias.',
  mesh: '468-point MediaPipe face mesh overlay. Visualizes landmark positions used for alignment and geometric ratio extraction.',
  delta: 'Edge-based differential overlay between aligned gallery and probe crops. Highlights persistent structural deviations.',
  overlap: 'Alpha-blended composite layout for manual symmetry verification.',
};

function Tooltip({ text, children }: { text: string; children: React.ReactNode }) {
  return (
    <div className="relative group/tip">
      {children}
      <div className="pointer-events-none absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-56 opacity-0 group-hover/tip:opacity-100 transition-opacity duration-200 z-50">
        <div className="bg-[#111] border border-[#333] rounded px-3 py-2 text-[9px] text-gray-300 font-mono leading-relaxed shadow-[0_4px_20px_rgba(0,0,0,0.8)]">
          {text}
          <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-[#333]"></div>
        </div>
      </div>
    </div>
  );
}

function useImageLoader(src: string | undefined): HTMLImageElement | null {
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  useEffect(() => {
    if (!src) { setImg(null); return; }
    let cancelled = false;
    const el = new Image();
    el.crossOrigin = 'anonymous';
    el.src = src;
    el.onload = () => { if (!cancelled) setImg(el); };
    return () => { cancelled = true; };
  }, [src]);
  return img;
}

/** Draws an image to a canvas, scaled to fit with optional overlay */
function drawPane(
  canvas: HTMLCanvasElement,
  baseImg: HTMLImageElement,
  overlayImg: HTMLImageElement | null,
  zoom: number,
  pan: { x: number; y: number },
  borderColor?: string,
  baseOpacity?: number,
  overlayOpacity?: number,
  xrayFilter?: boolean,
  points?: {x: number, y: number, lr: number}[]
) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const cw = canvas.clientWidth;
  const ch = canvas.clientHeight;
  canvas.width = cw;
  canvas.height = ch;

  const iw = baseImg.width;
  const ih = baseImg.height;
  const baseScale = Math.min(cw / iw, ch / ih, 1);
  const scale = baseScale * zoom;
  const dw = iw * scale;
  const dh = ih * scale;
  const ox = (cw - dw) / 2 + pan.x;
  const oy = (ch - dh) / 2 + pan.y;

  ctx.clearRect(0, 0, cw, ch);

  // X-RAY forensic filter: high-contrast inverted grayscale when no overlay present
  if (xrayFilter && !overlayImg) {
    ctx.filter = 'contrast(1.8) grayscale(1) invert(0.85) brightness(1.2)';
  }

  ctx.save();
  ctx.translate(ox, oy);
  ctx.scale(scale, scale);

  // Base image with configurable opacity (X-RAY dims this when overlay is present)
  ctx.globalAlpha = baseOpacity ?? 1.0;
  ctx.drawImage(baseImg, 0, 0, iw, ih);

  // Reset filter before drawing overlay so it renders correctly
  if (xrayFilter && !overlayImg) {
    ctx.restore();
    ctx.save();
    ctx.translate(ox, oy);
    ctx.scale(scale, scale);
    ctx.filter = 'none';
  }

  // Overlay with configurable opacity (X-RAY boosts this)
  if (overlayImg) {
    ctx.globalAlpha = overlayOpacity ?? 0.85;
    ctx.drawImage(overlayImg, 0, 0, iw, ih);
    ctx.globalAlpha = 1;
  }

  if (points && points.length > 0) {
    points.forEach(p => {
      const px = p.x * iw;
      const py = p.y * ih;
      ctx.beginPath();
      ctx.arc(px, py, Math.max(2, 4 / scale), 0, 2 * Math.PI);
      ctx.fillStyle = 'rgba(212, 175, 55, 0.9)'; // Gold color for marks
      ctx.fill();
      ctx.lineWidth = 1.5 / scale;
      ctx.strokeStyle = '#111';
      ctx.stroke();
    });
  }

  if (borderColor) {
    ctx.globalAlpha = 1;
    ctx.filter = 'none';
    ctx.strokeStyle = borderColor;
    ctx.lineWidth = 3 / scale;
    ctx.strokeRect(0, 0, iw, ih);
  }

  ctx.restore();
  ctx.filter = 'none';
}

/**
 * GLOBAL PANE ASSIGNMENT:
 *   Left  = Probe  (Unknown Target / User Upload)
 *   Right = Gallery (Vault Match / Known Alias)
 */
export default function SymmetryMerge({
  results,
  isXrayMode = false,
}: SymmetryMergeProps) {
  const galleryImageSrc = results?.gallery_aligned_b64;
  const probeImageSrc = results?.probe_aligned_b64;
  const deltaImageSrc = results?.scar_delta_b64;
  const galleryWireframeSrc = results?.gallery_wireframe_b64;
  const probeWireframeSrc = results?.probe_wireframe_b64;
  const [mode, setMode] = useState<ViewMode>('aligned');
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [overlapPos, setOverlapPos] = useState(50);

  const isDragging = useRef(false);
  const lastMouse = useRef({ x: 0, y: 0 });

  // Canvas refs for dual panes
  const leftCanvasRef = useRef<HTMLCanvasElement>(null);
  const rightCanvasRef = useRef<HTMLCanvasElement>(null);
  // Overlap mode refs
  const overlapContainerRef = useRef<HTMLDivElement>(null);
  const overlapLeftRef = useRef<HTMLCanvasElement>(null);
  const overlapRightRef = useRef<HTMLCanvasElement>(null);

  // Load all images
  const galleryImg = useImageLoader(galleryImageSrc);
  const probeImg = useImageLoader(probeImageSrc);
  const deltaImg = useImageLoader(deltaImageSrc);
  const gWireImg = useImageLoader(galleryWireframeSrc);
  const pWireImg = useImageLoader(probeWireframeSrc);

  const imagesReady = !!galleryImg && !!probeImg;

  // ── LEFT PANE = PROBE (+ delta overlay in delta mode, + wireframe in mesh mode) ──
  const getLeftOverlay = useCallback((): HTMLImageElement | null => {
    if (mode === 'mesh' && pWireImg) return pWireImg;
    if (mode === 'delta' && deltaImg) return deltaImg;
    return null;
  }, [mode, pWireImg, deltaImg]);

  // ── RIGHT PANE = GALLERY (+ delta overlay in delta mode, + wireframe in mesh mode) ──
  const getRightOverlay = useCallback((): HTMLImageElement | null => {
    if (mode === 'mesh' && gWireImg) return gWireImg;
    if (mode === 'delta' && deltaImg) return deltaImg;
    return null;
  }, [mode, gWireImg, deltaImg]);

  const getBorderColor = (): string | undefined => {
    if (mode === 'delta') return 'rgba(180, 0, 30, 0.5)';
    if (mode === 'mesh') return 'rgba(212, 175, 55, 0.3)';
    return undefined;
  };

  // X-RAY opacity values — active when overlays are present
  const hasOverlay = mode === 'mesh' || mode === 'delta';
  const baseOpacity = isXrayMode && hasOverlay ? 0.1 : 1.0;
  const overlayOpacity = isXrayMode && hasOverlay ? 1.0 : (mode === 'delta' ? 0.7 : 0.85);

  const probePoints = results?.correspondences?.map((c: any) => ({
    x: c.probe_pt[0],
    y: c.probe_pt[1],
    lr: c.lr
  })) || [];

  const galleryPoints = results?.correspondences?.map((c: any) => ({
    x: c.gallery_pt[0],
    y: c.gallery_pt[1],
    lr: c.lr
  })) || [];

  // Draw dual panes — LEFT = PROBE, RIGHT = GALLERY (both get delta overlay in delta mode)
  useEffect(() => {
    if (!imagesReady || mode === 'overlap') return;

    if (leftCanvasRef.current && probeImg) {
      drawPane(leftCanvasRef.current, probeImg, getLeftOverlay(), zoom, pan, getBorderColor(), baseOpacity, overlayOpacity, isXrayMode, probePoints);
    }
    if (rightCanvasRef.current && galleryImg) {
      drawPane(rightCanvasRef.current, galleryImg, getRightOverlay(), zoom, pan, getBorderColor(), baseOpacity, overlayOpacity, isXrayMode, galleryPoints);
    }
  });

  // Draw overlap panes — LEFT = PROBE, RIGHT = GALLERY
  useEffect(() => {
    if (!imagesReady || mode !== 'overlap') return;

    if (overlapLeftRef.current && probeImg) {
      drawPane(overlapLeftRef.current, probeImg, null, zoom, pan, undefined, undefined, undefined, isXrayMode, probePoints);
    }
    if (overlapRightRef.current && galleryImg) {
      drawPane(overlapRightRef.current, galleryImg, null, zoom, pan, undefined, undefined, undefined, isXrayMode, galleryPoints);
    }
  });

  // Pan handler
  const handlePointerDown = (cx: number, cy: number) => {
    isDragging.current = true;
    lastMouse.current = { x: cx, y: cy };
  };

  const handlePointerMove = (cx: number, cy: number) => {
    if (!isDragging.current) return;
    if (mode === 'overlap' && overlapContainerRef.current) {
      const rect = overlapContainerRef.current.getBoundingClientRect();
      const x = Math.max(0, Math.min(cx - rect.left, rect.width));
      setOverlapPos((x / rect.width) * 100);
    } else {
      const dx = cx - lastMouse.current.x;
      const dy = cy - lastMouse.current.y;
      setPan((prev: { x: number; y: number }) => ({ x: prev.x + dx, y: prev.y + dy }));
      lastMouse.current = { x: cx, y: cy };
    }
  };

  const handlePointerUp = () => { isDragging.current = false; };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const d = e.deltaY < 0 ? 1.05 : 0.95;
    setZoom((z: number) => Math.min(Math.max(z * d, 1), 8));
  };

  const commonPaneEvents = {
    onMouseDown: (e: React.MouseEvent) => handlePointerDown(e.clientX, e.clientY),
    onMouseUp: handlePointerUp,
    onMouseLeave: handlePointerUp,
    onMouseMove: (e: React.MouseEvent) => handlePointerMove(e.clientX, e.clientY),
    onWheel: handleWheel,
    onTouchStart: (e: React.TouchEvent) => handlePointerDown(e.touches[0].clientX, e.touches[0].clientY),
    onTouchEnd: handlePointerUp,
    onTouchMove: (e: React.TouchEvent) => handlePointerMove(e.touches[0].clientX, e.touches[0].clientY),
  };

  return (
    <div className="flex flex-col h-full w-full min-h-0 relative">
      {/* ── Toolbar ── */}
      <div className="flex justify-between items-center px-1 pb-1.5 shrink-0 select-none">
        <div>
          <h2 className="text-sm font-bold text-gray-100 font-mono tracking-wider leading-tight">FORENSIC INSPECTOR</h2>
          <p className="text-[10px] text-gray-500 font-mono">Dual-pane synchronized view · Probe ↔ Gallery</p>
        </div>

        <div className="flex gap-2">
          {/* Mode Toggle with Tooltips */}
          <div className="flex border border-[#333] rounded bg-[#111] overflow-hidden text-[10px] font-mono">
            <Tooltip text={VIEW_TOOLTIPS.aligned}>
              <button onClick={() => setMode('aligned')} className={`px-3 py-1 transition-colors ${mode === 'aligned' ? 'bg-[#D4AF37] text-black font-bold' : 'text-gray-400 hover:text-white'}`}>
                ALIGNED
              </button>
            </Tooltip>
            {galleryWireframeSrc && probeWireframeSrc && (
              <Tooltip text={VIEW_TOOLTIPS.mesh}>
                <button onClick={() => setMode('mesh')} className={`px-3 py-1 transition-colors border-l ${mode === 'mesh' ? 'bg-[#D4AF37] text-black font-bold border-[#D4AF37]' : 'text-gray-400 hover:text-[#D4AF37] border-[#333]'}`}>
                  MESH
                </button>
              </Tooltip>
            )}
            {deltaImageSrc && (
              <Tooltip text={VIEW_TOOLTIPS.delta}>
                <button onClick={() => setMode('delta')} className={`px-3 py-1 transition-colors border-l ${mode === 'delta' ? 'bg-[#1a0005] text-[#ff2040] font-bold border-[#5a0015] shadow-[inset_0_0_12px_rgba(180,0,30,0.3)]' : 'text-gray-400 hover:text-red-300 border-[#333]'}`}>
                  DELTA
                </button>
              </Tooltip>
            )}
            <Tooltip text={VIEW_TOOLTIPS.overlap}>
              <button onClick={() => setMode('overlap')} className={`px-3 py-1 transition-colors border-l ${mode === 'overlap' ? 'bg-[#D4AF37] text-black font-bold border-[#D4AF37]' : 'text-gray-400 hover:text-[#D4AF37] border-[#333]'}`}>
                OVERLAP
              </button>
            </Tooltip>
          </div>

          {/* Zoom Controls */}
          <div className="flex border border-[#333] rounded bg-[#111] overflow-hidden text-gray-400">
            <button onClick={() => setZoom((z: number) => Math.max(1, z - 0.2))} className="px-2 hover:bg-[#222] hover:text-white transition-colors">-</button>
            <div className="px-2 py-1 text-[10px] border-x border-[#333] font-mono min-w-[45px] text-center">
              {Math.round(zoom * 100)}%
            </div>
            <button onClick={() => setZoom((z: number) => Math.min(8, z + 0.2))} className="px-2 hover:bg-[#222] hover:text-white transition-colors">+</button>
            <button onClick={() => { setZoom(1); setPan({x:0, y:0}); }} className="px-2 py-1 text-[10px] border-l border-[#333] hover:bg-[#222] hover:text-[#D4AF37] transition-colors">
              RESET
            </button>
          </div>
        </div>
      </div>

      {/* ── High-Density Telemetry HUD ── */}
      {results && (
        <div className="mb-2 flex flex-col gap-[2px] shrink-0 font-mono text-[10px] uppercase select-none">
          {/* Provenance Veto Row */}
          {results.failed_provenance_veto && (
            <div className="px-2 py-1.5 flex justify-between items-center bg-[#1a0005] border border-[#5a0015] text-[#ff2040]">
              <span className="font-bold tracking-widest text-xs">DEEPFAKE VETO: SYNTHETIC PROVENANCE DETECTED</span>
              <span className="tracking-widest font-bold opacity-90">FUSION ABORTED</span>
            </div>
          )}

          {/* Main Verdict Row */}
          {!results.failed_provenance_veto && (
            <div className={`px-2 py-1.5 flex justify-between items-center border ${results.veto_triggered ? 'bg-[#1a0005] border-[#5a0015] text-[#ff2040]' : (results.fused_identity_score >= 40.0 ? 'bg-[#111100] border-[#D4AF37]/40 text-[#D4AF37]' : 'bg-[#0a0a0a] border-[#333] text-gray-400')}`}>
              <span className="font-bold tracking-wider text-xs">
                {results.veto_triggered ? 'VERDICT: MISMATCH (ARCFACE VETO)' : (results.fused_identity_score >= 40.0 ? 'VERDICT: MATCH' : 'VERDICT: INCONCLUSIVE')}
              </span>
              <span className="tracking-widest font-bold">FUSED SCORE: {results.fused_identity_score?.toFixed(2)}%</span>
            </div>
          )}

          {/* Telemetry Data Grid */}
          <div className="grid grid-cols-2 gap-[2px]">
            {/* Provenance Module */}
            <div className={`px-2 py-1 border flex justify-between items-center ${results.failed_provenance_veto ? 'bg-[#1a0005] border-[#5a0015] text-[#ff2040]' : 'bg-[#050505] border-[#222] text-gray-500'}`}>
               <span className="tracking-widest text-[9px]">SYNTH_ANOMALY:</span>
               <span className="font-bold text-gray-300">{results.synthetic_anomaly_score !== undefined ? results.synthetic_anomaly_score.toFixed(4) : 'N/A'}</span>
            </div>

            {/* Occlusion Module */}
            <div className="px-2 py-1 border bg-[#050505] border-[#222] text-gray-500 flex justify-between items-center">
              <span className="tracking-widest text-[9px]">OCCLUSION (RATIOS):</span>
              <span className="font-bold text-gray-300">
                {results.occlusion_percentage !== undefined ? `${(results.occlusion_percentage).toFixed(1)}% (${results.effective_geometric_ratios_used ?? 0} ACTIVE)` : 'N/A'}
              </span>
            </div>
          </div>

          {/* Dynamic Lists (Occlusions & Marks) */}
          {(results.occluded_regions?.length > 0 || results.correspondences?.length > 0) && (
            <div className="flex flex-col gap-[2px]">
              {results.occluded_regions && results.occluded_regions.length > 0 && (
                <div className="flex gap-[2px] flex-wrap">
                  {results.occluded_regions.map((region: string, i: number) => (
                    <div key={`occ-${i}`} className="px-1.5 py-0.5 border border-[#8a4000]/40 bg-[#3a1500]/20 text-[#ff8800]/80 tracking-widest text-[9px]">
                      MASKED: {region}
                    </div>
                  ))}
                </div>
              )}
              {results.correspondences && results.correspondences.length > 0 && (
                <div className="flex gap-[2px] flex-wrap">
                  {results.correspondences.map((c: any, i: number) => (
                    <div key={`corr-${i}`} className={`px-1.5 py-0.5 border bg-[#050505] tracking-widest text-[9px] ${results.veto_triggered || results.failed_provenance_veto ? 'border-[#5a0015] text-[#ff2040]/70' : 'border-[#D4AF37]/30 text-[#D4AF37]/80'}`}>
                      MARK {i+1} <span className="opacity-50 mx-0.5">LR:</span>{c.lr.toFixed(1)}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* ── Dual-Pane Viewport ── */}
      {!imagesReady ? (
        <div className="flex-1 min-h-0 flex items-center justify-center bg-[#0a0a0a] animate-pulse text-[#D4AF37] font-mono text-xs tracking-widest rounded border border-[#333]">
          LOADING BIO-DATA...
        </div>
      ) : mode === 'overlap' ? (
        /* ── OVERLAP MODE: Clip-path slider — Left=Probe, Right=Gallery ── */
        <div
          ref={overlapContainerRef}
          className="flex-1 min-h-0 relative overflow-hidden rounded border border-[#D4AF37]/30 bg-[#050505] cursor-col-resize"
          {...commonPaneEvents}
        >
          {/* Bottom layer: Gallery (full) */}
          <canvas ref={overlapRightRef} className="absolute inset-0 w-full h-full" />

          {/* Top layer: Probe (clipped) */}
          <div className="absolute inset-0" style={{ clipPath: `inset(0 ${100 - overlapPos}% 0 0)` }}>
            <canvas ref={overlapLeftRef} className="w-full h-full" />
          </div>

          {/* Slider handle — minimal line with small edge chevrons */}
          <div className="absolute top-0 bottom-0 pointer-events-none" style={{ left: `${overlapPos}%`, transform: 'translateX(-50%)' }}>
            <div className="w-[2px] h-full bg-[#D4AF37]/40 shadow-[0_0_6px_rgba(212,175,55,0.3)]" />
            {/* Top chevron */}
            <div className="absolute top-3 left-1/2 -translate-x-1/2">
              <svg className="w-3 h-3 text-[#D4AF37]/50" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 19l-7-7 7-7"></path></svg>
            </div>
            {/* Bottom chevron */}
            <div className="absolute bottom-3 left-1/2 -translate-x-1/2">
              <svg className="w-3 h-3 text-[#D4AF37]/50" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7"></path></svg>
            </div>
          </div>

          {/* Labels */}
          <div className="absolute top-2 left-3 text-[9px] font-mono text-[#D4AF37]/60 tracking-widest pointer-events-none">PROBE</div>
          <div className="absolute top-2 right-3 text-[9px] font-mono text-[#D4AF37]/60 tracking-widest pointer-events-none">GALLERY</div>
        </div>
      ) : (
        /* ── DUAL-PANE: Left=Probe, Right=Gallery ── */
        <div className="flex-1 min-h-0 grid grid-cols-2 gap-1">
          {/* Left Pane: Probe */}
          <div
            className={`relative overflow-hidden rounded border ${mode === 'delta' ? 'border-red-900/60' : mode === 'mesh' ? 'border-[#D4AF37]/30' : 'border-[#333]'} bg-[#050505] cursor-move`}
            {...commonPaneEvents}
          >
            <canvas ref={leftCanvasRef} className="block w-full h-full" />
            <div className="absolute top-2 left-3 text-[9px] font-mono text-gray-600 tracking-widest pointer-events-none">{mode === 'delta' ? <span className="text-red-500">PROBE + DELTA</span> : 'PROBE (A)'}</div>
          </div>

          {/* Right Pane: Gallery */}
          <div
            className={`relative overflow-hidden rounded border ${mode === 'delta' ? 'border-red-900/60' : mode === 'mesh' ? 'border-[#D4AF37]/30' : 'border-[#333]'} bg-[#050505] cursor-move`}
            {...commonPaneEvents}
          >
            <canvas ref={rightCanvasRef} className="block w-full h-full" />
            <div className="absolute top-2 left-3 text-[9px] font-mono text-gray-600 tracking-widest pointer-events-none">{mode === 'delta' ? <span className="text-red-500">GALLERY + DELTA</span> : 'GALLERY (B)'}</div>
          </div>
        </div>
      )}

      {/* ── Footer ── */}
      <div className="flex w-full justify-between pt-1 text-[9px] font-mono text-gray-600 tracking-widest shrink-0">
        <span>
          {mode === 'aligned' && 'CANONICAL ALIGNMENT'}
          {mode === 'mesh' && <span className="text-[#D4AF37]">3DMM WIREFRAME HUD</span>}
          {mode === 'delta' && <span className="text-red-500">BIOLOGICAL TOPOGRAPHY DELTA</span>}
          {mode === 'overlap' && <span className="text-[#D4AF37]">DRAG TO COMPARE OVERLAP</span>}
          {isXrayMode && <span className="ml-2 text-[#D4AF37] animate-pulse">· X-RAY ACTIVE</span>}
        </span>
        <span>SYNCHRONIZED · {Math.round(zoom * 100)}%</span>
      </div>
    </div>
  );
}
