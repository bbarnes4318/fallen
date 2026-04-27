'use client';

import React, { useRef, useState, useEffect, useCallback } from 'react';

interface SymmetryMergeProps {
  galleryImageSrc: string;
  probeImageSrc: string;
  deltaImageSrc?: string;
  galleryWireframeSrc?: string;
  probeWireframeSrc?: string;
  isXrayMode?: boolean;
}

type ViewMode = 'aligned' | 'mesh' | 'delta' | 'overlap';

/* ── Tooltip descriptions for each view mode ── */
const VIEW_TOOLTIPS: Record<ViewMode, string> = {
  aligned: 'Procrustes normalized planar view. Scales and centers faces to eliminate distance and angle bias.',
  mesh: '1404-D geometric topology. Maps 468 precise anatomical anchor points.',
  delta: 'Heatmap of structural deviations. Red areas indicate geometric mismatch.',
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
  overlayOpacity?: number
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
  ctx.save();
  ctx.translate(ox, oy);
  ctx.scale(scale, scale);

  // Base image with configurable opacity (X-RAY dims this)
  ctx.globalAlpha = baseOpacity ?? 1.0;
  ctx.drawImage(baseImg, 0, 0, iw, ih);

  // Overlay with configurable opacity (X-RAY boosts this)
  if (overlayImg) {
    ctx.globalAlpha = overlayOpacity ?? 0.85;
    ctx.drawImage(overlayImg, 0, 0, iw, ih);
    ctx.globalAlpha = 1;
  }

  if (borderColor) {
    ctx.globalAlpha = 1;
    ctx.strokeStyle = borderColor;
    ctx.lineWidth = 3 / scale;
    ctx.strokeRect(0, 0, iw, ih);
  }

  ctx.restore();
}

/**
 * GLOBAL PANE ASSIGNMENT:
 *   Left  = Probe  (Unknown Target / User Upload)
 *   Right = Gallery (Vault Match / Known Alias)
 */
export default function SymmetryMerge({
  galleryImageSrc,
  probeImageSrc,
  deltaImageSrc,
  galleryWireframeSrc,
  probeWireframeSrc,
  isXrayMode = false,
}: SymmetryMergeProps) {
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

  // ── LEFT PANE = PROBE ──
  const getLeftOverlay = useCallback((): HTMLImageElement | null => {
    if (mode === 'mesh' && pWireImg) return pWireImg;
    if (mode === 'delta' && deltaImg) return deltaImg;
    return null;
  }, [mode, pWireImg, deltaImg]);

  // ── RIGHT PANE = GALLERY ──
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

  // X-RAY opacity values
  const hasOverlay = mode === 'mesh' || mode === 'delta';
  const baseOpacity = isXrayMode && hasOverlay ? 0.1 : 1.0;
  const overlayOpacity = isXrayMode && hasOverlay ? 1.0 : 0.85;

  // Draw dual panes — LEFT = PROBE, RIGHT = GALLERY
  useEffect(() => {
    if (!imagesReady || mode === 'overlap') return;

    if (leftCanvasRef.current && probeImg) {
      drawPane(leftCanvasRef.current, probeImg, getLeftOverlay(), zoom, pan, getBorderColor(), baseOpacity, overlayOpacity);
    }
    if (rightCanvasRef.current && galleryImg) {
      drawPane(rightCanvasRef.current, galleryImg, getRightOverlay(), zoom, pan, getBorderColor(), baseOpacity, overlayOpacity);
    }
  });

  // Draw overlap panes — LEFT = PROBE, RIGHT = GALLERY
  useEffect(() => {
    if (!imagesReady || mode !== 'overlap') return;

    if (overlapLeftRef.current && probeImg) {
      drawPane(overlapLeftRef.current, probeImg, null, zoom, pan);
    }
    if (overlapRightRef.current && galleryImg) {
      drawPane(overlapRightRef.current, galleryImg, null, zoom, pan);
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

          {/* Slider handle */}
          <div className="absolute top-0 bottom-0 pointer-events-none" style={{ left: `${overlapPos}%`, transform: 'translateX(-50%)' }}>
            <div className="w-[1px] h-full bg-[#D4AF37]/20" />
            <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-5 h-5 rounded-full border border-[#D4AF37]/50 bg-[#0A0A0B]/80 flex items-center justify-center backdrop-blur-sm">
              <svg className="w-2 h-2 text-[#D4AF37]/70" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 9l4-4 4 4M16 15l-4 4-4-4"></path></svg>
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
            <div className="absolute top-2 left-3 text-[9px] font-mono text-gray-600 tracking-widest pointer-events-none">PROBE (A)</div>
          </div>

          {/* Right Pane: Gallery */}
          <div
            className={`relative overflow-hidden rounded border ${mode === 'delta' ? 'border-red-900/60' : mode === 'mesh' ? 'border-[#D4AF37]/30' : 'border-[#333]'} bg-[#050505] cursor-move`}
            {...commonPaneEvents}
          >
            <canvas ref={rightCanvasRef} className="block w-full h-full" />
            <div className="absolute top-2 left-3 text-[9px] font-mono text-gray-600 tracking-widest pointer-events-none">GALLERY (B)</div>
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
          {isXrayMode && hasOverlay && <span className="ml-2 text-[#D4AF37] animate-pulse">· X-RAY ACTIVE</span>}
        </span>
        <span>SYNCHRONIZED · {Math.round(zoom * 100)}%</span>
      </div>
    </div>
  );
}
