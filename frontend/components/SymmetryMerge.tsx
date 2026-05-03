'use client';

import React, { useRef, useState, useEffect, useCallback, useMemo } from 'react';
import {
  VerificationResult,
  RawPoint,
  Correspondence,
  MarkDebugCorrespondence,
  MarkDescriptor,
  ScoringTrace,
  MarkMatchStatus
} from '@/types/verification';

type ForensicPoint = {
  x: number;
  y: number;
  area?: number;
  lr?: number;
  isMatched?: boolean;
  matchIndex?: number;
  isRejected?: boolean;
};

interface SymmetryMergeProps {
  results: VerificationResult | null;
  isXrayMode?: boolean;
}

type ViewMode = 'aligned' | 'mesh' | 'delta' | 'overlap' | 'marks' | 'debug';

/* ── Tooltip descriptions for each view mode ── */
const VIEW_TOOLTIPS: Record<ViewMode, string> = {
  aligned: 'Procrustes normalized planar view. Scales and centers faces to eliminate distance and angle bias.',
  mesh: '468-point MediaPipe face mesh overlay. Visualizes landmark positions used for alignment and geometric ratio extraction.',
  delta: 'Pixel/edge difference map between aligned crops. Shows structural deviations only — NOT scar, mole, blemish, or mark correspondence evidence.',
  overlap: 'Alpha-blended composite layout for manual symmetry verification.',
  marks: 'Forensic mark visualization. Shows all raw detected marks (unmatched in cyan, matched correspondences in green) with diagnostic counts.',
  debug: 'Raw backend mark debug visualization with OpenCV overlays.',
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
  const [currentSrc, setCurrentSrc] = useState(src);

  if (src !== currentSrc) {
    setCurrentSrc(src);
    setImg(null);
  }

  useEffect(() => {
    if (!src) return;
    let cancelled = false;
    const el = new window.Image();
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
  points?: ForensicPoint[],
  rejectedPoints?: ForensicPoint[]
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
      if (p.x === undefined || p.y === undefined) return;
      const px = p.x * iw;
      const py = p.y * ih;
      const r = Math.max(4, Math.sqrt(p.area || 0) * 0.8);
      
      ctx.beginPath();
      ctx.arc(px, py, r, 0, 2 * Math.PI);
      
      if (p.isMatched === true) {
        // Green — explicit matched correspondence
        ctx.strokeStyle = 'rgba(0, 220, 80, 0.9)';
        ctx.lineWidth = 2 / scale;
        ctx.stroke();

        if (p.matchIndex !== undefined) {
          ctx.fillStyle = 'rgba(0, 220, 80, 0.9)';
          ctx.font = `bold ${11 / scale}px monospace`;
          ctx.fillText((p.matchIndex + 1).toString(), px + r + (4 / scale), py + (4 / scale));
        }
      } else if (p.isMatched === false) {
        // Cyan — detected but not matched by index
        ctx.strokeStyle = 'rgba(0, 200, 220, 0.9)';
        ctx.lineWidth = 1 / scale;
        ctx.stroke();
      } else {
        // Gray — unknown (no index correspondence data)
        ctx.strokeStyle = 'rgba(150, 150, 150, 0.6)';
        ctx.lineWidth = 1 / scale;
        ctx.stroke();
      }
    });
  }

  // Rejected candidates — dashed amber/red outlines (DEBUG_FORENSIC only)
  if (rejectedPoints && rejectedPoints.length > 0) {
    rejectedPoints.forEach(p => {
      if (p.x === undefined || p.y === undefined) return;
      const px = p.x * iw;
      const py = p.y * ih;
      const r = Math.max(4, Math.sqrt(p.area || 0) * 0.8);

      ctx.beginPath();
      ctx.setLineDash([3 / scale, 3 / scale]);
      ctx.arc(px, py, r, 0, 2 * Math.PI);
      ctx.strokeStyle = 'rgba(220, 140, 0, 0.7)';
      ctx.lineWidth = 1.5 / scale;
      ctx.stroke();
      ctx.setLineDash([]);
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
  const deltaImageSrc = results?.edge_delta_b64 ?? results?.scar_delta_b64;
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
  const pMarkDebugImg = useImageLoader(results?.probe_mark_debug_b64 ?? undefined);
  const gMarkDebugImg = useImageLoader(results?.gallery_mark_debug_b64 ?? undefined);

  const imagesReady = !!galleryImg && !!probeImg;

  // ── Three-tier debug flag separation ──
  const forensicDebugEnabled =
    process.env.NEXT_PUBLIC_DEBUG_FORENSIC === "true";
  const isDebugMode = mode === "debug";
  // Full forensic debug details require BOTH env flag AND debug tab
  const showForensicDebugDetails =
    forensicDebugEnabled && isDebugMode;
  // Only draw frontend mark circles as fallback when:
  // 1. Debug mode active, AND
  // 2. Backend did NOT provide pre-rendered debug overlay images
  const shouldDrawFrontendMarkPoints =
    mode === "marks" ||
    (isDebugMode &&
    !results?.probe_mark_debug_b64 &&
    !results?.gallery_mark_debug_b64);

  // ── LEFT PANE = PROBE (wireframe in mesh, debug overlay in debug — NO delta overlay) ──
  const getLeftOverlay = useCallback((): HTMLImageElement | null => {
    if (mode === 'mesh' && pWireImg) return pWireImg;
    if (mode === 'debug' && pMarkDebugImg) return pMarkDebugImg;
    return null;
  }, [mode, pWireImg, pMarkDebugImg]);

  // ── RIGHT PANE = GALLERY (wireframe in mesh, debug overlay in debug — NO delta overlay) ──
  const getRightOverlay = useCallback((): HTMLImageElement | null => {
    if (mode === 'mesh' && gWireImg) return gWireImg;
    if (mode === 'debug' && gMarkDebugImg) return gMarkDebugImg;
    return null;
  }, [mode, gWireImg, gMarkDebugImg]);

  const getBorderColor = useCallback((): string | undefined => {
    if (mode === 'delta') return 'rgba(180, 0, 30, 0.5)';
    if (mode === 'mesh') return 'rgba(212, 175, 55, 0.3)';
    return undefined;
  }, [mode]);

  // X-RAY opacity values — active when overlays are present
  const hasOverlay = mode === 'mesh' || mode === 'delta' || mode === 'debug';
  const baseOpacity = isXrayMode && hasOverlay ? 0.1 : 1.0;
  const overlayOpacity = isXrayMode && hasOverlay ? 1.0 : (mode === 'delta' ? 0.7 : 0.85);

  const normalizeMarks = (marks: unknown): RawPoint[] => {
    return Array.isArray(marks) ? (marks as RawPoint[]) : [];
  };

  const normalizeStringArray = (value: unknown): string[] => {
    return Array.isArray(value)
      ? value.filter((item): item is string => typeof item === "string")
      : [];
  };

  const normalizeCorrespondences = (value: unknown): Correspondence[] => {
    return Array.isArray(value) ? (value as Correspondence[]) : [];
  };

  const getPointCoords = useCallback((pt: RawPoint) => {
    if (!pt) return { x: undefined, y: undefined, area: 0 };
    if ('centroid' in pt && pt.centroid) return { x: pt.centroid[0], y: pt.centroid[1], area: ('area' in pt ? pt.area : 0) || 0 };
    if (Array.isArray(pt)) return { x: pt[0], y: pt[1], area: pt[2] || 0 };
    if ('x' in pt) return { x: pt.x, y: pt.y, area: ('area' in pt ? pt.area : 0) || 0 };
    return { x: undefined, y: undefined, area: 0 };
  }, []);

  // ── Compute canonical mark source arrays once ──
  const probeMarksSource = useMemo((): RawPoint[] => {
    const raw = normalizeMarks(results?.raw_probe_marks);
    return raw.length > 0 ? raw : normalizeMarks(results?.probe_data?.marks);
  }, [results]);

  const galleryMarksSource = useMemo((): RawPoint[] => {
    const raw = normalizeMarks(results?.raw_gallery_marks);
    return raw.length > 0 ? raw : normalizeMarks(results?.gallery_data?.marks);
  }, [results]);

  // ── Build matched index sets and LR maps from explicit correspondence indices ──
  const markMatchData = useMemo(() => {
    const corrs = normalizeCorrespondences(results?.correspondences);

    const matchedProbeIndices = new Set<number>();
    const matchedGalleryIndices = new Set<number>();
    const probeLrByIndex = new Map<number, number>();
    const galleryLrByIndex = new Map<number, number>();
    const probeIndexToPairId = new Map<number, number>();
    const galleryIndexToPairId = new Map<number, number>();

    let pairId = 0;
    for (const c of corrs) {
      const lr = typeof c.lr === "number" ? c.lr : undefined;
      const pIdx = c.probe_idx as number;
      const gIdx = c.gallery_idx as number;

      const pValid = Number.isInteger(pIdx) && pIdx >= 0 && pIdx < probeMarksSource.length;
      const gValid = Number.isInteger(gIdx) && gIdx >= 0 && gIdx < galleryMarksSource.length;

      if (pValid && gValid) {
        matchedProbeIndices.add(pIdx);
        matchedGalleryIndices.add(gIdx);
        if (lr !== undefined) {
          probeLrByIndex.set(pIdx, lr);
          galleryLrByIndex.set(gIdx, lr);
        }
        probeIndexToPairId.set(pIdx, pairId);
        galleryIndexToPairId.set(gIdx, pairId);
        pairId++;
      } else {
        if (pValid) {
          matchedProbeIndices.add(pIdx);
          if (lr !== undefined) probeLrByIndex.set(pIdx, lr);
        }
        if (gValid) {
          matchedGalleryIndices.add(gIdx);
          if (lr !== undefined) galleryLrByIndex.set(gIdx, lr);
        }
      }
    }

    return {
      corrs,
      matchedProbeIndices,
      matchedGalleryIndices,
      probeLrByIndex,
      galleryLrByIndex,
      probeIndexToPairId,
      galleryIndexToPairId,
      hasIndexCorrespondences: matchedProbeIndices.size > 0 || matchedGalleryIndices.size > 0,
    };
  }, [results, probeMarksSource.length, galleryMarksSource.length]);

  const mapPoint = useCallback(
    (m: RawPoint, index: number, side: "probe" | "gallery"): ForensicPoint | null => {
      const { x, y, area } = getPointCoords(m);
      if (x === undefined || y === undefined) return null;

      const isProbe = side === "probe";
      const matchedSet = isProbe
        ? markMatchData.matchedProbeIndices
        : markMatchData.matchedGalleryIndices;
      const lrMap = isProbe
        ? markMatchData.probeLrByIndex
        : markMatchData.galleryLrByIndex;
      const matchIndexMap = isProbe
        ? markMatchData.probeIndexToPairId
        : markMatchData.galleryIndexToPairId;

      return {
        x,
        y,
        area,
        lr: lrMap.get(index),
        isMatched: markMatchData.hasIndexCorrespondences
          ? matchedSet.has(index)
          : undefined,
        matchIndex: matchIndexMap.get(index),
      };
    },
    [getPointCoords, markMatchData]
  );

  const probePoints = useMemo((): ForensicPoint[] => {
    return probeMarksSource
      .map((m: RawPoint, i: number) => mapPoint(m, i, "probe"))
      .filter((p): p is ForensicPoint => p !== null);
  }, [probeMarksSource, mapPoint]);

  const galleryPoints = useMemo((): ForensicPoint[] => {
    return galleryMarksSource
      .map((m: RawPoint, i: number) => mapPoint(m, i, "gallery"))
      .filter((p): p is ForensicPoint => p !== null);
  }, [galleryMarksSource, mapPoint]);

  // ── Rejected points (DEBUG_FORENSIC only) ──
  const mapRejectedMark = useCallback((m: any): ForensicPoint | null => {
    const x = Array.isArray(m.centroid)
      ? m.centroid[0]
      : typeof m.x === "number"
      ? m.x
      : undefined;

    const y = Array.isArray(m.centroid)
      ? m.centroid[1]
      : typeof m.y === "number"
      ? m.y
      : undefined;

    if (
      typeof x !== "number" ||
      typeof y !== "number" ||
      !Number.isFinite(x) ||
      !Number.isFinite(y)
    ) {
      return null;
    }

    return {
      x,
      y,
      area: typeof m.area === "number" ? m.area : 0,
      isRejected: true,
    };
  }, []);

  const rejectedProbePoints = useMemo((): ForensicPoint[] => {
    if (!forensicDebugEnabled) return [];

    const rejected = results?.mark_debug?.rejected_probe_marks;
    if (!Array.isArray(rejected)) return [];

    return rejected
      .map((m: any) => mapRejectedMark(m))
      .filter((p): p is ForensicPoint => p !== null);
  }, [forensicDebugEnabled, results, mapRejectedMark]);

  const rejectedGalleryPoints = useMemo((): ForensicPoint[] => {
    if (!forensicDebugEnabled) return [];

    const rejected = results?.mark_debug?.rejected_gallery_marks;
    if (!Array.isArray(rejected)) return [];

    return rejected
      .map((m: any) => mapRejectedMark(m))
      .filter((p): p is ForensicPoint => p !== null);
  }, [forensicDebugEnabled, results, mapRejectedMark]);

  // Draw dual panes — LEFT = PROBE, RIGHT = GALLERY (both get delta overlay in delta mode)
  useEffect(() => {
    if (!imagesReady || mode === 'overlap') return;

    // MARKS mode: show ALL raw marks (matched green, unmatched cyan). No filtering.
    const probePts = shouldDrawFrontendMarkPoints ? probePoints : undefined;
    const galleryPts = shouldDrawFrontendMarkPoints ? galleryPoints : undefined;
    const rejProbePts = (shouldDrawFrontendMarkPoints && forensicDebugEnabled) ? rejectedProbePoints : undefined;
    const rejGalleryPts = (shouldDrawFrontendMarkPoints && forensicDebugEnabled) ? rejectedGalleryPoints : undefined;

    if (leftCanvasRef.current && probeImg) {
      drawPane(leftCanvasRef.current, probeImg, getLeftOverlay(), zoom, pan, getBorderColor(), baseOpacity, overlayOpacity, isXrayMode, probePts, rejProbePts);
    }
    if (rightCanvasRef.current && galleryImg) {
      drawPane(rightCanvasRef.current, galleryImg, getRightOverlay(), zoom, pan, getBorderColor(), baseOpacity, overlayOpacity, isXrayMode, galleryPts, rejGalleryPts);
    }
  }, [imagesReady, mode, probeImg, getLeftOverlay, zoom, pan, getBorderColor, baseOpacity, overlayOpacity, isXrayMode, probePoints, galleryImg, getRightOverlay, galleryPoints, shouldDrawFrontendMarkPoints, forensicDebugEnabled, rejectedProbePoints, rejectedGalleryPoints]);

  // Draw overlap panes — LEFT = PROBE, RIGHT = GALLERY
  useEffect(() => {
    if (!imagesReady || mode !== 'overlap') return;

    if (overlapLeftRef.current && probeImg) {
      drawPane(overlapLeftRef.current, probeImg, null, zoom, pan, undefined, undefined, undefined, isXrayMode, undefined);
    }
    if (overlapRightRef.current && galleryImg) {
      drawPane(overlapRightRef.current, galleryImg, null, zoom, pan, undefined, undefined, undefined, isXrayMode, undefined);
    }
  }, [imagesReady, mode, probeImg, zoom, pan, isXrayMode, probePoints, galleryImg, galleryPoints]);

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
                  PIXEL DELTA
                </button>
              </Tooltip>
            )}
            <Tooltip text={VIEW_TOOLTIPS.overlap}>
              <button onClick={() => setMode('overlap')} className={`px-3 py-1 transition-colors border-l ${mode === 'overlap' ? 'bg-[#D4AF37] text-black font-bold border-[#D4AF37]' : 'text-gray-400 hover:text-[#D4AF37] border-[#333]'}`}>
                OVERLAP
              </button>
            </Tooltip>
            <Tooltip text={VIEW_TOOLTIPS.marks}>
              <button onClick={() => setMode('marks')} className={`px-3 py-1 transition-colors border-l ${mode === 'marks' ? 'bg-emerald-900 text-emerald-200 font-bold border-emerald-700' : 'text-gray-400 hover:text-emerald-300 border-[#333]'}`}>
                MARKS
              </button>
            </Tooltip>
            {results?.probe_mark_debug_b64 && results?.gallery_mark_debug_b64 && (
              <Tooltip text={VIEW_TOOLTIPS.debug}>
                <button onClick={() => setMode('debug')} className={`px-3 py-1 transition-colors border-l ${mode === 'debug' ? 'bg-[#D4AF37] text-black font-bold border-[#D4AF37]' : 'text-gray-400 hover:text-[#D4AF37] border-[#333]'}`}>
                  DEBUG
                </button>
              </Tooltip>
            )}
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
          {!results.failed_provenance_veto && (() => {
            const fusedScore = results.fused_identity_score ?? 0;
            const bayesianPreVeto = results.bayesian_fused_score ?? ((results.audit_log?.posterior_probability ?? 0) * 100);
            return (
              <>
                <div className={`px-2 py-1.5 flex justify-between items-center border ${results.veto_triggered ? 'bg-[#1a0005] border-[#5a0015] text-[#ff2040]' : (fusedScore >= 40.0 ? 'bg-[#111100] border-[#D4AF37]/40 text-[#D4AF37]' : 'bg-[#0a0a0a] border-[#333] text-gray-400')}`}>
                  <span className="font-bold tracking-wider text-xs">
                    {results.veto_triggered ? 'BELOW THRESHOLD (FACE MODEL VETO)' : (fusedScore >= 40.0 ? 'EVIDENCE SUPPORTS COMMON SOURCE' : 'INCONCLUSIVE')}
                  </span>
                  <span className="tracking-widest font-bold">POSTERIOR PROBABILITY: {fusedScore.toFixed(2)}%</span>
                </div>
                {results.veto_triggered && (
                  <div className="px-2 py-1 flex justify-between items-center border border-[#5a0015]/60 bg-[#0d0002] text-[#ff6070] text-[9px] tracking-wider">
                    <span>BAYESIAN PRE-VETO: {bayesianPreVeto.toFixed(2)}%</span>
                    <span>DISPLAYED: 0% — {results.veto_reason === 'ARCFACE_VETO' ? 'FACE MODEL VETO' : results.veto_reason} POLICY</span>
                  </div>
                )}
              </>
            );
          })()}

          {/* Telemetry Data Grid */}
          <div className="grid grid-cols-2 gap-[2px]">
            {/* Provenance Module */}
            <div className={`px-2 py-1 border flex justify-between items-center ${results.failed_provenance_veto ? 'bg-[#1a0005] border-[#5a0015] text-[#ff2040]' : 'bg-[#050505] border-[#222] text-gray-500'}`}>
               <span className="tracking-widest text-[9px]">PROVENANCE CHECK:</span>
               <span className="font-bold text-gray-300">
                 {results.failed_provenance_veto === true
                   ? "FAILED — synthetic anomaly detected"
                   : typeof results.synthetic_anomaly_score === "number"
                   ? `PASSED · score ${results.synthetic_anomaly_score.toFixed(4)}`
                   : "Not evaluated"}
               </span>
            </div>

            {/* Occlusion Module */}
            <div className="px-2 py-1 border bg-[#050505] border-[#222] text-gray-500 flex justify-between items-center">
              <span className="tracking-widest text-[9px]">GEOMETRY COVERAGE:</span>
              <span className="font-bold text-gray-300">
                {typeof results.effective_geometric_ratios_used === "number"
                  ? `${results.effective_geometric_ratios_used} ratios active${typeof results.occlusion_percentage === "number" ? ` · ${results.occlusion_percentage.toFixed(1)}% occluded` : ''}`
                  : typeof results.occlusion_percentage === "number"
                  ? `${results.occlusion_percentage.toFixed(1)}% occluded`
                  : "Not evaluated"}
              </span>
            </div>
          </div>

          {/* Dynamic Lists (Occlusions & Marks) */}
          {(() => {
            const probeOccludedRegions = normalizeStringArray(results?.probe_data?.occluded_regions);
            const galleryOccludedRegions = normalizeStringArray(results?.gallery_data?.occluded_regions);
            const legacyOccludedRegions = normalizeStringArray(results?.occluded_regions);
            const safeCorrespondences = normalizeCorrespondences(results?.correspondences);

            const hasDynamicLists =
              probeOccludedRegions.length > 0 ||
              galleryOccludedRegions.length > 0 ||
              legacyOccludedRegions.length > 0 ||
              safeCorrespondences.length > 0;

            if (!hasDynamicLists) return null;

            return (
              <div className="flex flex-col gap-[2px]">
                {legacyOccludedRegions.length > 0 && !results.probe_data && !results.gallery_data && (
                  <div className="flex gap-[2px] flex-wrap">
                    {legacyOccludedRegions.map((region: string, i: number) => (
                      <div key={`occ-${i}`} className="px-1.5 py-0.5 border border-[#8a4000]/40 bg-[#3a1500]/20 text-[#ff8800]/80 tracking-widest text-[9px]">
                        MASKED: {region}
                      </div>
                    ))}
                  </div>
                )}
                {probeOccludedRegions.length > 0 && (
                  <div className="flex gap-[2px] flex-wrap">
                    {probeOccludedRegions.map((region: string, i: number) => (
                      <div key={`probe-occ-${i}`} className="px-1.5 py-0.5 border border-[#8a4000]/40 bg-[#3a1500]/20 text-[#ff8800]/80 tracking-widest text-[9px]">
                        PROBE MASK: {region}
                      </div>
                    ))}
                  </div>
                )}
                {galleryOccludedRegions.length > 0 && (
                  <div className="flex gap-[2px] flex-wrap">
                    {galleryOccludedRegions.map((region: string, i: number) => (
                      <div key={`gallery-occ-${i}`} className="px-1.5 py-0.5 border border-[#8a4000]/40 bg-[#3a1500]/20 text-[#ff8800]/80 tracking-widest text-[9px]">
                        GALLERY MASK: {region}
                      </div>
                    ))}
                  </div>
                )}
                {safeCorrespondences.length > 0 && (
                  isDebugMode ? (
                    <div className="flex gap-[2px] flex-wrap">
                      {safeCorrespondences.map((c: Correspondence, i: number) => (
                        <div key={`corr-${i}`} className={`px-1.5 py-0.5 border bg-[#050505] tracking-widest text-[9px] ${results.veto_triggered || results.failed_provenance_veto ? 'border-[#5a0015] text-[#ff2040]/70' : 'border-[#D4AF37]/30 text-[#D4AF37]/80'}`}>
                          MARK {i+1} <span className="opacity-50 mx-0.5">LR:</span>{(typeof c.lr === "number" ? c.lr : 0).toFixed(1)}
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className={`px-1.5 py-0.5 border bg-[#050505] tracking-widest text-[9px] ${results.veto_triggered || results.failed_provenance_veto ? 'border-[#5a0015] text-[#ff2040]/70' : 'border-[#D4AF37]/30 text-[#D4AF37]/80'}`}>
                      MARK CORRESPONDENCES: {safeCorrespondences.length}
                    </div>
                  )
                )}
              </div>
            );
          })()}
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
        <div className="flex-1 min-h-0 flex flex-col gap-2">
          <div className="flex-1 min-h-0 grid grid-cols-2 gap-1">
            {/* Left Pane: Probe */}
            <div
              className={`relative overflow-hidden rounded border ${mode === 'delta' ? 'border-red-900/60' : mode === 'mesh' ? 'border-[#D4AF37]/30' : 'border-[#333]'} bg-[#050505] cursor-move`}
              {...commonPaneEvents}
            >
              <canvas ref={leftCanvasRef} className="block w-full h-full" />
              <div className="absolute top-2 left-3 text-[9px] font-mono text-gray-600 tracking-widest pointer-events-none">{mode === 'debug' ? <span className="text-yellow-500">PROBE MARK DEBUG</span> : 'PROBE (A)'}</div>
            </div>

            {/* Right Pane: Gallery */}
            <div
              className={`relative overflow-hidden rounded border ${mode === 'delta' ? 'border-red-900/60' : mode === 'mesh' ? 'border-[#D4AF37]/30' : 'border-[#333]'} bg-[#050505] cursor-move`}
              {...commonPaneEvents}
            >
              <canvas ref={rightCanvasRef} className="block w-full h-full" />
              <div className="absolute top-2 left-3 text-[9px] font-mono text-gray-600 tracking-widest pointer-events-none">{mode === 'debug' ? <span className="text-yellow-500">GALLERY MARK DEBUG</span> : 'GALLERY (B)'}</div>
            </div>
          </div>

          {mode === 'marks' && (
            /* ── MARKS MODE: Professional Correspondence Evidence Card ── */
            <div className="h-48 flex-shrink-0 flex flex-col overflow-y-auto bg-[#050505] border border-emerald-900/40 rounded p-3 gap-3">
              {/* Status Banner */}
              {(() => {
                const diag = results?.mark_diagnostics;
                if (!diag) {
                  return (
                    <div className="px-3 py-2 border border-red-900/50 rounded bg-red-950/30 font-mono text-xs text-red-400">
                      <div className="font-bold tracking-wider mb-1">Mark evidence diagnostics unavailable.</div>
                      <div className="text-[10px] opacity-70 leading-relaxed">The backend did not return mark_diagnostics. This comparison cannot be audited for mark evidence.</div>
                    </div>
                  );
                }

                const status = diag.mark_match_status || results?.mark_match_status || 'UNKNOWN';
                const probeCount = diag.raw_probe_marks_count;
                const galCount = diag.raw_gallery_marks_count;
                const matchCount = diag.accepted_correspondences_count;
                const lrMarks = diag.lr_marks ?? results?.lr_marks;
                const rejectionSummary = diag.rejection_summary;

                let statusLabel = 'UNKNOWN';
                let statusColor = 'text-gray-400 border-gray-700 bg-[#0a0a0a]';
                if (status === 'EXACT_SELF_MATCH') {
                  statusLabel = 'Exact image self-match. Mark evidence is self-corresponding by identity.';
                  statusColor = 'text-emerald-300 border-emerald-700 bg-emerald-950/40';
                } else if (status === 'MATCHED') {
                  statusLabel = 'Shared facial marks detected';
                  statusColor = 'text-emerald-300 border-emerald-700 bg-emerald-950/40';
                } else if (status === 'INSUFFICIENT_MARKS') {
                  statusLabel = 'Insufficient marks for correspondence';
                  statusColor = 'text-yellow-400 border-yellow-800 bg-yellow-950/30';
                } else if (status === 'NO_MATCHES') {
                  statusLabel = 'No matching marks found';
                  statusColor = 'text-orange-400 border-orange-800 bg-orange-950/30';
                } else if (status === 'FACE_NOT_DETECTED') {
                  statusLabel = 'Face not detected — mark analysis could not proceed';
                  statusColor = 'text-red-400 border-red-800 bg-red-950/30';
                } else if (status === 'DETECTOR_UNAVAILABLE') {
                  statusLabel = 'Mark detector unavailable';
                  statusColor = 'text-red-400 border-red-800 bg-red-950/30';
                }

                return (
                  <>
                    <div className={`px-3 py-2 border rounded font-mono text-xs tracking-wider ${statusColor}`}>
                      <div className="flex justify-between items-center mb-1.5">
                        <span className="font-bold">{statusLabel}</span>
                        <div className="flex items-center gap-2 text-[8px] opacity-70">
                          <span>DETECTOR: {diag.detector_status || 'UNKNOWN'}</span>
                          <span>·</span>
                          <span>MATCHER: {diag.matcher_status || 'UNKNOWN'}</span>
                        </div>
                      </div>
                      <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-[9px] opacity-80 mb-1.5">
                        <div className="flex justify-between"><span>Probe Marks Detected</span><span className="font-bold">{probeCount}</span></div>
                        <div className="flex justify-between"><span>Gallery Marks Detected</span><span className="font-bold">{galCount}</span></div>
                        <div className="flex justify-between"><span>Accepted Correspondences</span><span className="font-bold text-emerald-400">{matchCount}</span></div>
                        <div className="flex justify-between"><span>Rejected Candidates</span><span className="font-bold text-amber-400">{diag.rejected_candidates_count ?? 0}</span></div>
                      </div>
                      <div className="flex justify-between items-center pt-1 border-t border-white/10 text-[10px]">
                        <span className="opacity-60">{matchCount} of {Math.max(probeCount, galCount)} marks matched</span>
                        {lrMarks != null && <span className="font-bold">LR_MARKS: {lrMarks.toFixed(4)}</span>}
                      </div>
                    </div>

                    {/* Diagnostic messaging for empty states */}
                    {rejectionSummary ? (
                      <div className="px-3 py-1.5 border border-amber-900/30 rounded bg-amber-950/20 font-mono text-[9px] text-amber-400/80 leading-relaxed">
                        {rejectionSummary}
                      </div>
                    ) : (
                      <>
                        {probeCount === 0 && galCount === 0 && (
                          <div className="px-3 py-1.5 border border-yellow-900/30 rounded bg-yellow-950/20 font-mono text-[9px] text-yellow-400/70 leading-relaxed">
                            No raw marks detected on either probe or gallery. The detector found no reliable candidates — likely due to lighting conditions, image crop, occlusion, or threshold filters.
                          </div>
                        )}
                        {(probeCount === 0 && galCount > 0) && (
                          <div className="px-3 py-1.5 border border-yellow-900/30 rounded bg-yellow-950/20 font-mono text-[9px] text-yellow-400/70 leading-relaxed">
                            No raw marks detected on probe. The detector rejected all candidates on the probe side due to lighting, crop, occlusion, or threshold filters.
                          </div>
                        )}
                        {(probeCount > 0 && galCount === 0) && (
                          <div className="px-3 py-1.5 border border-yellow-900/30 rounded bg-yellow-950/20 font-mono text-[9px] text-yellow-400/70 leading-relaxed">
                            No raw marks detected on gallery. The detector rejected all candidates on the gallery side due to lighting, crop, occlusion, or threshold filters.
                          </div>
                        )}
                        {probeCount > 0 && galCount > 0 && matchCount === 0 && status !== 'EXACT_SELF_MATCH' && (
                          <div className="px-3 py-1.5 border border-orange-900/30 rounded bg-orange-950/20 font-mono text-[9px] text-orange-400/70 leading-relaxed">
                            Marks were detected, but no accepted correspondences passed the matcher. The detected marks did not meet the spatial or morphological thresholds required for forensic correspondence.
                          </div>
                        )}
                      </>
                    )}

                    {/* Detector/matcher unavailable diagnostics */}
                    {(diag.detector_status === 'FACE_NOT_DETECTED' || diag.detector_status === 'UNKNOWN') && diag.detector_status !== 'OK' && diag.detector_status !== 'NO_CANDIDATES' && (
                      <div className="px-3 py-1.5 border border-red-900/40 rounded bg-red-950/20 font-mono text-[9px] text-red-400/80 leading-relaxed">
                        Mark detector returned status: {diag.detector_status}. Mark evidence cannot be evaluated.
                      </div>
                    )}
                    {(diag.matcher_status === 'UNKNOWN') && (
                      <div className="px-3 py-1.5 border border-red-900/40 rounded bg-red-950/20 font-mono text-[9px] text-red-400/80 leading-relaxed">
                        Mark matcher returned status: {diag.matcher_status}. Correspondence evaluation was not performed.
                      </div>
                    )}

                    {/* All candidates rejected diagnostic */}
                    {(diag.rejected_candidates_count > 0 && matchCount === 0 && probeCount > 0 && galCount > 0 && !rejectionSummary) && (
                      <div className="px-3 py-1.5 border border-amber-900/30 rounded bg-amber-950/20 font-mono text-[9px] text-amber-400/70 leading-relaxed">
                        All {diag.rejected_candidates_count} candidate mark pair(s) were rejected by the matcher. None met the required spatial and morphological thresholds.
                      </div>
                    )}

                    {/* LR = 1.0 or null — always show rejection_summary if available */}
                    {(lrMarks == null || lrMarks === 1.0) && rejectionSummary && (
                      <div className="px-3 py-1.5 border border-amber-900/30 rounded bg-amber-950/20 font-mono text-[9px] text-amber-400/80 leading-relaxed">
                        {rejectionSummary}
                      </div>
                    )}

                    {/* Self-match LR transparency note */}
                    {status === 'EXACT_SELF_MATCH' && (
                      <div className="px-3 py-1.5 border border-emerald-900/30 rounded bg-emerald-950/20 font-mono text-[9px] text-emerald-400/70 leading-relaxed">
                        Mark LR is neutral (1.0) for exact byte-identical image comparisons. The probe and gallery are the same source image, so mark evidence is self-corresponding by identity rather than independent forensic evidence.
                      </div>
                    )}

                    {/* Correspondence Evidence Cards */}
                    {(() => {
                      const safeCorrespondences = Array.isArray(results?.correspondences) ? results.correspondences : [];
                      if (safeCorrespondences.length === 0) {
                        return (
                          <div className="flex-1 flex items-center justify-center text-gray-500 font-mono text-xs tracking-widest">
                            NO ACCEPTED CORRESPONDENCES
                          </div>
                        );
                      }
                      return (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                          {safeCorrespondences.map((c: Correspondence, i: number) => {
                            const lr = typeof c.lr === 'number' ? c.lr : 0;
                            const lrColor = lr >= 10 ? 'text-emerald-300' : lr >= 1 ? 'text-yellow-300' : 'text-red-300';
                            return (
                              <div key={`mark-card-${i}`} className="border border-emerald-900/40 bg-[#0a0f0a] rounded p-2 font-mono text-[10px]">
                                <div className="flex justify-between items-center mb-1">
                                  <span className="text-emerald-400 font-bold tracking-wider">
                                    <span className="inline-flex items-center justify-center w-5 h-5 rounded-full bg-emerald-800/60 text-emerald-200 text-[9px] mr-1.5">{i + 1}</span>
                                    MARK {i + 1}
                                  </span>
                                  <span className={`font-bold ${lrColor}`}>LR: {lr.toFixed(2)}</span>
                                </div>
                                <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-gray-400">
                                  <div>TYPE: <span className="text-gray-200">{c.mark_type ?? '—'}</span></div>
                                  <div>REGION: <span className="text-gray-200">{c.face_region ?? '—'}</span></div>
                                  <div>P [{c.probe_idx ?? '?'}]: <span className="text-gray-300">{c.probe_centroid ? `(${c.probe_centroid[0]?.toFixed(3)}, ${c.probe_centroid[1]?.toFixed(3)})` : '—'}</span></div>
                                  <div>G [{c.gallery_idx ?? '?'}]: <span className="text-gray-300">{c.gallery_centroid ? `(${c.gallery_centroid[0]?.toFixed(3)}, ${c.gallery_centroid[1]?.toFixed(3)})` : '—'}</span></div>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      );
                    })()}

                    {/* Version Footer */}
                    <div className="text-[8px] font-mono text-gray-600 tracking-wider flex justify-between">
                      <span>DETECTOR: v{results?.mark_detector_version ?? '?'} · MATCHER: v{results?.mark_matcher_version ?? '?'}</span>
                      <span>MARK LRs: [{(results?.mark_lrs ?? []).map((lr: number) => lr?.toFixed(2) ?? '?').join(', ')}]</span>
                    </div>
                  </>
                );
              })()}
            </div>
          )}

          {mode === 'delta' && (
            /* ── PIXEL DELTA: Standalone Diagnostic Panel ── */
            <div className="shrink-0 flex flex-col gap-2 mt-2">
              <div className="relative overflow-hidden rounded border border-red-900/40 bg-[#050505] flex items-center justify-center" style={{ height: '180px' }}>
                {deltaImg ? (
                  <canvas
                    ref={(el) => {
                      if (!el || !deltaImg) return;
                      const ctx = el.getContext('2d');
                      if (!ctx) return;
                      const cw = el.clientWidth;
                      const ch = el.clientHeight;
                      el.width = cw;
                      el.height = ch;
                      const iw = deltaImg.width;
                      const ih = deltaImg.height;
                      const scale = Math.min(cw / iw, ch / ih, 1);
                      const dw = iw * scale;
                      const dh = ih * scale;
                      const ox = (cw - dw) / 2;
                      const oy = (ch - dh) / 2;
                      ctx.clearRect(0, 0, cw, ch);
                      ctx.drawImage(deltaImg, ox, oy, dw, dh);
                    }}
                    className="block w-full h-full"
                  />
                ) : (
                  <span className="text-gray-600 font-mono text-[10px] tracking-widest">NO DELTA IMAGE AVAILABLE</span>
                )}
                <div className="absolute top-2 left-3 text-[9px] font-mono text-red-400/70 tracking-widest pointer-events-none">PIXEL DIFFERENCE MAP</div>
              </div>
              <div className="text-[9px] font-mono text-red-300/70 tracking-wider border border-red-900/40 bg-red-950/10 rounded px-3 py-2 leading-relaxed" data-testid="pixel-delta-disclaimer">
                This is not scar, mole, blemish, or mark correspondence evidence. Pixel Delta shows structural pixel/edge differences between aligned crops only. It does not detect, match, or score facial marks and must not be interpreted as forensic mark evidence.
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Debug Forensic Panel ── */}
      {(() => {
        if (!(showForensicDebugDetails && results?.mark_debug)) return null;
        
        const debugProbeMarks: MarkDescriptor[] =
          Array.isArray(results.mark_debug.probe_marks_first_20)
            ? results.mark_debug.probe_marks_first_20
            : [];

        const debugGalleryMarks: MarkDescriptor[] =
          Array.isArray(results.mark_debug.gallery_marks_first_20)
            ? results.mark_debug.gallery_marks_first_20
            : [];

        const debugCorrespondences: MarkDebugCorrespondence[] =
          Array.isArray(results.mark_debug.correspondences_first_20)
            ? results.mark_debug.correspondences_first_20
            : (Array.isArray(results.mark_debug.correspondences) ? results.mark_debug.correspondences as MarkDebugCorrespondence[] : []);

        const unmatchedProbeIndices: number[] =
          Array.isArray(results.mark_debug.unmatched_probe_indices)
            ? results.mark_debug.unmatched_probe_indices.filter((n): n is number => typeof n === "number")
            : (Array.isArray(results.mark_debug.probe_unmatched_indices) ? results.mark_debug.probe_unmatched_indices.filter((n): n is number => typeof n === "number") : []);

        const unmatchedGalleryIndices: number[] =
          Array.isArray(results.mark_debug.unmatched_gallery_indices)
            ? results.mark_debug.unmatched_gallery_indices.filter((n): n is number => typeof n === "number")
            : (Array.isArray(results.mark_debug.gallery_unmatched_indices) ? results.mark_debug.gallery_unmatched_indices.filter((n): n is number => typeof n === "number") : []);

        return (
          <div className="shrink-0 mt-1 p-2 border border-yellow-800/50 bg-[#0e0e00] text-[8px] font-mono text-yellow-500/80 overflow-auto max-h-40">
            <div className="font-bold tracking-widest mb-1 text-yellow-400">DEBUG FORENSIC — BAYESIAN HUNGARIAN MATCHER ({results.mark_debug.version ?? results.mark_debug.matcher_version ?? 'N/A'})</div>
            <div className="grid grid-cols-3 gap-2">
              <div>
                <div className="text-yellow-600 mb-0.5">METRICS</div>
                <div>PROBE MARKS: {results.mark_debug.probe_marks_count ?? 0}</div>
                <div>GALLERY MARKS: {results.mark_debug.gallery_marks_count ?? 0}</div>
                <div>CORRESPONDENCES: {results.mark_debug.correspondences_count ?? 0}</div>
              </div>
              <div>
                <div className="text-yellow-600 mb-0.5">UNMATCHED INDICES</div>
                <div>PROBE: {JSON.stringify(unmatchedProbeIndices)}</div>
                <div>GALLERY: {JSON.stringify(unmatchedGalleryIndices)}</div>
              </div>
              <div>
                <div className="text-yellow-600 mb-0.5">TOP CORRESPONDENCES (LR)</div>
                {debugCorrespondences.slice(0, 3).map((c: MarkDebugCorrespondence, i: number) => (
                  <div key={i}>P[{c.probe_idx}] ↔ G[{c.gallery_idx}] (LR: {typeof c.lr === 'number' ? c.lr.toFixed(2) : 'N/A'})</div>
                ))}
              </div>
            </div>
          </div>
        );
      })()}
      {showForensicDebugDetails && results && !results.mark_debug ? (
        <div className="shrink-0 mt-1 p-2 border border-yellow-800/50 bg-[#0e0e00] text-[8px] font-mono text-yellow-500/80 overflow-auto max-h-40">
          <div className="font-bold tracking-widest mb-1 text-yellow-400">DEBUG FORENSIC — DELTA CORRESPONDENCE INTEGRITY</div>
          <div>raw_probe_marks: {probeMarksSource.length}</div>
          <div>raw_gallery_marks: {galleryMarksSource.length}</div>
          <div>correspondences: {markMatchData.corrs.length}</div>
          <div>matchedProbeIdx: [{[...markMatchData.matchedProbeIndices].join(', ')}]</div>
          <div>matchedGalleryIdx: [{[...markMatchData.matchedGalleryIndices].join(', ')}]</div>
          <div>hasIndexCorrespondences: {String(markMatchData.hasIndexCorrespondences)}</div>
          <div className={JSON.stringify(probeMarksSource) === JSON.stringify(galleryMarksSource) ? 'text-red-400 font-bold' : 'text-green-400'}>
            IDENTICAL ARRAYS: {JSON.stringify(probeMarksSource) === JSON.stringify(galleryMarksSource) ? '⚠ YES — BUG' : '✓ NO'}
          </div>
          <div className="mt-1 border-t border-yellow-800/30 pt-1">
            <div>First 5 probe pts: {JSON.stringify(probeMarksSource.slice(0, 5).map(p => {
              const c = (() => { if (!p) return null; if ('centroid' in p && p.centroid) return { x: p.centroid[0], y: p.centroid[1] }; if (Array.isArray(p)) return { x: p[0], y: p[1] }; if ('x' in p) return { x: p.x, y: p.y }; return null; })();
              return c;
            }))}</div>
            <div>First 5 gallery pts: {JSON.stringify(galleryMarksSource.slice(0, 5).map(p => {
              const c = (() => { if (!p) return null; if ('centroid' in p && p.centroid) return { x: p.centroid[0], y: p.centroid[1] }; if (Array.isArray(p)) return { x: p[0], y: p[1] }; if ('x' in p) return { x: p.x, y: p.y }; return null; })();
              return c;
            }))}</div>
            <div>First 5 corrs: {JSON.stringify(markMatchData.corrs.slice(0, 5).map(c => ({
              probe_idx: c.probe_idx, gallery_idx: c.gallery_idx, lr: typeof c.lr === 'number' ? c.lr.toFixed(2) : 'N/A'
            })))}</div>
          </div>
        </div>
      ) : null}

      {/* ── Bayesian Scoring Trace (DEBUG_FORENSIC only) ── */}
      {showForensicDebugDetails && results?.scoring_trace ? (() => {
        const st: ScoringTrace = results.scoring_trace;
        const vetoColor = st.veto_override_applied ? 'text-yellow-400' : (st.veto_triggered ? 'text-red-400' : 'text-green-400');
        const vetoLabel = st.veto_override_applied
          ? 'OVERRIDE'
          : (st.veto_triggered ? 'VETOED' : 'CLEAR');
        return (
          <div className="shrink-0 mt-1 p-2 border border-cyan-800/50 bg-[#001010] text-[8px] font-mono text-cyan-400/90 overflow-auto max-h-48">
            <div className="font-bold tracking-widest mb-1 text-cyan-300">BAYESIAN SCORING TRACE</div>
            <div className="grid grid-cols-2 gap-x-4 gap-y-0.5">
              <div>CALIBRATION: <span className={st.calibration_status === 'LOADED' ? 'text-green-400' : 'text-red-400'}>{st.calibration_status ?? 'N/A'}</span></div>
              <div>BENCHMARK: {st.calibration_benchmark ?? 'N/A'}</div>
              <div>LR_ENSEMBLE: {st.lr_ensemble_display ?? 'N/A'}</div>
              <div>LR_MARKS: {st.lr_marks_display ?? 'N/A'}</div>
              <div>LR_TOTAL: {st.lr_total_display ?? 'N/A'}</div>
              <div>POSTERIOR: {st.posterior_raw != null ? (st.posterior_raw * 100).toFixed(4) + '%' : 'N/A'}</div>
              <div>BAYESIAN POSTERIOR PRE-VETO: {st.fused_score_pre_veto?.toFixed(2) ?? 'N/A'}</div>
              <div>DISPLAYED DECISION SCORE: {st.fused_score_post_veto?.toFixed(2) ?? 'N/A'}</div>
              <div>VETO: <span className={vetoColor}>{vetoLabel}</span></div>
              <div>REASON: {st.veto_reason ?? 'NONE'}</div>
              <div>MARK OVERRIDE: <span className={st.mark_override_eligible ? 'text-yellow-400' : 'text-gray-500'}>{st.mark_override_eligible ? 'ELIGIBLE' : 'N/A'}</span></div>
              <div>TEMPORAL Δ: {st.temporal_delta_years != null ? st.temporal_delta_years.toFixed(1) + 'yr' : 'N/A'}</div>
              <div>TIER4 CAL: <span className={st.tier4_calibration_status === 'LOADED' ? 'text-green-400' : 'text-red-400'}>{st.tier4_calibration_status ?? 'N/A'}</span></div>
              <div>THRESHOLDS: {st.ensemble_thresholds_key ?? 'N/A'}</div>
            </div>
            {st.veto_override_reason ? (
              <div className="mt-1 border-t border-cyan-800/30 pt-1 text-yellow-400">
                OVERRIDE: {st.veto_override_reason}
              </div>
            ) : null}
          </div>
        );
      })() : null}

      {/* ── Calibration Status Badge ── */}
      {showForensicDebugDetails && results?.calibration_status ? (
        <div className={`shrink-0 mt-0.5 px-2 py-0.5 text-[8px] font-mono tracking-widest ${
          results.calibration_status === 'LOADED'
            ? 'bg-green-900/30 text-green-400 border border-green-800/40'
            : 'bg-red-900/30 text-red-400 border border-red-800/40'
        }`}>
          CALIBRATION: {results.calibration_status}
          {results.veto_override_applied ? ' · MARK OVERRIDE ACTIVE' : ''}
        </div>
      ) : null}

      {/* ── Footer ── */}
      <div className="flex w-full justify-between pt-1 text-[9px] font-mono text-gray-600 tracking-widest shrink-0">
        <span>
          {mode === 'aligned' && 'CANONICAL ALIGNMENT'}
          {mode === 'mesh' && <span className="text-[#D4AF37]">3DMM WIREFRAME HUD</span>}
          {mode === 'delta' && <span className="text-red-500">PIXEL DELTA — STRUCTURAL DIFFERENCE ONLY</span>}
          {mode === 'overlap' && <span className="text-[#D4AF37]">DRAG TO COMPARE OVERLAP</span>}
          {mode === 'marks' && <span className="text-emerald-400">MARK DETECTIONS — GREEN = MATCHED · CYAN = UNMATCHED RAW · GRAY = UNKNOWN</span>}
          {mode === 'debug' && <span className="text-yellow-500">GREEN = ACCEPTED MATCH · CYAN = UNMATCHED DETECTED MARK · GRAY = UNKNOWN</span>}
          {isXrayMode && <span className="ml-2 text-[#D4AF37] animate-pulse">· X-RAY ACTIVE</span>}
        </span>
        <span>SYNCHRONIZED · {Math.round(zoom * 100)}%</span>
      </div>
    </div>
  );
}
