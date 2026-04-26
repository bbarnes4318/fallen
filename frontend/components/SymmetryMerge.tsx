'use client';

import React, { useRef, useState, useEffect } from 'react';

interface SymmetryMergeProps {
  galleryImageSrc: string;
  probeImageSrc: string;
  deltaImageSrc?: string;
  galleryWireframeSrc?: string;
  probeWireframeSrc?: string;
}

export default function SymmetryMerge({ galleryImageSrc, probeImageSrc, deltaImageSrc, galleryWireframeSrc, probeWireframeSrc }: SymmetryMergeProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  // State
  const [sliderPos, setSliderPos] = useState(50);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [mode, setMode] = useState<'split' | 'pan' | 'delta' | 'wireframe'>('split');
  
  const [isDragging, setIsDragging] = useState(false);
  const lastMousePos = useRef({ x: 0, y: 0 });
  const [loadedSources, setLoadedSources] = useState({ gallery: '', probe: '' });
  const [deltaLoaded, setDeltaLoaded] = useState(false);
  const [wireframeLoaded, setWireframeLoaded] = useState(false);
  
  const galleryImgRef = useRef<HTMLImageElement | null>(null);
  const probeImgRef = useRef<HTMLImageElement | null>(null);
  const deltaImgRef = useRef<HTMLImageElement | null>(null);
  const gWireRef = useRef<HTMLImageElement | null>(null);
  const pWireRef = useRef<HTMLImageElement | null>(null);

  const imagesLoaded = loadedSources.gallery === galleryImageSrc && loadedSources.probe === probeImageSrc;

  // 1. Load Images
  useEffect(() => {
    let cancelled = false;
    let loadedCount = 0;
    const targetGallery = galleryImageSrc;
    const targetProbe = probeImageSrc;

    const onLoad = () => {
      loadedCount++;
      if (loadedCount === 2 && !cancelled) {
        setLoadedSources({ gallery: targetGallery, probe: targetProbe });
        setZoom(1); // Reset view on new images
        setPan({ x: 0, y: 0 });
      }
    };

    const gImg = new Image();
    gImg.crossOrigin = 'anonymous';
    gImg.src = galleryImageSrc;
    gImg.onload = onLoad;
    galleryImgRef.current = gImg;

    const pImg = new Image();
    pImg.crossOrigin = 'anonymous';
    pImg.src = probeImageSrc;
    pImg.onload = onLoad;
    probeImgRef.current = pImg;

    // Load delta image if available
    if (deltaImageSrc) {
      const dImg = new Image();
      dImg.crossOrigin = 'anonymous';
      dImg.src = deltaImageSrc;
      dImg.onload = () => { if (!cancelled) setDeltaLoaded(true); };
      deltaImgRef.current = dImg;
    } else {
      deltaImgRef.current = null;
      queueMicrotask(() => setDeltaLoaded(false));
    }

    // Load wireframe images if available
    if (galleryWireframeSrc && probeWireframeSrc) {
      let wireCount = 0;
      const onWireLoad = () => {
        wireCount++;
        if (wireCount === 2 && !cancelled) setWireframeLoaded(true);
      };

      const gwImg = new Image();
      gwImg.crossOrigin = 'anonymous';
      gwImg.src = galleryWireframeSrc;
      gwImg.onload = onWireLoad;
      gWireRef.current = gwImg;

      const pwImg = new Image();
      pwImg.crossOrigin = 'anonymous';
      pwImg.src = probeWireframeSrc;
      pwImg.onload = onWireLoad;
      pWireRef.current = pwImg;
    } else {
      gWireRef.current = null;
      pWireRef.current = null;
      queueMicrotask(() => setWireframeLoaded(false));
    }

    return () => { cancelled = true; };
  }, [galleryImageSrc, probeImageSrc, deltaImageSrc, galleryWireframeSrc, probeWireframeSrc]);

  // 2. Draw Canvas (Hardware Accelerated)
  useEffect(() => {
    if (!imagesLoaded || !canvasRef.current || !galleryImgRef.current || !probeImgRef.current || !containerRef.current) return;
    if (mode === 'delta' && (!deltaImgRef.current || !deltaLoaded)) return;
    if (mode === 'wireframe' && (!gWireRef.current || !pWireRef.current || !wireframeLoaded)) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const containerWidth = containerRef.current.clientWidth;
    const containerHeight = containerRef.current.clientHeight;

    // Set canvas to physical screen pixels
    canvas.width = containerWidth;
    canvas.height = containerHeight;

    const imgWidth = galleryImgRef.current.width;
    const imgHeight = galleryImgRef.current.height;
    
    // Base scale to fit container natively
    const baseScale = Math.min(containerWidth / imgWidth, containerHeight / imgHeight, 1);
    const currentScale = baseScale * zoom;

    // Calculate drawing center
    const drawWidth = imgWidth * currentScale;
    const drawHeight = imgHeight * currentScale;
    const offsetX = (containerWidth - drawWidth) / 2 + pan.x;
    const offsetY = (containerHeight - drawHeight) / 2 + pan.y;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Save state & apply transformations
    ctx.save();
    ctx.translate(offsetX, offsetY);
    ctx.scale(currentScale, currentScale);

    if (mode === 'delta' && deltaImgRef.current) {
      // ── DELTA MODE: Draw the scar map overlay, perfectly scaled ──
      ctx.drawImage(deltaImgRef.current, 0, 0, imgWidth, imgHeight);

      // Crimson vignette border
      ctx.strokeStyle = 'rgba(180, 0, 30, 0.5)';
      ctx.lineWidth = 3 / currentScale;
      ctx.strokeRect(0, 0, imgWidth, imgHeight);
    } else if (mode === 'wireframe' && gWireRef.current && pWireRef.current) {
      // ── WIREFRAME MODE: Split-slider over gold mesh HUD images ──
      const splitX = (sliderPos / 100) * imgWidth;

      // Draw Gallery wireframe (Left)
      if (splitX > 0) {
        ctx.drawImage(gWireRef.current, 0, 0, splitX, imgHeight, 0, 0, splitX, imgHeight);
      }

      // Draw Probe wireframe (Right)
      if (splitX < imgWidth) {
        ctx.drawImage(pWireRef.current, splitX, 0, imgWidth - splitX, imgHeight, splitX, 0, imgWidth - splitX, imgHeight);
      }

      // Draw Gold Guideline
      ctx.beginPath();
      ctx.moveTo(splitX, 0);
      ctx.lineTo(splitX, imgHeight);
      ctx.strokeStyle = '#D4AF37';
      ctx.lineWidth = 2 / currentScale;
      ctx.shadowBlur = 10 / currentScale;
      ctx.shadowColor = 'rgba(212, 175, 55, 0.8)';
      ctx.stroke();
    } else {
      // ── SPLIT / PAN MODE ──
      const splitX = (sliderPos / 100) * imgWidth;

      // Draw Gallery (Left)
      if (splitX > 0) {
        ctx.drawImage(galleryImgRef.current, 0, 0, splitX, imgHeight, 0, 0, splitX, imgHeight);
      }

      // Draw Probe (Right)
      if (splitX < imgWidth) {
        ctx.drawImage(probeImgRef.current, splitX, 0, imgWidth - splitX, imgHeight, splitX, 0, imgWidth - splitX, imgHeight);
      }

      // Draw Gold Guideline
      ctx.beginPath();
      ctx.moveTo(splitX, 0);
      ctx.lineTo(splitX, imgHeight);
      ctx.strokeStyle = '#D4AF37';
      ctx.lineWidth = 2 / currentScale;
      ctx.shadowBlur = 10 / currentScale;
      ctx.shadowColor = 'rgba(212, 175, 55, 0.8)';
      ctx.stroke();
    }
    
    ctx.restore();
  }, [sliderPos, imagesLoaded, zoom, pan, mode, deltaLoaded, wireframeLoaded]);

  // 3. Interaction Handlers
  const handlePointerDown = (clientX: number, clientY: number) => {
    setIsDragging(true);
    lastMousePos.current = { x: clientX, y: clientY };
  };

  const handlePointerMove = (clientX: number, clientY: number) => {
    if (!isDragging || !containerRef.current) return;
    
    if (mode === 'split' || mode === 'wireframe') {
      const rect = containerRef.current.getBoundingClientRect();
      const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
      setSliderPos((x / rect.width) * 100);
    } else if (mode === 'pan' || mode === 'delta') {
      const dx = clientX - lastMousePos.current.x;
      const dy = clientY - lastMousePos.current.y;
      setPan(prev => ({ x: prev.x + dx, y: prev.y + dy }));
      lastMousePos.current = { x: clientX, y: clientY };
    }
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const zoomSensitivity = 0.05;
    const delta = e.deltaY < 0 ? 1 + zoomSensitivity : 1 - zoomSensitivity;
    setZoom(prev => Math.min(Math.max(prev * delta, 1), 8)); // Clamp zoom between 1x and 8x
  };

  return (
    <div className="flex flex-col h-full w-full min-h-0 relative">
      {/* ── Toolbar ── */}
      <div className="flex justify-between items-center px-1 pb-1.5 shrink-0 select-none">
        <div>
          <h2 className="text-sm font-bold text-gray-100 font-mono tracking-wider leading-tight">FORENSIC INSPECTOR</h2>
          <p className="text-[10px] text-gray-500 font-mono">Zoom & align</p>
        </div>

        {/* Zoom & Mode Controls */}
        <div className="flex gap-2">
          <div className="flex border border-[#333] rounded bg-[#111] overflow-hidden text-[10px] font-mono">
            <button 
              onClick={() => setMode('split')} 
              className={`px-3 py-1 transition-colors ${mode === 'split' ? 'bg-[#D4AF37] text-black font-bold' : 'text-gray-400 hover:text-white'}`}
            >
              SPLIT
            </button>
            <button 
              onClick={() => setMode('pan')} 
              className={`px-3 py-1 transition-colors border-l border-[#333] ${mode === 'pan' ? 'bg-[#D4AF37] text-black font-bold' : 'text-gray-400 hover:text-white'}`}
            >
              PAN
            </button>
            {deltaImageSrc && (
              <button 
                onClick={() => setMode('delta')} 
                className={`px-3 py-1 transition-colors border-l ${
                  mode === 'delta' 
                    ? 'bg-[#1a0005] text-[#ff2040] font-bold border-[#5a0015] shadow-[inset_0_0_12px_rgba(180,0,30,0.3)]' 
                    : 'text-gray-400 hover:text-red-300 border-[#333]'
                }`}
              >
                DELTA
              </button>
            )}
            {galleryWireframeSrc && probeWireframeSrc && (
              <button 
                onClick={() => setMode('wireframe')} 
                className={`px-3 py-1 transition-colors border-l ${
                  mode === 'wireframe' 
                    ? 'bg-[#D4AF37] text-black font-bold border-[#D4AF37]' 
                    : 'text-gray-400 hover:text-[#D4AF37] border-[#333]'
                }`}
              >
                MESH
              </button>
            )}
          </div>

          <div className="flex border border-[#333] rounded bg-[#111] overflow-hidden text-gray-400">
            <button onClick={() => setZoom(z => Math.max(1, z - 0.2))} className="px-2 hover:bg-[#222] hover:text-white transition-colors">-</button>
            <div className="px-2 py-1 text-[10px] border-x border-[#333] font-mono min-w-[45px] text-center">
              {Math.round(zoom * 100)}%
            </div>
            <button onClick={() => setZoom(z => Math.min(8, z + 0.2))} className="px-2 hover:bg-[#222] hover:text-white transition-colors">+</button>
            <button onClick={() => { setZoom(1); setPan({x:0, y:0}); }} className="px-2 py-1 text-[10px] border-l border-[#333] hover:bg-[#222] hover:text-[#D4AF37] transition-colors">
              RESET
            </button>
          </div>
        </div>
      </div>

      {/* ── Canvas Viewport ── */}
      <div 
        ref={containerRef}
        className={`relative flex-1 min-h-0 overflow-hidden rounded border ${mode === 'delta' ? 'border-red-900/60' : mode === 'wireframe' ? 'border-[#D4AF37]/30' : 'border-[#333]'} bg-[#050505] ${mode === 'split' || mode === 'wireframe' ? 'cursor-col-resize' : 'cursor-move'}`}
        onMouseDown={(e) => handlePointerDown(e.clientX, e.clientY)}
        onMouseUp={() => setIsDragging(false)}
        onMouseLeave={() => setIsDragging(false)}
        onMouseMove={(e) => handlePointerMove(e.clientX, e.clientY)}
        onWheel={handleWheel}
        // Touch support
        onTouchStart={(e) => handlePointerDown(e.touches[0].clientX, e.touches[0].clientY)}
        onTouchEnd={() => setIsDragging(false)}
        onTouchMove={(e) => handlePointerMove(e.touches[0].clientX, e.touches[0].clientY)}
      >
        {!imagesLoaded ? (
          <div className="flex items-center justify-center w-full h-full bg-[#0a0a0a] animate-pulse text-[#D4AF37] font-mono text-xs tracking-widest">
            LOADING BIO-DATA...
          </div>
        ) : (
          <canvas ref={canvasRef} className="block w-full h-full" />
        )}
      </div>

      {/* ── Footer ── */}
      <div className="flex w-full justify-between pt-1 text-[9px] font-mono text-gray-600 tracking-widest shrink-0">
        <span>GALLERY (A)</span>
        {mode === 'pan' && <span className="text-[#D4AF37]">DRAG TO PAN IMAGE</span>}
        {mode === 'delta' && <span className="text-red-500">BIOLOGICAL TOPOGRAPHY DELTA</span>}
        {mode === 'wireframe' && <span className="text-[#D4AF37]">3DMM WIREFRAME HUD</span>}
        <span>PROBE (B)</span>
      </div>
    </div>
  );
}
