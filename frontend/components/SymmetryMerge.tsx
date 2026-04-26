'use client';

import React, { useRef, useState, useEffect } from 'react';

interface SymmetryMergeProps {
  galleryImageSrc: string;
  probeImageSrc: string;
}

export default function SymmetryMerge({ galleryImageSrc, probeImageSrc }: SymmetryMergeProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [sliderPos, setSliderPos] = useState(50);
  const [isDragging, setIsDragging] = useState(false);
  const [loadedSources, setLoadedSources] = useState({ gallery: '', probe: '' });
  
  const galleryImgRef = useRef<HTMLImageElement | null>(null);
  const probeImgRef = useRef<HTMLImageElement | null>(null);

  const imagesLoaded = loadedSources.gallery === galleryImageSrc && loadedSources.probe === probeImageSrc;

  // Load Images
  useEffect(() => {
    let cancelled = false;
    let loadedCount = 0;
    const targetGallery = galleryImageSrc;
    const targetProbe = probeImageSrc;

    const onLoad = () => {
      loadedCount++;
      if (loadedCount === 2 && !cancelled) {
        setLoadedSources({ gallery: targetGallery, probe: targetProbe });
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

    return () => { cancelled = true; };
  }, [galleryImageSrc, probeImageSrc]);

  // Draw Canvas — fits parent container height
  useEffect(() => {
    if (!imagesLoaded || !canvasRef.current || !galleryImgRef.current || !probeImgRef.current || !containerRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = galleryImgRef.current.width;
    const height = galleryImgRef.current.height;
    
    // Scale to fit the container (both width and height constrained)
    const containerWidth = containerRef.current.clientWidth;
    const containerHeight = containerRef.current.clientHeight;
    const scale = Math.min(containerWidth / width, containerHeight / height, 1);
    
    const scaledWidth = width * scale;
    const scaledHeight = height * scale;

    canvas.width = scaledWidth;
    canvas.height = scaledHeight;

    const splitX = (sliderPos / 100) * scaledWidth;

    ctx.clearRect(0, 0, scaledWidth, scaledHeight);

    // Draw Left Half (Gallery)
    if (splitX > 0) {
      ctx.drawImage(
        galleryImgRef.current,
        0, 0, (sliderPos / 100) * width, height,
        0, 0, splitX, scaledHeight
      );
    }

    // Draw Right Half (Probe)
    if (splitX < scaledWidth) {
      ctx.drawImage(
        probeImgRef.current,
        (sliderPos / 100) * width, 0, width - ((sliderPos / 100) * width), height,
        splitX, 0, scaledWidth - splitX, scaledHeight
      );
    }

    // Draw Gold Guideline
    ctx.beginPath();
    ctx.moveTo(splitX, 0);
    ctx.lineTo(splitX, scaledHeight);
    ctx.strokeStyle = '#D4AF37';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.shadowBlur = 10;
    ctx.shadowColor = 'rgba(212, 175, 55, 0.5)';
    ctx.stroke();
    ctx.shadowBlur = 0;

  }, [sliderPos, imagesLoaded]);

  const handleMove = (clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
    setSliderPos((x / rect.width) * 100);
  };

  const onMouseMove = (e: React.MouseEvent) => {
    if (isDragging) handleMove(e.clientX);
  };

  const onTouchMove = (e: React.TouchEvent) => {
    if (isDragging) handleMove(e.touches[0].clientX);
  };

  return (
    <div className="flex flex-col h-full w-full min-h-0">
      {/* Header row */}
      <div className="flex justify-between items-center px-1 pb-1.5 shrink-0">
        <div>
          <h2 className="text-sm font-bold text-gray-100 font-mono tracking-wider leading-tight">SYMMETRY MERGE</h2>
          <p className="text-[10px] text-gray-500 font-mono">Structural alignment</p>
        </div>
        <div className="text-[#D4AF37] font-mono text-[10px] px-2 py-0.5 bg-[#1a1a1a] rounded border border-[#D4AF37]/30">
          {Math.round(sliderPos)}%
        </div>
      </div>

      {/* Canvas area — fills remaining height */}
      <div 
        ref={containerRef}
        className="relative flex-1 min-h-0 overflow-hidden rounded border border-[#333] cursor-col-resize flex items-center justify-center bg-[#080808]"
        onMouseDown={() => setIsDragging(true)}
        onMouseUp={() => setIsDragging(false)}
        onMouseLeave={() => setIsDragging(false)}
        onMouseMove={onMouseMove}
        onTouchStart={() => setIsDragging(true)}
        onTouchEnd={() => setIsDragging(false)}
        onTouchMove={onTouchMove}
      >
        {!imagesLoaded ? (
          <div className="flex items-center justify-center w-full h-full bg-[#111] animate-pulse text-[#D4AF37] font-mono text-xs">
            LOADING…
          </div>
        ) : (
          <>
            <canvas ref={canvasRef} className="block max-w-full max-h-full pointer-events-none" />
            
            {/* Slider Handle */}
            <div 
              className="absolute top-0 bottom-0 w-1 pointer-events-none"
              style={{ left: `${sliderPos}%`, transform: 'translateX(-50%)' }}
            >
              <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-6 h-6 rounded-full bg-[#0a0a0a] border-2 border-[#D4AF37] shadow-[0_0_10px_rgba(212,175,55,0.4)] flex items-center justify-center">
                <div className="flex space-x-0.5">
                  <div className="w-0.5 h-2 bg-[#D4AF37]"></div>
                  <div className="w-0.5 h-2 bg-[#D4AF37]"></div>
                </div>
              </div>
            </div>
          </>
        )}
      </div>

      {/* Footer labels */}
      <div className="flex w-full justify-between pt-1 text-[9px] font-mono text-gray-600 tracking-widest shrink-0">
        <span>GALLERY (A)</span>
        <span>PROBE (B)</span>
      </div>
    </div>
  );
}
