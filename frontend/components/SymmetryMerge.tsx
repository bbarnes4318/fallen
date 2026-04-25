'use client';

import React, { useRef, useState, useEffect } from 'react';

interface SymmetryMergeProps {
  galleryImageSrc: string;
  probeImageSrc: string;
}

export default function SymmetryMerge({ galleryImageSrc, probeImageSrc }: SymmetryMergeProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [sliderPos, setSliderPos] = useState(50); // 0 to 100 percentage
  const [isDragging, setIsDragging] = useState(false);
  const [imagesLoaded, setImagesLoaded] = useState(false);
  
  const galleryImgRef = useRef<HTMLImageElement | null>(null);
  const probeImgRef = useRef<HTMLImageElement | null>(null);

  // Load Images
  useEffect(() => {
    let loadedCount = 0;
    const onLoad = () => {
      loadedCount++;
      if (loadedCount === 2) {
        setImagesLoaded(true);
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
  }, [galleryImageSrc, probeImageSrc]);

  // Draw Canvas
  useEffect(() => {
    if (!imagesLoaded || !canvasRef.current || !galleryImgRef.current || !probeImgRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Use dimensions of gallery image as base (assuming they are pre-aligned/same size)
    const width = galleryImgRef.current.width;
    const height = galleryImgRef.current.height;
    
    // Scale canvas down if it's too big, maintaining aspect ratio
    const maxWidth = containerRef.current?.clientWidth || 800;
    const scale = Math.min(1, maxWidth / width);
    
    const scaledWidth = width * scale;
    const scaledHeight = height * scale;

    canvas.width = scaledWidth;
    canvas.height = scaledHeight;

    const splitX = (sliderPos / 100) * scaledWidth;

    // Clear canvas
    ctx.clearRect(0, 0, scaledWidth, scaledHeight);

    // Draw Left Half (Gallery)
    if (splitX > 0) {
      ctx.drawImage(
        galleryImgRef.current,
        0, 0, (sliderPos / 100) * width, height, // Source rect
        0, 0, splitX, scaledHeight // Dest rect
      );
    }

    // Draw Right Half (Probe)
    if (splitX < scaledWidth) {
      ctx.drawImage(
        probeImgRef.current,
        (sliderPos / 100) * width, 0, width - ((sliderPos / 100) * width), height, // Source
        splitX, 0, scaledWidth - splitX, scaledHeight // Dest
      );
    }

    // Draw Gold Guideline Overlay
    ctx.beginPath();
    ctx.moveTo(splitX, 0);
    ctx.lineTo(splitX, scaledHeight);
    ctx.strokeStyle = '#D4AF37'; // Subtle Gold
    ctx.lineWidth = 2;
    ctx.stroke();

    // Subtle glow effect on the line
    ctx.shadowBlur = 10;
    ctx.shadowColor = 'rgba(212, 175, 55, 0.5)';
    ctx.stroke();
    ctx.shadowBlur = 0; // reset

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
    <div className="flex flex-col items-center w-full max-w-4xl mx-auto p-6 bg-[#0a0a0a] border border-[#1f1f1f] rounded-xl shadow-2xl">
      <div className="flex justify-between w-full mb-4 items-end">
        <div>
          <h2 className="text-xl font-bold text-gray-100 font-mono tracking-wider">SYMMETRY MERGE</h2>
          <p className="text-sm text-gray-400 font-mono">Structural alignment analysis</p>
        </div>
        <div className="text-[#D4AF37] font-mono text-sm px-3 py-1 bg-[#1a1a1a] rounded border border-[#D4AF37]/30">
          SPLIT: {Math.round(sliderPos)}%
        </div>
      </div>

      <div 
        ref={containerRef}
        className="relative w-full overflow-hidden rounded-lg cursor-col-resize border border-[#333]"
        onMouseDown={() => setIsDragging(true)}
        onMouseUp={() => setIsDragging(false)}
        onMouseLeave={() => setIsDragging(false)}
        onMouseMove={onMouseMove}
        onTouchStart={() => setIsDragging(true)}
        onTouchEnd={() => setIsDragging(false)}
        onTouchMove={onTouchMove}
      >
        {!imagesLoaded ? (
          <div className="flex items-center justify-center w-full h-[400px] bg-[#111] animate-pulse text-[#D4AF37] font-mono">
            INITIALIZING CANVAS...
          </div>
        ) : (
          <>
            <canvas ref={canvasRef} className="block w-full h-auto pointer-events-none" />
            
            {/* Interactive Slider Handle Overlay */}
            <div 
              className="absolute top-0 bottom-0 w-1 bg-transparent flex justify-center items-center pointer-events-none"
              style={{ left: `${sliderPos}%`, transform: 'translateX(-50%)' }}
            >
              <div className="w-8 h-8 rounded-full bg-[#0a0a0a] border-2 border-[#D4AF37] shadow-[0_0_15px_rgba(212,175,55,0.4)] flex items-center justify-center">
                <div className="flex space-x-1">
                  <div className="w-0.5 h-3 bg-[#D4AF37]"></div>
                  <div className="w-0.5 h-3 bg-[#D4AF37]"></div>
                </div>
              </div>
            </div>
          </>
        )}
      </div>

      <div className="flex w-full justify-between mt-4 text-xs font-mono text-gray-500 tracking-widest">
        <span>GALLERY RECORD (A)</span>
        <span>PROBE UPLOAD (B)</span>
      </div>
    </div>
  );
}
