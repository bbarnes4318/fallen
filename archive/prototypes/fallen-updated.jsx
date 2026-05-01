import React, { useState, useEffect } from 'react';
import { ScanFace, Fingerprint, Lock, ShieldAlert, Activity, ChevronRight, CheckCircle2, Hexagon, Network, Cpu, Aperture, Globe, Scale, Server, ShieldCheck, Zap, Database, Code2, Terminal, Layers, Box, User, Crosshair, MapPin } from 'lucide-react';

export default function App() {
  const [terminalLines, setTerminalLines] = useState([]);
  const [isScanning, setIsScanning] = useState(true);
  const [activeTab, setActiveTab] = useState('request');

  // Simulated Terminal Boot Sequence reflecting the new Tier 4 Mark Detection Engine
  const bootSequence = [
    { text: "INITIALIZING BIOMETRIC PIPELINE v2.0...", delay: 500 },
    { text: "FETCHING PROBE & GALLERY PAYLOADS...", delay: 1200 },
    { text: "> RUNNING PAD: LAPLACIAN VARIANCE = 142.8 (BLUR_CHECK_PASSED)", color: "text-emerald-400", delay: 2000 },
    { text: "> APPLYING CLAHE NORMALIZATION...", delay: 2800 },
    { text: "> EXTRACTING MEDIAPIPE 468-NODE MESH...", delay: 3500 },
    { text: "> COMPUTING ARCFACE 512-D EMBEDDING...", delay: 4200 },
    { text: "> CALCULATING L2 GEOMETRIC RATIOS...", delay: 4800 },
    { text: "> EXTRACTING LBP MICRO-TOPOLOGY...", delay: 5500 },
    { text: "> ISOLATING SKIN SURFACE & DETECTING DISCRETE MARKS...", delay: 6200 },
    { text: "  [!] 4 MARKS DETECTED. COMPUTING HUNGARIAN BIPARTITE MATCHING...", color: "text-amber-400", delay: 6900 },
    { text: "CALIBRATION LOADED: LFW (5,989 PAIRS)", color: "text-cyan-400", delay: 7600 },
    { text: "FUSING SCORES (55/20/10/15 WEIGHTS)...", delay: 8300 },
    { text: "[VERIFIED] STATISTICAL CERTAINTY: 99.9999%", color: "text-emerald-400", delay: 9000 },
    { text: "GENERATING SCAR DELTA & DENSITY MAPS...", delay: 9700 },
    { text: "[SYS] FORENSIC RECEIPT MIGRATED TO COLD ROOM.", delay: 10400 },
    { text: "AWAITING NEXT VERIFICATION...", delay: 11500 },
  ];

  useEffect(() => {
    let timeouts = [];
    bootSequence.forEach((line) => {
      const timeout = setTimeout(() => {
        setTerminalLines((prev) => [...prev, line]);
        if (line.text.includes("AWAITING")) {
          setIsScanning(false);
        }
      }, line.delay);
      timeouts.push(timeout);
    });

    return () => timeouts.forEach(clearTimeout);
  }, []);

  return (
    <div className="min-h-screen bg-[#030712] text-slate-300 font-sans selection:bg-cyan-500/30 overflow-x-hidden relative">
      {/* Custom Styles for Animations & Effects */}
      <style dangerouslySetInnerHTML={{__html: `
        @keyframes scan {
          0% { transform: translateY(-100%); opacity: 0; }
          10% { opacity: 1; }
          90% { opacity: 1; }
          100% { transform: translateY(1000%); opacity: 0; }
        }
        .animate-scanner {
          animation: scan 2.5s cubic-bezier(0.4, 0, 0.2, 1) infinite;
        }
        @keyframes marquee {
          0% { transform: translateX(0); }
          100% { transform: translateX(-50%); }
        }
        .animate-marquee {
          display: flex;
          width: 200%;
          animation: marquee 30s linear infinite;
        }
        @keyframes float {
          0%, 100% { transform: translateY(0); }
          50% { transform: translateY(-10px); }
        }
        .animate-float {
          animation: float 6s ease-in-out infinite;
        }
        .glass-panel {
          background: rgba(15, 23, 42, 0.4);
          backdrop-filter: blur(16px);
          -webkit-backdrop-filter: blur(16px);
          border: 1px solid rgba(34, 211, 238, 0.1);
        }
        .cyan-gradient-text {
          background: linear-gradient(to right, #22d3ee, #3b82f6);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }
        .mesh-bg {
          background-image: 
            radial-gradient(at 40% 20%, hsla(189, 100%, 50%, 0.05) 0px, transparent 50%),
            radial-gradient(at 80% 0%, hsla(220, 100%, 50%, 0.05) 0px, transparent 50%),
            radial-gradient(at 0% 50%, hsla(189, 100%, 50%, 0.02) 0px, transparent 50%);
        }
        .mark-match-green {
          box-shadow: 0 0 10px rgba(16, 185, 129, 0.6);
          border-color: #10b981;
        }
        .mark-unmatch-yellow {
          box-shadow: 0 0 10px rgba(245, 158, 11, 0.6);
          border-color: #f59e0b;
        }
      `}} />

      {/* Background Elements */}
      <div className="absolute inset-0 z-0 pointer-events-none opacity-[0.02]" 
           style={{ backgroundImage: 'linear-gradient(#22d3ee 1px, transparent 1px), linear-gradient(90deg, #22d3ee 1px, transparent 1px)', backgroundSize: '3rem 3rem' }}>
      </div>
      <div className="absolute inset-0 z-0 mesh-bg pointer-events-none"></div>

      {/* Navigation */}
      <nav className="fixed top-0 w-full z-50 glass-panel border-b border-cyan-500/10">
        <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Aperture className="w-8 h-8 text-cyan-400" />
            <span className="text-xl font-bold tracking-widest text-white">AURUM<span className="text-cyan-500 font-light">SHIELD</span></span>
          </div>
          <div className="hidden md:flex items-center space-x-8 text-sm font-medium text-slate-400 tracking-wide">
            <a href="#engine" className="hover:text-cyan-400 transition-colors">QUAD-TIER ENGINE</a>
            <a href="#vault" className="hover:text-cyan-400 transition-colors">IDENTITY GRAPH</a>
            <a href="#compliance" className="hover:text-cyan-400 transition-colors">DAUBERT FORENSICS</a>
            <a href="#api" className="hover:text-cyan-400 transition-colors">API</a>
          </div>
          <div>
            <button className="bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 px-6 py-2.5 rounded-md text-sm font-bold tracking-wide transition-all shadow-[0_0_15px_rgba(34,211,238,0.1)] hover:shadow-[0_0_25px_rgba(34,211,238,0.2)]">
              ACCESS API
            </button>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <div className="relative z-10 pt-40 pb-20 px-6 max-w-7xl mx-auto flex flex-col lg:flex-row items-center justify-between gap-16">
        
        {/* Left Copy */}
        <div className="w-full lg:w-1/2 space-y-8">
          <div className="inline-flex items-center space-x-2 border border-cyan-500/30 bg-cyan-500/10 px-3 py-1 rounded-full">
            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span>
            <span className="text-xs font-mono text-cyan-300 tracking-widest uppercase">Tier 4 Mark Engine Deployed</span>
          </div>
          
          <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight text-white leading-[1.1]">
            Mathematical <br />
            Certainty in <br/>
            <span className="cyan-gradient-text">Facial Verification.</span>
          </h1>
          
          <p className="text-lg md:text-xl text-slate-400 leading-relaxed max-w-xl">
            A Forensic-Grade biometric engine fusing 512-D embeddings, geometric analysis, micro-topology, and Hungarian mark correspondence. Forensic-Grade forensics.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 pt-4">
            <button className="flex items-center justify-center space-x-2 bg-cyan-500 text-slate-950 px-8 py-4 rounded-md font-bold tracking-wide hover:bg-cyan-400 transition-colors shadow-[0_0_20px_rgba(34,211,238,0.4)]">
              <span>RUN VERIFICATION</span>
              <ChevronRight className="w-5 h-5" />
            </button>
            <button className="flex items-center justify-center space-x-2 glass-panel hover:bg-white/5 px-8 py-4 rounded-md font-bold tracking-wide text-white transition-colors">
              <Cpu className="w-5 h-5 text-slate-400" />
              <span>READ METHODOLOGY</span>
            </button>
          </div>
        </div>

        {/* Right UI/Terminal Visualizer */}
        <div className="w-full lg:w-1/2 relative">
          <div className="absolute inset-0 bg-cyan-500/10 blur-[120px] rounded-full pointer-events-none"></div>
          
          <div className="glass-panel rounded-xl border border-slate-700/50 shadow-[0_0_50px_rgba(0,0,0,0.5)] overflow-hidden relative flex flex-col md:flex-row animate-float">
            
            {/* Viewfinder / Scanner Side */}
            <div className="w-full md:w-2/5 border-b md:border-b-0 md:border-r border-slate-700/50 p-6 flex flex-col items-center justify-center bg-slate-900/50 relative">
              <div className="relative w-32 h-32 md:w-40 md:h-40 border border-cyan-500/20 flex items-center justify-center overflow-hidden">
                {/* Corner Brackets */}
                <div className="absolute top-0 left-0 w-3 h-3 border-t-2 border-l-2 border-cyan-400"></div>
                <div className="absolute top-0 right-0 w-3 h-3 border-t-2 border-r-2 border-cyan-400"></div>
                <div className="absolute bottom-0 left-0 w-3 h-3 border-b-2 border-l-2 border-cyan-400"></div>
                <div className="absolute bottom-0 right-0 w-3 h-3 border-b-2 border-r-2 border-cyan-400"></div>
                
                {/* Face Vector */}
                <ScanFace className={`w-20 h-20 text-slate-600 transition-colors duration-1000 ${!isScanning ? 'text-cyan-400' : ''}`} strokeWidth={1} />
                
                {/* Scanline */}
                {isScanning && (
                  <div className="absolute top-0 left-0 w-full h-[2px] bg-cyan-400 shadow-[0_0_15px_#22d3ee] animate-scanner"></div>
                )}
              </div>
              <div className="mt-6 text-center">
                <div className="text-[10px] font-mono text-slate-500 mb-1">PROBE STATUS</div>
                <div className={`text-xs font-mono font-bold tracking-widest ${isScanning ? 'text-amber-400 animate-pulse' : 'text-emerald-400'}`}>
                  {isScanning ? 'HUNGARIAN MATCHING...' : 'MARK CORRESPONDENCE OK'}
                </div>
              </div>
            </div>

            {/* Terminal Side */}
            <div className="w-full md:w-3/5 p-4 font-mono text-[10px] sm:text-xs h-64 md:h-80 overflow-y-auto flex flex-col relative bg-slate-950/80">
              <div className="flex justify-between items-center mb-4 border-b border-slate-800 pb-2">
                 <span className="text-slate-500">PIPELINE.LOG</span>
                 <span className="text-cyan-500/50">v2.0</span>
              </div>
              {terminalLines.map((line, idx) => (
                <div key={idx} className={`mb-1.5 ${line.color || 'text-slate-400'}`}>
                  {line.text}
                </div>
              ))}
              <div className="mt-auto animate-pulse text-cyan-500">_</div>
            </div>

          </div>

          {/* Floating Stat Cards */}
          <div className="absolute -bottom-6 -left-6 glass-panel px-6 py-4 rounded-lg border border-slate-700/50 shadow-xl hidden lg:block z-20">
            <div className="text-xs font-mono text-slate-500 mb-1">FUSED EQUAL ERROR RATE</div>
            <div className="text-2xl font-bold text-white">0.2031 <span className="text-emerald-500 text-sm ml-2">Verified</span></div>
          </div>
          <div className="absolute -top-6 -right-6 glass-panel px-6 py-4 rounded-lg border border-slate-700/50 shadow-xl hidden lg:block z-20">
            <div className="text-xs font-mono text-slate-500 mb-1">TIER 4 ACTIVE</div>
            <div className="text-lg font-bold text-amber-400 flex items-center">MARK MCS {'>'} 0.35</div>
          </div>
        </div>
      </div>

      {/* Live Telemetry Marquee */}
      <div className="border-y border-white/5 bg-slate-900/50 overflow-hidden py-3 mt-12 relative z-10 backdrop-blur-md">
        <div className="animate-marquee font-mono text-xs text-slate-400 flex space-x-12">
          {[...Array(3)].map((_, j) => (
            <React.Fragment key={j}>
              <div className="flex items-center"><span className="text-cyan-500 mr-2">●</span> PRE-DECODE HASH: ENABLED </div>
              <div className="flex items-center"><span className="text-emerald-500 mr-2">●</span> PAD: BLUR_CHECK_PASSED (142.8σ²)</div>
              <div className="flex items-center"><span className="text-amber-500 mr-2">●</span> MARK CORRESPONDENCE: MATCHED</div>
              <div className="flex items-center"><span className="text-cyan-500 mr-2">●</span> FUSED EER: 0.2031</div>
              <div className="flex items-center"><span className="text-amber-500 mr-2">●</span> Immutable ANCHOR: COMMITTED</div>
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* Quad-Tiered Engine Section */}
      <div id="engine" className="max-w-7xl mx-auto px-6 py-32 relative z-10">
        <div className="text-center mb-20">
          <h2 className="text-3xl md:text-5xl font-bold text-white mb-6">The Quad-Tiered Identity Engine</h2>
          <p className="text-slate-400 max-w-2xl mx-auto text-lg leading-relaxed">
            The pipeline dynamically shifts from 3-tier to 4-tier verification, fusing deep neural networks, geometry, micro-topology, and rigorous mathematical mark correspondence.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {[
            {
              icon: Hexagon,
              weight: "55%",
              title: "Structural Identity",
              subtitle: "ArcFace 512-D Latent Space",
              desc: "Extracts deep features using ResNet-100. Computed via Cosine Similarity, forming the core identity signal. Triggers an automatic veto if similarity drops below 0.40."
            },
            {
              icon: Network,
              weight: "20%",
              title: "Geometric Biometrics",
              subtitle: "468-Node MediaPipe Mesh",
              desc: "Analyzes 12 scale-invariant anthropometric ratios. Scored using L2 Euclidean Distance to catch structural anomalies that 2D deepfakes often miss."
            },
            {
              icon: Fingerprint,
              weight: "10%",
              title: "Micro-Topology",
              subtitle: "Local Binary Patterns",
              desc: "Generates rotation-invariant texture histograms. Scored via Chi-Squared distance to mathematically verify the physical surface variance of the skin."
            },
            {
              icon: Crosshair,
              weight: "15%",
              title: "Mark Correspondence",
              subtitle: "Hungarian Bipartite Matching",
              desc: "Isolates scars/moles via adaptive thresholding. Computes an optimal match based on spatial centroid, area, intensity, and circularity. Min. 3 marks required."
            }
          ].map((tier, i) => (
            <div key={i} className="glass-panel p-6 rounded-xl border border-slate-800 hover:border-cyan-500/40 transition-all duration-300 group hover:-translate-y-1 relative overflow-hidden">
              <div className="absolute top-0 right-0 bg-slate-800/80 px-3 py-1 rounded-bl-lg border-l border-b border-slate-700 text-xs font-mono text-cyan-400 font-bold">
                {tier.weight} WEIGHT
              </div>
              <div className="bg-slate-900 w-12 h-12 rounded-lg flex items-center justify-center mb-5 border border-slate-700 group-hover:border-cyan-500/50 transition-colors mt-2">
                <tier.icon className="w-6 h-6 text-slate-400 group-hover:text-cyan-400 transition-colors" />
              </div>
              <h3 className="text-lg font-bold text-white mb-1">{tier.title}</h3>
              <div className="text-[11px] font-mono text-cyan-500 mb-3">{tier.subtitle}</div>
              <p className="text-sm text-slate-400 leading-relaxed">{tier.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* 1:N Vault Target Acquisition (Graph Overhaul) */}
      <div id="vault" className="border-t border-white/5 bg-gradient-to-b from-slate-950 to-slate-900 relative z-10 overflow-hidden">
        <div className="absolute top-0 right-0 w-[800px] h-[800px] bg-cyan-500/5 blur-[150px] rounded-full pointer-events-none"></div>
        <div className="max-w-7xl mx-auto px-6 py-32">
          <div className="flex flex-col lg:flex-row items-center gap-16">
            
            {/* Left Content */}
            <div className="w-full lg:w-1/2 space-y-6 z-10">
              <div className="text-xs font-mono text-cyan-500 tracking-widest uppercase">Decoupled Architecture</div>
              <h2 className="text-3xl md:text-5xl font-bold text-white leading-tight">
                Interactive <br/>Identity Graph.
              </h2>
              <p className="text-slate-400 text-lg leading-relaxed">
                The completely overhauled UI maps Fallen identity networks asynchronously. Identify critical matches and track high-connectivity anomalies through a visual thumbnail graph linked to comprehensive entity dossiers.
              </p>
              
              <div className="pt-6 space-y-6">
                <div className="flex items-start">
                  <div className="bg-amber-500/10 p-3 rounded-lg border border-amber-500/30 mr-4">
                    <Activity className="w-5 h-5 text-amber-400" />
                  </div>
                  <div>
                    <h4 className="text-white font-bold text-lg mb-1">Anomaly Ring Detection</h4>
                    <p className="text-sm text-slate-500 leading-relaxed">Face thumbnails rendered on nodes instantly highlight potential duplicates or aliases with an identifying gold anomaly ring.</p>
                  </div>
                </div>
                <div className="flex items-start">
                  <div className="bg-cyan-500/10 p-3 rounded-lg border border-cyan-500/30 mr-4">
                    <Layers className="w-5 h-5 text-cyan-400" />
                  </div>
                  <div>
                    <h4 className="text-white font-bold text-lg mb-1">Entity Dossier Panel</h4>
                    <p className="text-sm text-slate-500 leading-relaxed">Clicking a node reveals a deep-dive side panel showing connected entities, exact match scores, and instantly generated KMS-signed thumbnail URIs.</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Right Interactive/Visual Node Graph Mockup with New Features */}
            <div className="w-full lg:w-1/2 relative h-[500px] flex items-center justify-center border border-slate-700/50 rounded-xl bg-[#0b1120] overflow-hidden shadow-2xl">
               
               {/* Contextual Legend Panel Mockup */}
               <div className="absolute top-4 left-4 bg-slate-900/80 backdrop-blur-md p-3 rounded border border-slate-700 text-[9px] font-mono text-slate-400 z-30 shadow-lg">
                  <div className="text-cyan-400 font-bold mb-2">NETWORK LEGEND</div>
                  <div className="flex items-center gap-2 mb-1.5">
                    <div className="w-3 h-3 rounded-full border border-amber-500 shadow-[0_0_5px_#f59e0b] bg-slate-800"></div> Anomaly (Group 2)
                  </div>
                  <div className="flex items-center gap-2 mb-1.5">
                    <div className="w-3 h-3 rounded-full border border-slate-500 bg-slate-800"></div> Standard Node
                  </div>
                  <div className="flex items-center gap-2 mb-1.5">
                    <div className="w-4 h-[2px] bg-red-500"></div> Critical Match &gt;95%
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-[2px] bg-slate-600"></div> Strong Match 90-95%
                  </div>
               </div>

               {/* Entity Dossier Side Panel Mockup */}
               <div className="absolute top-0 right-0 w-48 h-full bg-slate-900/90 border-l border-slate-700 p-4 z-30 backdrop-blur-md transform transition-transform shadow-2xl">
                  <div className="text-[10px] font-mono text-cyan-500 mb-4">ENTITY DOSSIER</div>
                  <div className="w-16 h-16 rounded-full border-2 border-amber-500 mx-auto mb-3 bg-slate-800 flex items-center justify-center overflow-hidden shadow-[0_0_15px_#f59e0b]">
                    <User className="w-8 h-8 text-amber-500/50" />
                  </div>
                  <div className="text-center font-bold text-white text-sm mb-1">Subject_Alpha</div>
                  <div className="text-center font-mono text-[9px] text-amber-400 mb-4">GROUP 2 ANOMALY</div>
                  
                  <div className="text-[9px] font-mono text-slate-500 mb-2 border-b border-slate-800 pb-1">CONNECTIONS</div>
                  <div className="space-y-2">
                    <div className="flex items-center gap-2 bg-slate-950 p-1.5 rounded border border-slate-800">
                      <div className="w-6 h-6 rounded-full bg-slate-800 flex items-center justify-center border border-slate-600"><User className="w-3 h-3 text-slate-500" /></div>
                      <div>
                        <div className="text-[9px] text-white">Subject_092</div>
                        <div className="text-[8px] text-red-400 font-mono">98.2% MATCH</div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 bg-slate-950 p-1.5 rounded border border-slate-800">
                      <div className="w-6 h-6 rounded-full bg-slate-800 flex items-center justify-center border border-slate-600"><User className="w-3 h-3 text-slate-500" /></div>
                      <div>
                        <div className="text-[9px] text-white">Alias_Charlie</div>
                        <div className="text-[8px] text-slate-400 font-mono">92.4% MATCH</div>
                      </div>
                    </div>
                  </div>
               </div>

               <div className="absolute inset-0 flex items-center justify-center pr-24">
                  {/* Central Node (Face Thumbnail Mock) */}
                  <div className="relative z-20 w-20 h-20 bg-slate-800 border-2 border-amber-500 rounded-full flex items-center justify-center shadow-[0_0_20px_#f59e0b] animate-pulse-slow">
                     <User className="w-8 h-8 text-slate-400" />
                  </div>
                  
                  {/* Connecting Lines & Surrounding Nodes (Face Thumbnail Mocks) */}
                  {[
                    { rot: 30, delay: '0s', critical: true, anomaly: false, dist: 120 },
                    { rot: 110, delay: '0.2s', critical: false, anomaly: false, dist: 140 },
                    { rot: 180, delay: '0.4s', critical: true, anomaly: true, dist: 110 },
                    { rot: 250, delay: '0.6s', critical: false, anomaly: false, dist: 150 },
                    { rot: 320, delay: '0.8s', critical: false, anomaly: false, dist: 130 },
                  ].map((node, i) => (
                    <div key={i} className="absolute inset-0 flex items-center justify-center pointer-events-none" style={{ transform: `rotate(${node.rot}deg)` }}>
                      <div className="w-full h-full relative" style={{ width: node.dist * 2, height: node.dist * 2 }}>
                        <div className={`absolute top-0 left-1/2 w-[2px] h-1/2 origin-bottom ${node.critical ? 'bg-red-500 shadow-[0_0_8px_#ef4444]' : 'bg-slate-600'}`}></div>
                        
                        <div className={`absolute top-0 left-1/2 -translate-x-1/2 -translate-y-1/2 w-10 h-10 rounded-full border-2 ${node.anomaly ? 'bg-slate-800 border-amber-500 shadow-[0_0_10px_#f59e0b]' : 'bg-slate-800 border-slate-500'} flex items-center justify-center transform`} style={{ transform: `translate(-50%, -50%) rotate(-${node.rot}deg)` }}>
                          <User className="w-5 h-5 text-slate-500" />
                        </div>
                      </div>
                    </div>
                  ))}
               </div>
            </div>

          </div>
        </div>
      </div>

      {/* Forensic Evidence Visual Section (Receipt & Delta Tab) */}
      <div id="compliance" className="border-t border-white/5 bg-slate-900/40 relative z-10">
        <div className="max-w-7xl mx-auto px-6 py-32">
          <div className="flex flex-col lg:flex-row items-center gap-16">
            
            <div className="w-full lg:w-1/2 space-y-6">
              <div className="text-xs font-mono text-cyan-500 tracking-widest uppercase">Visual Forensics & Delta Output</div>
              <h2 className="text-3xl md:text-5xl font-bold text-white leading-tight">
                Defensible in <br/>Cross-Examination.
              </h2>
              <p className="text-slate-400 text-lg leading-relaxed">
                Raw scores aren't enough for the courtroom. The API generates permanent visual evidence, explicitly detailing algorithmic decisions from landmark density down to individual mark correspondences.
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 pt-6">
                <div>
                  <h4 className="text-white font-bold mb-2 flex items-center"><Activity className="w-4 h-4 text-cyan-400 mr-2"/> Landmark Density</h4>
                  <p className="text-sm text-slate-500">Maps a Gaussian kernel over the 468 mesh points. Proves exactly where the system anchored its measurements.</p>
                </div>
                <div>
                  <h4 className="text-white font-bold mb-2 flex items-center"><MapPin className="w-4 h-4 text-cyan-400 mr-2"/> Scar/Mark Delta Tracker</h4>
                  <p className="text-sm text-slate-500">Visualizes the Hungarian match matrix. Green circles highlight mathematically matched anomalies; yellow highlight unmatched features.</p>
                </div>
              </div>
            </div>

            {/* Simulated Visual Evidence UI with Delta Tab */}
            <div className="w-full lg:w-1/2">
              <div className="glass-panel border border-slate-700/50 rounded-xl shadow-[0_0_50px_rgba(0,0,0,0.5)] p-2 relative overflow-hidden">
                <div className="bg-slate-950 rounded-lg p-6 border border-slate-800">
                  <div className="flex justify-between items-end mb-6 border-b border-slate-800 pb-4">
                    <div>
                      <div className="text-[10px] font-mono text-cyan-500 tracking-widest mb-1">EVIDENCE LOCKER</div>
                      <div className="text-lg font-bold text-white tracking-widest uppercase">Forensic Attestation Record</div>
                    </div>
                    <div className="text-right">
                      <div className="text-xl font-black text-emerald-400">98.48%</div>
                      <div className="text-[10px] font-mono text-slate-500">FUSED SIMILARITY</div>
                    </div>
                  </div>

                  {/* 4-Panel Mockup including the new Delta Tab Concept */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
                    <div className="aspect-square bg-slate-900 rounded-md relative overflow-hidden border border-slate-700/50 flex flex-col items-center justify-center p-2 group">
                      <ScanFace className="w-8 h-8 text-slate-600 mb-2" strokeWidth={1.5} />
                      <span className="text-[9px] font-mono text-slate-400 relative z-10 text-center">PROBE<br/>CROP</span>
                    </div>
                    <div className="aspect-square bg-slate-900 rounded-md relative overflow-hidden border border-slate-700/50 flex flex-col items-center justify-center p-2">
                      <ScanFace className="w-8 h-8 text-slate-600 mb-2" strokeWidth={1.5} />
                      <span className="text-[9px] font-mono text-slate-400 relative z-10 text-center">GALLERY<br/>CROP</span>
                    </div>
                    <div className="aspect-square bg-slate-950 rounded-md relative overflow-hidden border border-cyan-500/30 flex flex-col items-center justify-center p-2">
                      <div className="absolute inset-0 opacity-40 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-cyan-500 via-blue-500 to-transparent blur-md scale-75"></div>
                      <Aperture className="w-8 h-8 text-cyan-300 mb-2 relative z-10" strokeWidth={1.5} />
                      <span className="text-[9px] font-mono text-cyan-400 relative z-10 text-center">DENSITY<br/>MAP</span>
                    </div>
                    {/* The New Delta Tab Mockup */}
                    <div className="aspect-square bg-slate-900 rounded-md relative overflow-hidden border border-emerald-500/30 flex flex-col items-center justify-center p-2">
                      {/* Simulated marks */}
                      <div className="absolute w-2 h-2 rounded-full mark-match-green top-[30%] left-[30%] border"></div>
                      <div className="absolute w-2 h-2 rounded-full mark-match-green bottom-[40%] right-[30%] border"></div>
                      <div className="absolute w-2 h-2 rounded-full mark-unmatch-yellow top-[60%] left-[20%] border"></div>
                      <Crosshair className="w-8 h-8 text-emerald-400/50 mb-2 relative z-10" strokeWidth={1.5} />
                      <span className="text-[9px] font-mono text-emerald-400 relative z-10 text-center">MARK<br/>DELTA</span>
                    </div>
                  </div>

                  <div className="space-y-2 font-mono text-[9px] sm:text-[10px] bg-slate-900 p-3 rounded-md border border-slate-800 grid grid-cols-1 sm:grid-cols-2 gap-x-4">
                    <div className="flex justify-between border-b border-slate-800 pb-1 sm:border-none sm:pb-0">
                      <span className="text-slate-500">PAD METHOD:</span>
                      <span className="text-emerald-400">LAPLACIAN (142.8)</span>
                    </div>
                    <div className="flex justify-between border-b border-slate-800 pb-1 sm:border-none sm:pb-0">
                      <span className="text-slate-500">STATISTICAL CERTAINTY:</span>
                      <span className="text-white">1 IN 4.2M FAR</span>
                    </div>
                    <div className="flex justify-between border-b border-slate-800 pb-1 sm:border-none sm:pb-0">
                      <span className="text-slate-500">MARK CORRESPONDENCE:</span>
                      <span className="text-emerald-400">0.82 MCS (3 MATCHED)</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-500">PROBE SHA-256:</span>
                      <span className="text-slate-400 truncate max-w-[80px]">e3b0c44298fc1c...</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

          </div>
        </div>
      </div>

      {/* Deployment Vectors (Use Cases) */}
      <div id="vectors" className="border-t border-white/5 bg-slate-900/10 relative z-10">
        <div className="max-w-7xl mx-auto px-6 py-32">
          <div className="mb-16">
            <h2 className="text-3xl md:text-5xl font-bold text-white mb-6">Deployment Vectors.</h2>
            <p className="text-slate-400 max-w-2xl text-lg leading-relaxed">
              Designed for environments where false acceptances carry catastrophic legal, financial, or physical consequences.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              {
                icon: Globe,
                title: "Principal FBO Escrow",
                desc: "Clear high-value physical capital transfers with non-repudiable biometric authorization mapped directly to escrow conditions."
              },
              {
                icon: Scale,
                title: "Forensic Legal Discovery",
                desc: "Generate Daubert-standard documentation to definitively prove or disprove identity claims during federal and civil litigation."
              },
              {
                icon: Server,
                title: "Zero-Trust Infrastructure",
                desc: "Protect sensitive air-gapped networks and critical infrastructure through mathematically proven, multi-modal verification."
              },
              {
                icon: ShieldCheck,
                title: "Cross-Border Security",
                desc: "Deploy high-throughput 1:N target acquisition against an encrypted identity graph without exposing raw biometric databases."
              }
            ].map((vector, i) => (
              <div key={i} className="bg-slate-900/50 p-8 rounded-xl border border-slate-800/50 hover:bg-slate-800 transition-colors">
                <vector.icon className="w-8 h-8 text-cyan-500 mb-6" />
                <h4 className="text-lg font-bold text-white mb-3">{vector.title}</h4>
                <p className="text-sm text-slate-400 leading-relaxed">{vector.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* API / Developer Section */}
      <div id="api" className="border-t border-white/5 bg-[#030712] relative z-10">
        <div className="max-w-7xl mx-auto px-6 py-32">
          <div className="flex flex-col lg:flex-row items-center gap-16">
            
            <div className="w-full lg:w-5/12 space-y-6">
              <div className="text-xs font-mono text-cyan-500 tracking-widest uppercase">Developer-First</div>
              <h2 className="text-3xl md:text-5xl font-bold text-white leading-tight">
                Integrate in <br/>Minutes.
              </h2>
              <p className="text-slate-400 text-lg leading-relaxed">
                Connect your escrow platform, physical security gates, or clearinghouse directly to the API. Simple REST endpoints return massive forensic payloads.
              </p>
              <div className="space-y-4 pt-6 font-mono text-sm">
                <div className="flex items-center text-slate-300 bg-slate-900/50 border border-slate-800 p-3 rounded-md">
                  <span className="text-emerald-400 font-bold w-16">POST</span>
                  <span>/v2/generate-upload-urls</span>
                </div>
                <div className="flex items-center text-slate-300 bg-slate-900/50 border border-slate-800 p-3 rounded-md">
                  <span className="text-emerald-400 font-bold w-16">POST</span>
                  <span>/v2/verify/fuse</span>
                </div>
                <div className="flex items-center text-slate-300 bg-slate-900/50 border border-slate-800 p-3 rounded-md">
                  <span className="text-amber-400 font-bold w-16">POST</span>
                  <span>/v2/vault/search</span>
                </div>
                <div className="flex items-center text-slate-300 bg-slate-900/50 border border-slate-800 p-3 rounded-md">
                  <span className="text-cyan-400 font-bold w-16">GET</span>
                  <span>/v2/vault/network</span>
                </div>
              </div>
            </div>

            {/* API Code Snippet Mockup */}
            <div className="w-full lg:w-7/12">
              <div className="glass-panel rounded-xl overflow-hidden shadow-2xl border border-slate-700/50">
                <div className="flex bg-slate-900 px-4 py-3 border-b border-slate-800 space-x-4">
                  <button onClick={() => setActiveTab('request')} className={`text-xs font-mono font-bold tracking-widest pb-1 border-b-2 transition-colors ${activeTab === 'request' ? 'text-cyan-400 border-cyan-400' : 'text-slate-500 border-transparent hover:text-slate-300'}`}>CURL REQUEST</button>
                  <button onClick={() => setActiveTab('response')} className={`text-xs font-mono font-bold tracking-widest pb-1 border-b-2 transition-colors ${activeTab === 'response' ? 'text-cyan-400 border-cyan-400' : 'text-slate-500 border-transparent hover:text-slate-300'}`}>JSON RESPONSE</button>
                </div>
                <div className="p-6 bg-[#0B1120] text-sm font-mono overflow-x-auto">
                  {activeTab === 'request' ? (
                    <pre className="text-slate-300">
                      <code dangerouslySetInnerHTML={{__html: `curl -X POST https://api.Fallen.com/v2/verify/fuse \\
  -H "Authorization: Bearer <span class="text-emerald-400">sk_live_...</span>" \\
  -H "Content-Type: application/json" \\
  -d '{
    <span class="text-cyan-400">"gallery_url"</span>: <span class="text-amber-300">"gs://bucket/gallery_a82j.jpg"</span>,
    <span class="text-cyan-400">"probe_url"</span>: <span class="text-amber-300">"gs://bucket/probe_x91k.jpg"</span>
  }'`}} />
                    </pre>
                  ) : (
                    <pre className="text-slate-300">
                      <code dangerouslySetInnerHTML={{__html: `{
  <span class="text-cyan-400">"fused_identity_score"</span>: <span class="text-purple-400">98.48</span>,
  <span class="text-cyan-400">"conclusion"</span>: <span class="text-amber-300">"Strongest Support for Common Source"</span>,
  <span class="text-cyan-400">"veto_triggered"</span>: <span class="text-purple-400">false</span>,
  <span class="text-cyan-400">"mark_correspondence_score"</span>: <span class="text-purple-400">0.82</span>,
  <span class="text-cyan-400">"audit_log"</span>: {
    <span class="text-cyan-400">"statistical_certainty"</span>: <span class="text-amber-300">"99.9999%"</span>,
    <span class="text-cyan-400">"false_acceptance_rate"</span>: <span class="text-amber-300">"1 in 4.2 Million"</span>,
    <span class="text-cyan-400">"liveness_check"</span>: {
      <span class="text-cyan-400">"method"</span>: <span class="text-amber-300">"LAPLACIAN_VARIANCE"</span>,
      <span class="text-cyan-400">"status"</span>: <span class="text-amber-300">"BLUR_CHECK_PASSED"</span>
    },
    <span class="text-cyan-400">"probe_file_hash"</span>: <span class="text-amber-300">"e3b0c44298fc..."</span>
  }
}`}} />
                    </pre>
                  )}
                </div>
              </div>
            </div>

          </div>
        </div>
      </div>

      {/* Compliance & Security Grid */}
      <div id="security" className="max-w-7xl mx-auto px-6 py-20 relative z-10">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
           <div className="p-6 rounded-lg bg-slate-900/30 border border-slate-800/50 flex flex-col items-center">
             <Database className="w-6 h-6 text-cyan-500 mb-3" />
             <div className="text-sm font-bold text-white mb-1">Immutable Immutable Ledger</div>
             <div className="text-xs text-slate-500">Cryptographically Anchored</div>
           </div>
           <div className="p-6 rounded-lg bg-slate-900/30 border border-slate-800/50 flex flex-col items-center">
             <Lock className="w-6 h-6 text-cyan-500 mb-3" />
             <div className="text-sm font-bold text-white mb-1">AES-256-GCM Envelope</div>
             <div className="text-xs text-slate-500">GCP KMS Backed Encryption</div>
           </div>
           <div className="p-6 rounded-lg bg-slate-900/30 border border-slate-800/50 flex flex-col items-center">
             <Zap className="w-6 h-6 text-cyan-500 mb-3" />
             <div className="text-sm font-bold text-white mb-1">O(N²) Async Decoupling</div>
             <div className="text-xs text-slate-500">Massive Scale Graphing</div>
           </div>
           <div className="p-6 rounded-lg bg-slate-900/30 border border-slate-800/50 flex flex-col items-center">
             <ShieldCheck className="w-6 h-6 text-cyan-500 mb-3" />
             <div className="text-sm font-bold text-white mb-1">Pre-Decode Hashing</div>
             <div className="text-xs text-slate-500">Zero Spoliation of Evidence</div>
           </div>
        </div>
      </div>

      {/* Footer CTA */}
      <div className="relative z-10 border-t border-white/5 bg-black/80">
        <div className="max-w-4xl mx-auto px-6 py-24 text-center">
          <Cpu className="w-16 h-16 text-cyan-500/20 mx-auto mb-8" />
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-6">Integrate the Verification Engine.</h2>
          <p className="text-slate-400 mb-10 text-lg">
            Access the REST API. Decoupled O(N²) identity graphs, KMS envelope encryption, and mathematically proven biometrics.
          </p>
          <button className="bg-cyan-500 hover:bg-cyan-400 text-slate-950 px-8 py-4 rounded-md font-bold tracking-wide transition-all shadow-[0_0_20px_rgba(34,211,238,0.3)] hover:shadow-[0_0_30px_rgba(34,211,238,0.5)]">
            VIEW API DOCUMENTATION
          </button>
        </div>
        
        <div className="border-t border-slate-900 py-8">
          <div className="max-w-7xl mx-auto px-6 flex flex-col md:flex-row justify-between items-center text-xs font-mono text-slate-600">
            <div>© 2026 Fallen FORENSICS. ALL RIGHTS RESERVED.</div>
            <div className="flex space-x-6 mt-4 md:mt-0">
              <a href="#" className="hover:text-slate-400 transition-colors">METHODOLOGY.MD</a>
              <a href="#" className="hover:text-slate-400 transition-colors">API REF</a>
              <a href="#" className="hover:text-slate-400 transition-colors">SYSTEM STATUS</a>
            </div>
          </div>
        </div>
      </div>
      
    </div>
  );
}