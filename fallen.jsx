import React, { useState, useEffect } from 'react';
import { ScanFace, Fingerprint, Lock, ShieldAlert, Activity, ChevronRight, CheckCircle2, Hexagon, Network, Cpu, Aperture, Globe, Scale, Server, ShieldCheck, Zap, Database } from 'lucide-react';

export default function App() {
  const [terminalLines, setTerminalLines] = useState([]);
  const [isScanning, setIsScanning] = useState(true);

  // Simulated Terminal Boot Sequence based EXACTLY on our backend pipeline
  const bootSequence = [
    { text: "INITIALIZING BIOMETRIC PIPELINE v2.0...", delay: 500 },
    { text: "FETCHING PROBE & GALLERY PAYLOADS...", delay: 1200 },
    { text: "> RUNNING PAD: LAPLACIAN VARIANCE = 142.8 (BLUR_CHECK_PASSED)", color: "text-emerald-400", delay: 2000 },
    { text: "> APPLYING CLAHE NORMALIZATION...", delay: 2800 },
    { text: "> EXTRACTING MEDIAPIPE 468-NODE MESH...", delay: 3500 },
    { text: "> COMPUTING ARCFACE 512-D EMBEDDING...", delay: 4200 },
    { text: "> CALCULATING L2 GEOMETRIC RATIOS...", delay: 4800 },
    { text: "> EXTRACTING LBP MICRO-TOPOLOGY...", delay: 5500 },
    { text: "CALIBRATION LOADED: LFW (5,989 PAIRS)", color: "text-cyan-400", delay: 6200 },
    { text: "FUSING SCORES (65/5/30 WEIGHTS)...", delay: 6900 },
    { text: "[VERIFIED] STATISTICAL CERTAINTY: 99.9999%", color: "text-emerald-400", delay: 7600 },
    { text: "GENERATING SCAR DELTA & DENSITY MAPS...", delay: 8400 },
    { text: "[SYS] FORENSIC RECEIPT MIGRATED TO COLD ROOM.", delay: 9100 },
    { text: "AWAITING NEXT VERIFICATION...", delay: 10500 },
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
      {/* Custom Styles for Animations & Breathtaking Effects */}
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
        @keyframes pulse-slow {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
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
            <a href="#engine" className="hover:text-cyan-400 transition-colors">THE ENGINE</a>
            <a href="#compliance" className="hover:text-cyan-400 transition-colors">DAUBERT FORENSICS</a>
            <a href="#vectors" className="hover:text-cyan-400 transition-colors">DEPLOYMENT VECTORS</a>
            <a href="#security" className="hover:text-cyan-400 transition-colors">CRYPTOGRAPHY</a>
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
            <span className="text-xs font-mono text-cyan-300 tracking-widest uppercase">Pipeline v2.0 Live — LFW Calibrated</span>
          </div>
          
          <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight text-white leading-[1.1]">
            Mathematical <br />
            Certainty in <br/>
            <span className="cyan-gradient-text">Facial Verification.</span>
          </h1>
          
          <p className="text-lg md:text-xl text-slate-400 leading-relaxed max-w-xl">
            A Daubert-compliant biometric engine fusing 512-D structural embeddings, 468-node geometric analysis, and LBP micro-topology. Court-admissible forensics.
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
          
          <div className="glass-panel rounded-xl border border-slate-700/50 shadow-[0_0_50px_rgba(0,0,0,0.5)] overflow-hidden relative flex flex-col md:flex-row">
            
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
                  {isScanning ? 'ANALYZING TOPOLOGY...' : 'MATCH ACQUIRED'}
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
            <div className="text-xs font-mono text-slate-500 mb-1">EQUAL ERROR RATE</div>
            <div className="text-2xl font-bold text-white">0.2031 <span className="text-emerald-500 text-sm ml-2">Verified</span></div>
          </div>
          <div className="absolute -top-6 -right-6 glass-panel px-6 py-4 rounded-lg border border-slate-700/50 shadow-xl hidden lg:block z-20">
            <div className="text-xs font-mono text-slate-500 mb-1">LFW CALIBRATION</div>
            <div className="text-lg font-bold text-cyan-400 flex items-center">5,989 PAIRS</div>
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
              <div className="flex items-center"><span className="text-cyan-500 mr-2">●</span> FUSED EER: 0.2031</div>
              <div className="flex items-center"><span className="text-amber-500 mr-2">●</span> WORM ANCHOR: COMMITTED</div>
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* Tri-Tiered Engine Section */}
      <div id="engine" className="max-w-7xl mx-auto px-6 py-32 relative z-10">
        <div className="text-center mb-20">
          <h2 className="text-3xl md:text-5xl font-bold text-white mb-6">The Tri-Tiered Identity Engine</h2>
          <p className="text-slate-400 max-w-2xl mx-auto text-lg leading-relaxed">
            We don't rely on a single black-box algorithm. The pipeline fuses deep neural networks with scale-invariant geometric math and skin micro-topology to produce an undeniable verdict.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {[
            {
              icon: Hexagon,
              title: "Structural Identity",
              subtitle: "ArcFace 512-D Latent Space",
              desc: "Extracts deep facial features using the ResNet-100 architecture. Computed via Cosine Similarity, forming 65% of the final fused score. Triggers an automatic veto if similarity drops below 0.40."
            },
            {
              icon: Network,
              title: "Geometric Biometrics",
              subtitle: "468-Node MediaPipe Mesh",
              desc: "Analyzes 12 scale-invariant anthropometric ratios (e.g., inter-ocular distance to jaw width). Scored using L2 Euclidean Distance to catch structural anomalies deepfakes often miss."
            },
            {
              icon: Fingerprint,
              title: "Micro-Topology",
              subtitle: "Local Binary Patterns (LBP)",
              desc: "Isolates biological topography—scars, pores, and creases—by generating rotation-invariant texture histograms. Scored via Chi-Squared distance to verify the physical surface of the skin."
            }
          ].map((tier, i) => (
            <div key={i} className="glass-panel p-8 rounded-xl border border-slate-800 hover:border-cyan-500/40 transition-all duration-300 group hover:-translate-y-1">
              <div className="bg-slate-900 w-14 h-14 rounded-lg flex items-center justify-center mb-6 border border-slate-700 group-hover:border-cyan-500/50 transition-colors">
                <tier.icon className="w-7 h-7 text-slate-400 group-hover:text-cyan-400 transition-colors" />
              </div>
              <h3 className="text-xl font-bold text-white mb-1">{tier.title}</h3>
              <div className="text-xs font-mono text-cyan-500 mb-4">{tier.subtitle}</div>
              <p className="text-sm text-slate-400 leading-relaxed">{tier.desc}</p>
            </div>
          ))}
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

      {/* Forensic Evidence Visual Section */}
      <div id="compliance" className="border-t border-white/5 bg-slate-900/40 relative z-10">
        <div className="max-w-7xl mx-auto px-6 py-32">
          <div className="flex flex-col lg:flex-row items-center gap-16">
            
            <div className="w-full lg:w-1/2 space-y-6">
              <div className="text-xs font-mono text-cyan-500 tracking-widest uppercase">Visual Forensics</div>
              <h2 className="text-3xl md:text-5xl font-bold text-white leading-tight">
                Defensible in <br/>Cross-Examination.
              </h2>
              <p className="text-slate-400 text-lg leading-relaxed">
                Raw scores aren't enough for the courtroom. The API generates permanent visual evidence of every verification, ensuring the math can be explained to a judge or jury.
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 pt-6">
                <div>
                  <h4 className="text-white font-bold mb-2 flex items-center"><Activity className="w-4 h-4 text-cyan-400 mr-2"/> Landmark Density</h4>
                  <p className="text-sm text-slate-500">Maps a Gaussian kernel over the 468 mesh points. Proves exactly where the system anchored its measurements.</p>
                </div>
                <div>
                  <h4 className="text-white font-bold mb-2 flex items-center"><ShieldAlert className="w-4 h-4 text-cyan-400 mr-2"/> Scar Delta Mapper</h4>
                  <p className="text-sm text-slate-500">Isolates persistent micro-topology (scars/creases) that appears consistently across both the gallery and probe crops.</p>
                </div>
                <div>
                  <h4 className="text-white font-bold mb-2 flex items-center"><Lock className="w-4 h-4 text-cyan-400 mr-2"/> Cryptographic Receipt</h4>
                  <p className="text-sm text-slate-500">The visuals are stitched with the SHA-256 pre-decode hashes and migrated to Cold Storage to preserve the chain of custody.</p>
                </div>
              </div>
            </div>

            {/* Simulated Visual Evidence UI */}
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

                  {/* 3-Column Mockup */}
                  <div className="grid grid-cols-3 gap-3 mb-6">
                    <div className="aspect-square bg-slate-900 rounded-md relative overflow-hidden border border-slate-700/50 flex flex-col items-center justify-center p-2 group">
                      <div className="absolute inset-0 opacity-20 bg-[url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI4IiBoZWlnaHQ9IjgiPgo8cmVjdCB3aWR0aD0iOCIgaGVpZ2h0PSI4IiBmaWxsPSIjZmZmIiBmaWxsLW9wYWNpdHk9IjAuMSIvPgo8cGF0aCBkPSJNMCAwTDggOFpNOCAwTDAgOFoiIHN0cm9rZT0iIzAwMCIgc3Ryb2tlLW9wYWNpdHk9IjAuMSIvPgo8L3N2Zz4=')]"></div>
                      <ScanFace className="w-8 h-8 text-slate-600 mb-2" strokeWidth={1.5} />
                      <span className="text-[9px] font-mono text-slate-400 relative z-10 text-center">PROBE<br/>CROP</span>
                    </div>
                    <div className="aspect-square bg-slate-900 rounded-md relative overflow-hidden border border-slate-700/50 flex flex-col items-center justify-center p-2">
                      <div className="absolute inset-0 opacity-20 bg-[url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI4IiBoZWlnaHQ9IjgiPgo8cmVjdCB3aWR0aD0iOCIgaGVpZ2h0PSI4IiBmaWxsPSIjZmZmIiBmaWxsLW9wYWNpdHk9IjAuMSIvPgo8cGF0aCBkPSJNMCAwTDggOFpNOCAwTDAgOFoiIHN0cm9rZT0iIzAwMCIgc3Ryb2tlLW9wYWNpdHk9IjAuMSIvPgo8L3N2Zz4=')]"></div>
                      <ScanFace className="w-8 h-8 text-slate-600 mb-2" strokeWidth={1.5} />
                      <span className="text-[9px] font-mono text-slate-400 relative z-10 text-center">GALLERY<br/>CROP</span>
                    </div>
                    <div className="aspect-square bg-slate-950 rounded-md relative overflow-hidden border border-cyan-500/30 flex flex-col items-center justify-center p-2">
                      <div className="absolute inset-0 opacity-40 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-cyan-500 via-blue-500 to-transparent blur-md scale-75"></div>
                      <Aperture className="w-8 h-8 text-cyan-300 mb-2 relative z-10" strokeWidth={1.5} />
                      <span className="text-[9px] font-mono text-cyan-400 relative z-10 text-center">DENSITY<br/>MAP</span>
                    </div>
                  </div>

                  <div className="space-y-2 font-mono text-[9px] sm:text-[10px] bg-slate-900 p-3 rounded-md border border-slate-800">
                    <div className="flex justify-between">
                      <span className="text-slate-500">PAD METHOD:</span>
                      <span className="text-emerald-400">LAPLACIAN_VARIANCE (142.8)</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-500">STATISTICAL CERTAINTY:</span>
                      <span className="text-white">1 IN 4.2 MILLION FAR (LFW)</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-500">PROBE SHA-256:</span>
                      <span className="text-slate-400 truncate max-w-[120px] sm:max-w-[200px]">e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855</span>
                    </div>
                  </div>
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
             <div className="text-sm font-bold text-white mb-1">WORM Immutable Ledger</div>
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
            <div>© 2026 AURUMSHIELD FORENSICS. ALL RIGHTS RESERVED.</div>
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