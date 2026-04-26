'use client';

import React, { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import SymmetryMerge from '@/components/SymmetryMerge';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

const IdentityGraph = dynamic(() => import('@/components/IdentityGraph'), { ssr: false });

function getApiUrl(): string {
  if (typeof window !== 'undefined' && window.location.hostname.includes('facial-frontend')) {
    return window.location.origin.replace('facial-frontend', 'facial-backend');
  }
  return process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
}

export default function Home() {
  const [probeFile, setProbeFile] = useState<File | null>(null);
  const [probePreview, setProbePreview] = useState<string>('');

  const [step, setStep] = useState<'idle' | 'uploading' | 'frontalizing' | 'calculating' | 'complete' | 'error'>('idle');
  const [errorMsg, setErrorMsg] = useState('');

  interface VerificationResult {
    structural_score: number;
    soft_biometrics_score: number;
    micro_topology_score: number;
    fused_identity_score: number;
    veto_triggered: boolean;
    conclusion: string;
    gallery_heatmap_b64: string;
    probe_heatmap_b64: string;
    gallery_aligned_b64: string;
    probe_aligned_b64: string;
    scar_delta_b64: string;
    gallery_wireframe_b64: string;
    probe_wireframe_b64: string;
  }

  const [results, setResults] = useState<VerificationResult | null>(null);
  const [isXrayMode, setIsXrayMode] = useState(false);
  const [isExporting, setIsExporting] = useState(false);

  const [token, setToken] = useState<string | null>(null);
  const [password, setPassword] = useState('');
  const [loginError, setLoginError] = useState('');
  const [viewMode, setViewMode] = useState<'acquisition' | 'graph'>('acquisition');

  useEffect(() => {
    const savedToken = localStorage.getItem('operator_token');
    if (savedToken) {
      const restore = () => setToken(savedToken);
      queueMicrotask(restore);
    }
  }, []);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoginError('');
    try {
      const res = await fetch(`${getApiUrl()}/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password })
      });
      if (!res.ok) throw new Error('Invalid credentials');
      const data = await res.json();
      setToken(data.access_token);
      localStorage.setItem('operator_token', data.access_token);
    } catch (err: unknown) {
      setLoginError(err instanceof Error ? err.message : 'Login failed');
    }
  };

  const handleLogout = () => {
    setToken(null);
    localStorage.removeItem('operator_token');
    setStep('idle');
    setProbeFile(null);
    if (probePreview) URL.revokeObjectURL(probePreview);
    setProbePreview('');
    setResults(null);
    setIsXrayMode(false);
    setErrorMsg('');
    setLoginError('');
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (probePreview) URL.revokeObjectURL(probePreview);
      setProbeFile(file);
      setProbePreview(URL.createObjectURL(file));
    }
  };

  const startSequence = async () => {
    if (!probeFile) return;
    const uploadContentType = probeFile.type || 'image/jpeg';
    try {
      setStep('uploading');
      
      // 1. Get Pre-Signed URLs from FastAPI (Pass Content-Types dynamically)
      const urlRes = await fetch(`${getApiUrl()}/generate-upload-urls`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          probe_content_type: uploadContentType
        })
      });
      if (urlRes.status === 401) {
        handleLogout();
        return;
      }
      if (!urlRes.ok) {
        const errBody = await urlRes.json().catch(() => null);
        throw new Error(errBody?.detail || 'Failed to secure upload channels.');
      }
      const { probe_upload_url, probe_gs_uri } = await urlRes.json();
      
      // 2. Direct Client Upload to GCS
      const probeUploadRes = await fetch(probe_upload_url, {
        method: 'PUT',
        body: probeFile,
        headers: { 'Content-Type': uploadContentType },
        mode: 'cors'
      });
      if (!probeUploadRes.ok) {
        const errText = await probeUploadRes.text().catch(() => '');
        throw new Error(`Image upload failed (${probeUploadRes.status}): ${errText.slice(0, 200)}`);
      }
      
      // probe_gs_uri is passed directly to the verify call
      
      // UX Pacing
      setStep('frontalizing');
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      setStep('calculating');
      
      // 3. The Verification Call
      const verifyRes = await fetch(`${getApiUrl()}/vault/search`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          probe_url: probe_gs_uri
        })
      });
      
      if (verifyRes.status === 401) {
        handleLogout();
        return;
      }
      if (!verifyRes.ok) {
        const errBody = await verifyRes.json().catch(() => null);
        throw new Error(errBody?.detail || 'Verification pipeline failed.');
      }
      const data = await verifyRes.json();
      
      setResults(data);
      setStep('complete');
      
    } catch (err: unknown) {
      setErrorMsg(err instanceof Error ? err.message : 'An unknown error occurred');
      setStep('error');
    }
  };

  const generateForensicReport = async () => {
    if (!results) return;
    setIsExporting(true);

    try {
      // Build hidden dossier template
      const container = document.createElement('div');
      container.style.cssText = 'position:absolute;left:-9999px;top:0;width:800px;padding:0;margin:0;';
      container.innerHTML = `
        <div style="background:#0A0A0B;color:#E0E0E0;font-family:'Courier New',monospace;padding:48px 40px;width:800px;box-sizing:border-box;">
          <!-- Header -->
          <div style="border-bottom:2px solid #D4AF37;padding-bottom:16px;margin-bottom:24px;">
            <div style="font-size:11px;color:#D4AF37;letter-spacing:6px;margin-bottom:4px;">▓▓ CLASSIFIED ▓▓</div>
            <div style="font-size:22px;font-weight:bold;color:white;letter-spacing:4px;">SCARGODS <span style="color:#D4AF37;">BIOMETRIC INTELLIGENCE</span></div>
            <div style="font-size:10px;color:#666;margin-top:6px;letter-spacing:3px;">FORENSIC VERIFICATION DOSSIER</div>
          </div>

          <!-- Metadata -->
          <div style="display:flex;justify-content:space-between;margin-bottom:24px;border:1px solid #222;padding:12px 16px;background:#0d0d0e;">
            <div>
              <div style="font-size:9px;color:#666;letter-spacing:2px;">REPORT GENERATED</div>
              <div style="font-size:12px;color:white;margin-top:2px;">${new Date().toISOString().replace('T', ' ').slice(0, 19)} UTC</div>
            </div>
            <div style="text-align:right;">
              <div style="font-size:9px;color:#666;letter-spacing:2px;">CLASSIFICATION</div>
              <div style="font-size:12px;color:#D4AF37;margin-top:2px;font-weight:bold;">LEVEL 3 — RESTRICTED</div>
            </div>
          </div>

          <!-- Subject Images -->
          <div style="display:flex;gap:16px;margin-bottom:24px;">
            <div style="flex:1;border:1px solid #333;padding:8px;background:#111;text-align:center;">
              <img src="${results.gallery_aligned_b64}" style="width:100%;height:auto;display:block;" />
              <div style="font-size:9px;color:#666;letter-spacing:3px;margin-top:8px;">GALLERY (REFERENCE)</div>
            </div>
            <div style="flex:1;border:1px solid #333;padding:8px;background:#111;text-align:center;">
              <img src="${results.probe_aligned_b64}" style="width:100%;height:auto;display:block;" />
              <div style="font-size:9px;color:#666;letter-spacing:3px;margin-top:8px;">PROBE (TARGET)</div>
            </div>
          </div>

          <!-- Scoring Grid -->
          <div style="border:1px solid #333;margin-bottom:16px;">
            <div style="display:flex;border-bottom:1px solid #222;">
              <div style="flex:1;padding:14px 16px;border-right:1px solid #222;">
                <div style="font-size:9px;color:#666;letter-spacing:2px;">TIER 1: STRUCTURAL</div>
                <div style="font-size:28px;color:white;font-weight:bold;margin-top:4px;">${results.structural_score}%</div>
              </div>
              <div style="flex:1;padding:14px 16px;border-right:1px solid #222;">
                <div style="font-size:9px;color:#666;letter-spacing:2px;">TIER 2: SOFT BIO</div>
                <div style="font-size:28px;color:white;font-weight:bold;margin-top:4px;">${results.soft_biometrics_score}%</div>
              </div>
              <div style="flex:1;padding:14px 16px;">
                <div style="font-size:9px;color:#666;letter-spacing:2px;">TIER 3: MICRO-TOPO</div>
                <div style="font-size:28px;color:white;font-weight:bold;margin-top:4px;">${results.micro_topology_score}%</div>
              </div>
            </div>
          </div>

          <!-- Fused Score -->
          <div style="border:2px solid #D4AF37;padding:20px 24px;margin-bottom:16px;background:#1a170d;position:relative;">
            <div style="font-size:9px;color:#D4AF37;letter-spacing:3px;">FUSED IDENTITY SCORE</div>
            <div style="font-size:48px;color:#D4AF37;font-weight:bold;margin-top:4px;">${results.fused_identity_score}%</div>
          </div>

          <!-- Conclusion -->
          <div style="border:1px solid ${results.veto_triggered ? '#7f1d1d' : '#333'};padding:16px 20px;background:${results.veto_triggered ? 'rgba(127,29,29,0.15)' : '#0d0d0e'};margin-bottom:24px;">
            <div style="font-size:9px;color:#666;letter-spacing:2px;margin-bottom:6px;">CONCLUSION</div>
            <div style="font-size:14px;color:${results.veto_triggered ? '#f87171' : '#e5e5e5'};font-weight:bold;">${results.conclusion}</div>
            ${results.veto_triggered ? '<div style="margin-top:10px;display:inline-block;padding:4px 12px;background:#7f1d1d;color:#fee2e2;font-size:10px;letter-spacing:3px;border:1px solid #ef4444;">ACE-V VETO TRIGGERED</div>' : '<div style="margin-top:10px;display:inline-block;padding:4px 12px;background:#0a0a0a;color:#22c55e;font-size:10px;letter-spacing:3px;border:1px solid rgba(34,197,94,0.3);">NO DISCREPANCY DETECTED</div>'}
          </div>

          <!-- Footer -->
          <div style="border-top:1px solid #222;padding-top:12px;display:flex;justify-content:space-between;">
            <div style="font-size:8px;color:#444;letter-spacing:2px;">SCARGODS BIOMETRIC INTELLIGENCE DIVISION</div>
            <div style="font-size:8px;color:#444;letter-spacing:2px;">DOCUMENT ID: ${crypto.randomUUID().slice(0, 8).toUpperCase()}</div>
          </div>
        </div>
      `;

      document.body.appendChild(container);

      // Wait for images to settle in the DOM
      await new Promise(resolve => setTimeout(resolve, 300));

      const canvas = await html2canvas(container.firstElementChild as HTMLElement, {
        backgroundColor: '#0A0A0B',
        scale: 2,
        useCORS: true,
        logging: false,
      });

      document.body.removeChild(container);

      // Generate PDF sized to the rendered canvas
      const imgData = canvas.toDataURL('image/png');
      const pdfWidth = canvas.width;
      const pdfHeight = canvas.height;
      const pdf = new jsPDF({
        orientation: pdfHeight > pdfWidth ? 'portrait' : 'landscape',
        unit: 'px',
        format: [pdfWidth, pdfHeight],
      });
      pdf.addImage(imgData, 'PNG', 0, 0, pdfWidth, pdfHeight);

      const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
      pdf.save(`SCARGODS_FORENSIC_DOSSIER_${timestamp}.pdf`);
    } catch (err) {
      console.error('PDF export failed:', err);
    } finally {
      setIsExporting(false);
    }
  };

  // ─── LOGIN SCREEN ─────────────────────────────────────
  if (!token) {
    return (
      <main className="h-screen w-screen overflow-hidden bg-[#0A0A0B] text-[#E0E0E0] flex items-center justify-center font-mono selection:bg-[#D4AF37] selection:text-black">
        <div className="w-full max-w-sm p-8 bg-[#0d0d0e] border border-[#1f1f1f] rounded-xl shadow-2xl">
          <div className="text-center mb-6">
            <h1 className="text-xl font-bold tracking-widest text-white mb-1">
              ZERO-TRUST <span className="text-[#D4AF37]">PORTAL</span>
            </h1>
            <p className="text-gray-500 text-[10px]">AWAITING OPERATOR CREDENTIALS</p>
          </div>
          
          <form onSubmit={handleLogin} className="space-y-4">
            <input 
              type="password" 
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="ENTER PASSPHRASE"
              className="w-full bg-[#111] border border-[#333] text-white p-3 text-center text-sm tracking-widest focus:outline-none focus:border-[#D4AF37] transition-colors"
            />
            
            {loginError && (
              <div className="text-red-500 text-[10px] text-center">{loginError}</div>
            )}
            
            <button 
              type="submit"
              className="w-full py-2.5 bg-[#D4AF37] text-black font-bold text-sm tracking-widest hover:bg-[#b5952f] transition-colors shadow-[0_0_15px_rgba(212,175,55,0.2)]"
            >
              AUTHENTICATE
            </button>
          </form>
        </div>
      </main>
    );
  }

  // ─── MAIN TERMINAL ────────────────────────────────────
  return (
    <main className="h-screen w-screen overflow-hidden bg-[#0A0A0B] text-[#E0E0E0] font-mono selection:bg-[#D4AF37] selection:text-black flex flex-col">
      
      {/* ── Top Bar ── */}
      <header className="shrink-0 flex justify-between items-center px-5 py-2.5 border-b border-[#1a1a1a]">
        <div>
          <h1 className="text-base font-bold tracking-widest text-white leading-tight">
            BIOMETRIC VERIFICATION <span className="text-[#D4AF37]">ENGINE</span>
          </h1>
          <p className="text-gray-600 text-[10px] mt-0.5">Level 3 Topology & 3DMM Frontalization Pipeline</p>
        </div>
        <div className="flex items-center gap-3">
          {/* ── View Mode Toggle ── */}
          <div className="flex border border-[#1f1f1f] rounded overflow-hidden">
            <button
              onClick={() => setViewMode('acquisition')}
              className={`px-3 py-1 text-[10px] tracking-widest font-bold transition-all ${
                viewMode === 'acquisition'
                  ? 'bg-[#D4AF37]/15 text-[#D4AF37] border-r border-[#D4AF37]/30'
                  : 'bg-[#0d0d0e] text-gray-600 hover:text-gray-400 border-r border-[#1f1f1f]'
              }`}
            >
              ACQUISITION
            </button>
            <button
              onClick={() => setViewMode('graph')}
              className={`px-3 py-1 text-[10px] tracking-widest font-bold transition-all ${
                viewMode === 'graph'
                  ? 'bg-[#D4AF37]/15 text-[#D4AF37]'
                  : 'bg-[#0d0d0e] text-gray-600 hover:text-gray-400'
              }`}
            >
              GRAPH
            </button>
          </div>
          <button 
            onClick={handleLogout}
            className="text-[10px] text-gray-500 hover:text-red-400 transition-colors border border-transparent hover:border-red-900/50 px-2.5 py-1 rounded"
          >
            END SESSION
          </button>
        </div>
      </header>

      {/* ── Body ── */}
      {viewMode === 'graph' ? (
        <div className="flex-1 min-h-0">
          <IdentityGraph />
        </div>
      ) : (
      <div className="flex-1 min-h-0 p-4">

        {/* ════ IDLE: Upload Panel ════ */}
        {step === 'idle' && (
          <div className="h-full flex flex-col items-center justify-center gap-6 w-full">
            <div className="w-full max-w-xl">
              {/* Single Target Upload Dropzone */}
              <div className="border border-dashed border-[#D4AF37]/50 rounded-lg p-8 flex flex-col items-center justify-center bg-[#0d0d0e] hover:border-[#D4AF37] hover:bg-[#111] transition-all relative min-h-[300px]">
                <input 
                  type="file" accept="image/*" 
                  onChange={handleFileChange}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                />
                
                {/* Crosshair aesthetic elements */}
                <div className="absolute top-4 left-4 w-4 h-4 border-t border-l border-[#D4AF37]/50"></div>
                <div className="absolute top-4 right-4 w-4 h-4 border-t border-r border-[#D4AF37]/50"></div>
                <div className="absolute bottom-4 left-4 w-4 h-4 border-b border-l border-[#D4AF37]/50"></div>
                <div className="absolute bottom-4 right-4 w-4 h-4 border-b border-r border-[#D4AF37]/50"></div>

                {probePreview ? (
                  /* eslint-disable-next-line @next/next/no-img-element */
                  <img src={probePreview} alt="Target" className="max-h-[220px] object-contain rounded shadow-[0_0_20px_rgba(212,175,55,0.15)] z-0 relative" />
                ) : (
                  <div className="flex flex-col items-center text-[#D4AF37]/80">
                    <div className="w-16 h-16 rounded-full border-2 border-dotted border-[#D4AF37] flex items-center justify-center mb-4 relative animate-[spin_15s_linear_infinite]">
                      <svg className="w-6 h-6 animate-none" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M12 4v16m8-8H4"></path></svg>
                    </div>
                    <span className="text-[#D4AF37] font-bold text-lg tracking-[0.2em] mb-1">ACQUIRE UNKNOWN TARGET</span>
                    <span className="text-gray-500 text-xs tracking-widest">DRAG AND DROP OR CLICK TO UPLOAD</span>
                  </div>
                )}
              </div>
            </div>

            <button 
              onClick={startSequence}
              disabled={!probeFile}
              className={`px-8 py-3 text-sm font-bold tracking-widest rounded-sm border transition-all ${probeFile ? 'bg-[#D4AF37] text-black border-[#D4AF37] hover:bg-[#b5952f] shadow-[0_0_20px_rgba(212,175,55,0.3)]' : 'bg-[#111] text-gray-500 border-[#333] cursor-not-allowed'}`}
            >
              INITIATE VAULT SWEEP
            </button>
          </div>
        )}

        {/* ════ LOADING ════ */}
        {['uploading', 'frontalizing', 'calculating'].includes(step) && (
          <div className="h-full flex flex-col items-center justify-center">
            <div className="w-12 h-12 border-4 border-[#333] border-t-[#D4AF37] rounded-full animate-spin mb-4"></div>
            <p className="text-sm text-gray-300 tracking-widest animate-pulse">
              {step === 'uploading' && "UPLOADING TARGET..."}
              {step === 'frontalizing' && "SCANNING ENCRYPTED VAULT..."}
              {step === 'calculating' && "DECRYPTING IDENTITY VECTORS..."}
            </p>
          </div>
        )}

        {/* ════ ERROR ════ */}
        {step === 'error' && (
          <div className="h-full flex flex-col items-center justify-center">
            <div className="p-5 border border-red-900 bg-red-950/30 text-red-400 rounded-lg text-center max-w-lg">
              <p className="font-bold text-sm mb-1">SYSTEM ERROR</p>
              <p className="text-xs">{errorMsg}</p>
              <button onClick={() => setStep('idle')} className="mt-3 px-4 py-1.5 text-xs bg-red-900/50 hover:bg-red-900/80 rounded border border-red-700">RESET</button>
            </div>
          </div>
        )}

        {/* ════ RESULTS DASHBOARD ════ */}
        {step === 'complete' && results && (
          <div className="h-full grid grid-cols-[1fr_auto] gap-4 min-h-0">

            {/* ── Left: SymmetryMerge (takes all available height) ── */}
            <div className="flex flex-col min-h-0 min-w-0">
              {/* X-Ray toggle */}
              <div className="flex justify-between items-center mb-2 shrink-0">
                <button
                  onClick={() => setIsXrayMode(!isXrayMode)}
                  className={`flex items-center gap-2 px-3 py-1 border rounded text-[10px] tracking-widest transition-all ${
                    isXrayMode 
                      ? 'border-[#D4AF37] bg-[#D4AF37]/10 text-[#D4AF37] shadow-[0_0_10px_rgba(212,175,55,0.2)]' 
                      : 'border-[#333] bg-[#111] text-gray-500 hover:text-gray-300'
                  }`}
                >
                  <div className={`w-2 h-2 rounded-full ${isXrayMode ? 'bg-[#D4AF37] animate-pulse' : 'bg-gray-600'}`}></div>
                  X-RAY
                </button>
                <button 
                  onClick={() => {
                    setStep('idle');
                    setProbeFile(null);
                    if (probePreview) URL.revokeObjectURL(probePreview);
                    setProbePreview('');
                    setIsXrayMode(false);
                    setResults(null);
                  }}
                  className="text-[10px] text-gray-500 hover:text-white border border-[#333] hover:border-gray-500 px-3 py-1 rounded tracking-widest transition-colors"
                >
                  NEW RUN
                </button>
              </div>

              {/* Canvas fill */}
              <div className="flex-1 min-h-0 bg-[#0d0d0e] border border-[#1f1f1f] rounded-lg p-2">
                <SymmetryMerge 
                  galleryImageSrc={isXrayMode ? results.gallery_heatmap_b64 : results.gallery_aligned_b64} 
                  probeImageSrc={isXrayMode ? results.probe_heatmap_b64 : results.probe_aligned_b64}
                  deltaImageSrc={results.scar_delta_b64}
                  galleryWireframeSrc={results.gallery_wireframe_b64}
                  probeWireframeSrc={results.probe_wireframe_b64}
                />
              </div>
            </div>

            {/* ── Right: Scoring Panel ── */}
            <div className="w-52 flex flex-col gap-2 min-h-0 shrink-0">
              {/* Tier Cards */}
              <div className="border border-[#1f1f1f] bg-[#0d0d0e] rounded-lg p-3">
                <h3 className="text-gray-500 text-[9px] mb-1 tracking-wider">TIER 1: STRUCTURAL</h3>
                <div className="text-2xl text-white font-bold">{results.structural_score}%</div>
              </div>
              <div className="border border-[#1f1f1f] bg-[#0d0d0e] rounded-lg p-3">
                <h3 className="text-gray-500 text-[9px] mb-1 tracking-wider">TIER 2: SOFT BIO</h3>
                <div className="text-2xl text-white font-bold">{results.soft_biometrics_score}%</div>
              </div>
              <div className="border border-[#1f1f1f] bg-[#0d0d0e] rounded-lg p-3">
                <h3 className="text-gray-500 text-[9px] mb-1 tracking-wider">TIER 3: MICRO-TOPO</h3>
                <div className="text-2xl text-white font-bold">{results.micro_topology_score}%</div>
              </div>
              
              {/* Fused Score — Hero */}
              <div className="border border-[#D4AF37]/40 bg-[#1a170d] rounded-lg p-3 relative overflow-hidden">
                <div className="absolute top-0 right-0 w-10 h-10 bg-[#D4AF37]/10 rounded-bl-full"></div>
                <h3 className="text-[#D4AF37] text-[9px] mb-1 tracking-wider">FUSED IDENTITY</h3>
                <div className="text-3xl text-[#D4AF37] font-bold">{results.fused_identity_score}%</div>
              </div>

              {/* Conclusion */}
              <div className={`flex-1 border rounded-lg p-3 flex flex-col justify-between ${results.veto_triggered ? 'border-red-900 bg-red-950/30' : 'border-[#1f1f1f] bg-[#0d0d0e]'}`}>
                <div>
                  <h3 className="text-gray-500 text-[9px] mb-1 tracking-wider">CONCLUSION</h3>
                  <p className={`text-xs leading-snug ${results.veto_triggered ? 'text-red-400 font-bold' : 'text-gray-200'}`}>
                    {results.conclusion}
                  </p>
                </div>
                <div className="mt-2">
                  {results.veto_triggered ? (
                    <div className="px-2 py-1 bg-red-900 text-red-100 text-[9px] tracking-widest border border-red-500 text-center rounded">
                      ACE-V VETO
                    </div>
                  ) : (
                    <div className="px-2 py-1 bg-[#0a0a0a] text-green-500 text-[9px] tracking-widest border border-green-900/50 text-center rounded">
                      NO DISCREPANCY
                    </div>
                  )}
                </div>
              </div>

              {/* Export Dossier */}
              <button
                onClick={generateForensicReport}
                disabled={isExporting}
                className={`w-full py-2.5 text-[10px] font-bold tracking-widest border-2 rounded transition-all ${
                  isExporting
                    ? 'border-[#333] bg-[#111] text-gray-500 cursor-wait'
                    : 'border-[#D4AF37] bg-[#0a0a0a] text-[#D4AF37] hover:bg-[#D4AF37]/10 hover:shadow-[0_0_15px_rgba(212,175,55,0.4)]'
                }`}
              >
                {isExporting ? 'COMPILING REPORT...' : 'EXPORT CLASSIFIED DOSSIER'}
              </button>
            </div>

          </div>
        )}

      </div>
      )}
    </main>
  );
}

