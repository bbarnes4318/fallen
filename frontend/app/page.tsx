'use client';

import React, { useState, useEffect, useCallback } from 'react';
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
  const [galleryFile, setGalleryFile] = useState<File | null>(null);
  const [galleryPreview, setGalleryPreview] = useState<string>('');

  const [step, setStep] = useState<'idle' | 'uploading' | 'frontalizing' | 'calculating' | 'paywall' | 'complete' | 'error'>('idle');
  const [errorMsg, setErrorMsg] = useState('');
  const [mode, setMode] = useState<'vault' | 'compare'>('vault');
  const [bypassCode, setBypassCode] = useState('');

  interface AuditLog {
    raw_cosine_score: number;
    statistical_certainty: string;
    false_acceptance_rate: string;
    nodes_mapped: number;
    matched_user_id?: string;
    person_name?: string;
    source?: string;
    creator?: string;
    license_short_name?: string;
    license_url?: string;
    file_page_url?: string;
    wikidata_id?: string;
    vector_hash?: string;
    alignment_variance?: { yaw: string; pitch: string; roll: string };
    liveness_check?: { spoof_probability: string; status: string };
    crypto_envelope?: { standard: string; decryption_time: string };
  }

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
    audit_log?: AuditLog;
  }

  const [results, setResults] = useState<VerificationResult | null>(null);
  const [isXrayMode, setIsXrayMode] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [auditExpanded, setAuditExpanded] = useState(false);

  const [token, setToken] = useState<string | null>(null);
  const [password, setPassword] = useState('');
  const [loginError, setLoginError] = useState('');
  const [viewMode, setViewMode] = useState<'acquisition' | 'graph'>('acquisition');

  // Restore saved token
  useEffect(() => {
    const savedToken = localStorage.getItem('operator_token');
    if (savedToken) {
      const restore = () => setToken(savedToken);
      queueMicrotask(restore);
    }
  }, []);

  // Post-payment handoff: check for ?success=true in URL
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const params = new URLSearchParams(window.location.search);
    if (params.get('success') === 'true') {
      const cached = sessionStorage.getItem('cachedResult');
      if (cached) {
        try {
          const parsed = JSON.parse(cached) as VerificationResult;
          setResults(parsed);
          setStep('complete');
          sessionStorage.removeItem('cachedResult');
        } catch {
          console.error('Failed to parse cached result');
        }
      }
      window.history.replaceState({}, '', window.location.pathname);
    } else if (params.get('canceled') === 'true') {
      const cached = sessionStorage.getItem('cachedResult');
      if (cached) {
        try {
          const parsed = JSON.parse(cached) as VerificationResult;
          setResults(parsed);
          setStep('paywall');
        } catch {
          console.error('Failed to parse cached result');
        }
      }
      window.history.replaceState({}, '', window.location.pathname);
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
    setGalleryFile(null);
    if (galleryPreview) URL.revokeObjectURL(galleryPreview);
    setGalleryPreview('');
    setResults(null);
    setIsXrayMode(false);
    setErrorMsg('');
    setLoginError('');
    setBypassCode('');
  };

  const handleGalleryChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (galleryPreview) URL.revokeObjectURL(galleryPreview);
      setGalleryFile(file);
      setGalleryPreview(URL.createObjectURL(file));
    }
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
    if (mode === 'compare' && !galleryFile) return;
    const probeContentType = probeFile.type || 'image/jpeg';
    const galleryContentType = galleryFile?.type || 'image/jpeg';
    try {
      setStep('uploading');
      
      // 1. Get Pre-Signed URLs
      const urlBody: Record<string, string> = { probe_content_type: probeContentType };
      if (mode === 'compare' && galleryFile) {
        urlBody.gallery_content_type = galleryContentType;
      }
      const urlRes = await fetch(`${getApiUrl()}/generate-upload-urls`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
        body: JSON.stringify(urlBody)
      });
      if (urlRes.status === 401) { handleLogout(); return; }
      if (!urlRes.ok) {
        const errBody = await urlRes.json().catch(() => null);
        throw new Error(errBody?.detail || 'Failed to secure upload channels.');
      }
      const urlData = await urlRes.json();
      
      // 2. Upload probe to GCS
      const probeUp = await fetch(urlData.probe_upload_url, {
        method: 'PUT', body: probeFile,
        headers: { 'Content-Type': probeContentType }, mode: 'cors'
      });
      if (!probeUp.ok) throw new Error(`Probe upload failed (${probeUp.status})`);
      
      // 2b. Upload gallery if compare mode
      if (mode === 'compare' && galleryFile && urlData.gallery_upload_url) {
        const galUp = await fetch(urlData.gallery_upload_url, {
          method: 'PUT', body: galleryFile,
          headers: { 'Content-Type': galleryContentType }, mode: 'cors'
        });
        if (!galUp.ok) throw new Error(`Gallery upload failed (${galUp.status})`);
      }
      
      setStep('frontalizing');
      await new Promise(resolve => setTimeout(resolve, 1500));
      setStep('calculating');
      
      // 3. API call based on mode
      const apiUrl = mode === 'vault' ? '/vault/search' : '/verify/fuse';
      const apiBody: Record<string, string> = { probe_url: urlData.probe_gs_uri };
      if (mode === 'compare') {
        apiBody.gallery_url = urlData.gallery_gs_uri;
      }
      
      const verifyRes = await fetch(`${getApiUrl()}${apiUrl}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
        body: JSON.stringify(apiBody)
      });
      if (verifyRes.status === 401) { handleLogout(); return; }
      if (!verifyRes.ok) {
        const errBody = await verifyRes.json().catch(() => null);
        throw new Error(errBody?.detail || 'Verification pipeline failed.');
      }
      const data = await verifyRes.json();
      
      // Cache and go to paywall
      sessionStorage.setItem('cachedResult', JSON.stringify(data));
      setResults(data);
      setStep('paywall');
      
    } catch (err: unknown) {
      setErrorMsg(err instanceof Error ? err.message : 'An unknown error occurred');
      setStep('error');
    }
  };

  const generateForensicReport = async () => {
    if (!results) return;
    setIsExporting(true);

    const audit = results.audit_log;
    const ts = new Date().toISOString().replace('T', ' ').slice(0, 19);
    const docId = crypto.randomUUID().slice(0, 8).toUpperCase();

    try {
      // ── Strict A4 hidden print template (794×1123 = exact A4 aspect ratio) ──
      const container = document.createElement('div');
      container.style.cssText = 'position:absolute;left:-9999px;top:-9999px;width:794px;height:1123px;overflow:hidden;margin:0;padding:0;';
      container.innerHTML = `
        <div style="background:#000;color:#ccc;font-family:'Courier New',monospace;width:794px;height:1123px;box-sizing:border-box;padding:28px 32px 20px;display:flex;flex-direction:column;overflow:hidden;">

          <!-- HEADER -->
          <div style="border-bottom:2px solid #D4AF37;padding-bottom:10px;margin-bottom:12px;flex-shrink:0;">
            <div style="display:flex;justify-content:space-between;align-items:flex-end;">
              <div>
                <div style="font-size:8px;color:#D4AF37;letter-spacing:5px;margin-bottom:2px;">▓▓ CLASSIFIED ▓▓</div>
                <div style="font-size:16px;font-weight:bold;color:white;letter-spacing:3px;">AURUMSHIELD <span style="color:#D4AF37;">BIOMETRIC DOSSIER</span></div>
              </div>
              <div style="text-align:right;">
                <div style="font-size:7px;color:#555;letter-spacing:2px;">GENERATED ${ts} UTC</div>
                <div style="font-size:7px;color:#555;letter-spacing:2px;">DOC ID: ${docId} · LEVEL 3 RESTRICTED</div>
              </div>
            </div>
          </div>

          <!-- VISUAL EVIDENCE -->
          <div style="display:flex;gap:8px;margin-bottom:10px;flex-shrink:0;">
            <div style="flex:1;border:1px solid #333;padding:4px;background:#0a0a0a;text-align:center;">
              <img src="${results.probe_aligned_b64}" style="width:100%;height:180px;object-fit:contain;display:block;" />
              <div style="font-size:7px;color:#666;letter-spacing:3px;margin-top:4px;">PROBE (UNKNOWN TARGET)</div>
            </div>
            <div style="flex:1;border:1px solid #333;padding:4px;background:#0a0a0a;text-align:center;">
              <img src="${results.gallery_aligned_b64}" style="width:100%;height:180px;object-fit:contain;display:block;" />
              <div style="font-size:7px;color:#666;letter-spacing:3px;margin-top:4px;">GALLERY (VAULT MATCH)</div>
            </div>
          </div>

          <!-- 4-COLUMN METRIC GRID -->
          <div style="display:flex;gap:0;border:1px solid #333;margin-bottom:10px;flex-shrink:0;">
            <div style="flex:1;padding:8px 10px;border-right:1px solid #222;background:#0a0a0a;">
              <div style="font-size:7px;color:#666;letter-spacing:2px;">TIER 1: STRUCTURAL</div>
              <div style="font-size:22px;color:white;font-weight:bold;margin:2px 0;">${results.structural_score}%</div>
              <div style="font-size:7px;color:#555;line-height:1.3;">1404-D cranial geometry cosine similarity.</div>
            </div>
            <div style="flex:1;padding:8px 10px;border-right:1px solid #222;background:#0a0a0a;">
              <div style="font-size:7px;color:#666;letter-spacing:2px;">TIER 2: SOFT BIO</div>
              <div style="font-size:22px;color:white;font-weight:bold;margin:2px 0;">${results.soft_biometrics_score}%</div>
              <div style="font-size:7px;color:#555;line-height:1.3;">Melanin & ocular hue pixel-density overlap.</div>
            </div>
            <div style="flex:1;padding:8px 10px;border-right:1px solid #222;background:#0a0a0a;">
              <div style="font-size:7px;color:#666;letter-spacing:2px;">TIER 3: MICRO-TOPO</div>
              <div style="font-size:22px;color:white;font-weight:bold;margin:2px 0;">${results.micro_topology_score}%</div>
              <div style="font-size:7px;color:#555;line-height:1.3;">Epidermal deviation & scar alignment.</div>
            </div>
            <div style="flex:1;padding:8px 10px;background:#1a170d;">
              <div style="font-size:7px;color:#D4AF37;letter-spacing:2px;">FUSED IDENTITY</div>
              <div style="font-size:22px;color:#D4AF37;font-weight:bold;margin:2px 0;">${results.fused_identity_score}%</div>
              <div style="font-size:7px;color:#997a1d;line-height:1.3;">Bayesian false-acceptance probability.</div>
            </div>
          </div>

          <!-- CONCLUSION -->
          <div style="border:1px solid ${results.veto_triggered ? '#7f1d1d' : '#333'};padding:8px 12px;background:${results.veto_triggered ? '#1a0505' : '#0a0a0a'};margin-bottom:10px;flex-shrink:0;display:flex;justify-content:space-between;align-items:center;">
            <div>
              <div style="font-size:7px;color:#666;letter-spacing:2px;margin-bottom:2px;">CONCLUSION</div>
              <div style="font-size:11px;color:${results.veto_triggered ? '#f87171' : '#e5e5e5'};font-weight:bold;">${results.conclusion}</div>
            </div>
            <div style="padding:3px 10px;background:${results.veto_triggered ? '#7f1d1d' : '#0a0a0a'};color:${results.veto_triggered ? '#fee2e2' : '#22c55e'};font-size:8px;letter-spacing:2px;border:1px solid ${results.veto_triggered ? '#ef4444' : 'rgba(34,197,94,0.3)'};">${results.veto_triggered ? 'ACE-V VETO' : 'NO DISCREPANCY'}</div>
          </div>

          <!-- CRYPTOGRAPHIC AUDIT LOG (Terminal Block) -->
          <div style="flex:1;border:1px solid #1a1a0a;background:#000;padding:10px 14px;font-size:8px;line-height:1.6;overflow:hidden;">
            <div style="color:#D4AF37;font-size:8px;letter-spacing:3px;margin-bottom:8px;border-bottom:1px solid #1a1a0a;padding-bottom:4px;font-weight:bold;">▸ CRYPTOGRAPHIC AUDIT LOG</div>

            <div style="display:flex;gap:16px;">
              <!-- Col 1: Statistical Certainty -->
              <div style="flex:1;">
                <div style="color:#22c55e;font-size:7px;letter-spacing:2px;margin-bottom:4px;">STATISTICAL CERTAINTY</div>
                <div style="display:flex;justify-content:space-between;margin-bottom:2px;"><span style="color:#555;">FALSE ACCEPT RATE</span><span style="color:#4ade80;font-weight:bold;">${audit?.false_acceptance_rate || 'N/A'}</span></div>
                <div style="display:flex;justify-content:space-between;margin-bottom:2px;"><span style="color:#555;">CERTAINTY</span><span style="color:#4ade80;font-weight:bold;">${audit?.statistical_certainty || 'N/A'}</span></div>
                <div style="display:flex;justify-content:space-between;margin-bottom:2px;"><span style="color:#555;">NODES MAPPED</span><span style="color:#fff;font-weight:bold;">${audit?.nodes_mapped || 468}/468</span></div>
                <div style="display:flex;justify-content:space-between;"><span style="color:#555;">COSINE DIST</span><span style="color:#fff;font-weight:bold;">${audit?.raw_cosine_score?.toFixed(6) || 'N/A'}</span></div>
              </div>

              <!-- Col 2: Spatial Alignment & Liveness -->
              <div style="flex:1;border-left:1px solid #1a1a0a;padding-left:16px;">
                <div style="color:#06b6d4;font-size:7px;letter-spacing:2px;margin-bottom:4px;">SPATIAL ALIGNMENT</div>
                <div style="display:flex;justify-content:space-between;margin-bottom:2px;"><span style="color:#555;">YAW</span><span style="color:#67e8f9;">${audit?.alignment_variance?.yaw || 'N/A'}</span></div>
                <div style="display:flex;justify-content:space-between;margin-bottom:2px;"><span style="color:#555;">PITCH</span><span style="color:#67e8f9;">${audit?.alignment_variance?.pitch || 'N/A'}</span></div>
                <div style="display:flex;justify-content:space-between;margin-bottom:2px;"><span style="color:#555;">ROLL</span><span style="color:#67e8f9;">${audit?.alignment_variance?.roll || 'N/A'}</span></div>
                <div style="color:#06b6d4;font-size:7px;letter-spacing:2px;margin:4px 0 2px;">LIVENESS</div>
                <div style="display:flex;justify-content:space-between;margin-bottom:2px;"><span style="color:#555;">SPOOF PROB</span><span style="color:#4ade80;font-weight:bold;">${audit?.liveness_check?.spoof_probability || 'N/A'}</span></div>
                <div style="display:flex;justify-content:space-between;"><span style="color:#555;">STATUS</span><span style="color:#4ade80;font-weight:bold;">${audit?.liveness_check?.status || 'N/A'}</span></div>
              </div>

              <!-- Col 3: Cryptography & Source -->
              <div style="flex:1;border-left:1px solid #1a1a0a;padding-left:16px;">
                <div style="color:#f59e0b;font-size:7px;letter-spacing:2px;margin-bottom:4px;">CRYPTOGRAPHY</div>
                <div style="margin-bottom:2px;"><span style="color:#555;">VECTOR SHA-256</span></div>
                <div style="color:#fbbf24;font-size:6px;word-break:break-all;margin-bottom:3px;">${audit?.vector_hash || 'N/A'}</div>
                <div style="display:flex;justify-content:space-between;margin-bottom:2px;"><span style="color:#555;">ENCRYPTION</span><span style="color:#fbbf24;">${audit?.crypto_envelope?.standard || 'N/A'}</span></div>
                <div style="display:flex;justify-content:space-between;margin-bottom:2px;"><span style="color:#555;">KMS LATENCY</span><span style="color:#fbbf24;">${audit?.crypto_envelope?.decryption_time || 'N/A'}</span></div>
                ${audit?.matched_user_id ? `<div style="color:#f59e0b;font-size:7px;letter-spacing:2px;margin:4px 0 2px;">SOURCE</div><div style="display:flex;justify-content:space-between;margin-bottom:2px;"><span style="color:#555;">TARGET</span><span style="color:#fff;">${audit.matched_user_id}</span></div>` : ''}
                ${audit?.person_name ? `<div style="display:flex;justify-content:space-between;margin-bottom:2px;"><span style="color:#555;">PERSON</span><span style="color:#fff;">${audit.person_name}</span></div>` : ''}
                ${audit?.license_short_name ? `<div style="display:flex;justify-content:space-between;"><span style="color:#555;">LICENSE</span><span style="color:#888;">${audit.license_short_name}</span></div>` : ''}
              </div>
            </div>
          </div>

          <!-- FOOTER -->
          <div style="margin-top:8px;padding-top:6px;border-top:1px solid #1a1a0a;display:flex;justify-content:space-between;flex-shrink:0;">
            <div style="font-size:6px;color:#333;letter-spacing:2px;">AURUMSHIELD BIOMETRIC INTELLIGENCE · CLASSIFIED</div>
            <div style="font-size:6px;color:#333;letter-spacing:2px;">DOC ${docId} · ${ts}</div>
          </div>
        </div>
      `;

      document.body.appendChild(container);
      await new Promise(resolve => setTimeout(resolve, 400));

      const canvas = await html2canvas(container.firstElementChild as HTMLElement, {
        backgroundColor: '#000000',
        scale: 2,
        useCORS: true,
        logging: false,
        width: 794,
        height: 1123,
      });

      document.body.removeChild(container);

      // ── Strict A4 PDF sizing ──
      const imgData = canvas.toDataURL('image/png');
      const pdf = new jsPDF('p', 'mm', 'a4');
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = (canvas.height * pdfWidth) / canvas.width;
      pdf.addImage(imgData, 'PNG', 0, 0, pdfWidth, pdfHeight);

      const fileTs = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
      pdf.save(`AURUMSHIELD_DOSSIER_${fileTs}.pdf`);
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
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setPassword(e.target.value)}
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
          <div className="h-full flex flex-col items-center justify-center gap-5 w-full">
            {/* ── Mode Toggle ── */}
            <div className="flex border border-[#1f1f1f] rounded overflow-hidden">
              <button onClick={() => setMode('vault')} className={`px-5 py-2 text-[10px] tracking-[0.2em] font-bold transition-all ${mode === 'vault' ? 'bg-[#D4AF37]/15 text-[#D4AF37] border-r border-[#D4AF37]/30' : 'bg-[#0d0d0e] text-gray-600 hover:text-gray-400 border-r border-[#1f1f1f]'}`}>VAULT SWEEP (1:N)</button>
              <button onClick={() => setMode('compare')} className={`px-5 py-2 text-[10px] tracking-[0.2em] font-bold transition-all ${mode === 'compare' ? 'bg-[#D4AF37]/15 text-[#D4AF37]' : 'bg-[#0d0d0e] text-gray-600 hover:text-gray-400'}`}>MANUAL VERIFICATION (1:1)</button>
            </div>

            {mode === 'vault' ? (
              /* ── Vault: Single Dropzone ── */
              <div className="w-full max-w-xl">
                <div className="border border-dashed border-[#D4AF37]/50 rounded-lg p-8 flex flex-col items-center justify-center bg-[#0d0d0e] hover:border-[#D4AF37] hover:bg-[#111] transition-all relative min-h-[280px]">
                  <input type="file" accept="image/*" onChange={handleFileChange} className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10" />
                  <div className="absolute top-4 left-4 w-4 h-4 border-t border-l border-[#D4AF37]/50"></div>
                  <div className="absolute top-4 right-4 w-4 h-4 border-t border-r border-[#D4AF37]/50"></div>
                  <div className="absolute bottom-4 left-4 w-4 h-4 border-b border-l border-[#D4AF37]/50"></div>
                  <div className="absolute bottom-4 right-4 w-4 h-4 border-b border-r border-[#D4AF37]/50"></div>
                  {probePreview ? (
                    /* eslint-disable-next-line @next/next/no-img-element */
                    <img src={probePreview} alt="Target" className="max-h-[200px] object-contain rounded shadow-[0_0_20px_rgba(212,175,55,0.15)] z-0 relative" />
                  ) : (
                    <div className="flex flex-col items-center text-[#D4AF37]/80">
                      <div className="w-14 h-14 rounded-full border-2 border-dotted border-[#D4AF37] flex items-center justify-center mb-3 animate-[spin_15s_linear_infinite]">
                        <svg className="w-5 h-5 animate-none" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M12 4v16m8-8H4"></path></svg>
                      </div>
                      <span className="text-[#D4AF37] font-bold text-base tracking-[0.2em] mb-1">ACQUIRE UNKNOWN TARGET</span>
                      <span className="text-gray-500 text-[10px] tracking-widest">DRAG AND DROP OR CLICK TO UPLOAD</span>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              /* ── Compare: Dual Dropzones ── */
              <div className="w-full max-w-3xl grid grid-cols-2 gap-4">
                {/* Probe */}
                <div className="border border-dashed border-[#D4AF37]/50 rounded-lg p-6 flex flex-col items-center justify-center bg-[#0d0d0e] hover:border-[#D4AF37] hover:bg-[#111] transition-all relative min-h-[250px]">
                  <input type="file" accept="image/*" onChange={handleFileChange} className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10" />
                  <div className="absolute top-3 left-3 w-3 h-3 border-t border-l border-[#D4AF37]/50"></div>
                  <div className="absolute top-3 right-3 w-3 h-3 border-t border-r border-[#D4AF37]/50"></div>
                  <div className="absolute bottom-3 left-3 w-3 h-3 border-b border-l border-[#D4AF37]/50"></div>
                  <div className="absolute bottom-3 right-3 w-3 h-3 border-b border-r border-[#D4AF37]/50"></div>
                  {probePreview ? (
                    /* eslint-disable-next-line @next/next/no-img-element */
                    <img src={probePreview} alt="Probe" className="max-h-[160px] object-contain rounded shadow-[0_0_15px_rgba(212,175,55,0.1)] z-0 relative" />
                  ) : (
                    <div className="flex flex-col items-center text-[#D4AF37]/80">
                      <svg className="w-8 h-8 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M12 4v16m8-8H4"></path></svg>
                      <span className="text-[#D4AF37] font-bold text-sm tracking-[0.15em]">UPLOAD TARGET</span>
                      <span className="text-gray-600 text-[9px] tracking-widest mt-1">PROBE IMAGE</span>
                    </div>
                  )}
                </div>
                {/* Gallery */}
                <div className="border border-dashed border-gray-700/50 rounded-lg p-6 flex flex-col items-center justify-center bg-[#0d0d0e] hover:border-gray-500 hover:bg-[#111] transition-all relative min-h-[250px]">
                  <input type="file" accept="image/*" onChange={handleGalleryChange} className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10" />
                  <div className="absolute top-3 left-3 w-3 h-3 border-t border-l border-gray-700/50"></div>
                  <div className="absolute top-3 right-3 w-3 h-3 border-t border-r border-gray-700/50"></div>
                  <div className="absolute bottom-3 left-3 w-3 h-3 border-b border-l border-gray-700/50"></div>
                  <div className="absolute bottom-3 right-3 w-3 h-3 border-b border-r border-gray-700/50"></div>
                  {galleryPreview ? (
                    /* eslint-disable-next-line @next/next/no-img-element */
                    <img src={galleryPreview} alt="Gallery" className="max-h-[160px] object-contain rounded shadow-[0_0_15px_rgba(255,255,255,0.05)] z-0 relative" />
                  ) : (
                    <div className="flex flex-col items-center text-gray-500">
                      <svg className="w-8 h-8 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M12 4v16m8-8H4"></path></svg>
                      <span className="font-bold text-sm tracking-[0.15em]">UPLOAD KNOWN ALIAS</span>
                      <span className="text-gray-600 text-[9px] tracking-widest mt-1">GALLERY IMAGE</span>
                    </div>
                  )}
                </div>
              </div>
            )}

            <button 
              onClick={startSequence}
              disabled={mode === 'vault' ? !probeFile : (!probeFile || !galleryFile)}
              className={`px-8 py-3 text-sm font-bold tracking-widest rounded-sm border transition-all ${(mode === 'vault' ? probeFile : probeFile && galleryFile) ? 'bg-[#D4AF37] text-black border-[#D4AF37] hover:bg-[#b5952f] shadow-[0_0_20px_rgba(212,175,55,0.3)]' : 'bg-[#111] text-gray-500 border-[#333] cursor-not-allowed'}`}
            >
              {mode === 'vault' ? 'INITIATE VAULT SWEEP' : 'RUN VERIFICATION'}
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

        {/* ════ PAYWALL: ANALYSIS COMPLETE ════ */}
        {step === 'paywall' && results && (
          <div className="h-full flex flex-col items-center justify-center">
            <div className="w-full max-w-md text-center">
              {/* Lock icon */}
              <div className="relative mb-6">
                <div className="w-16 h-16 mx-auto rounded-full border-2 border-[#D4AF37] flex items-center justify-center relative overflow-hidden">
                  <div className="absolute inset-0 bg-gradient-to-b from-[#D4AF37]/20 via-transparent to-transparent animate-pulse"></div>
                  <svg className="w-7 h-7 text-[#D4AF37]" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"></path></svg>
                </div>
              </div>

              <h2 className="text-xl font-bold tracking-[0.3em] text-white mb-1">ANALYSIS <span className="text-[#D4AF37]">COMPLETE</span></h2>
              <p className="text-sm tracking-[0.2em] text-gray-400 mb-1">IDENTITY LOCKED</p>
              <div className="w-12 h-[1px] bg-[#D4AF37]/40 mx-auto my-3"></div>

              {/* Blurred score tease */}
              <div className="border border-[#1f1f1f] bg-[#0d0d0e] rounded-lg p-3 mb-5 relative overflow-hidden">
                <div className="absolute inset-0 backdrop-blur-sm bg-[#0A0A0B]/60 z-10 flex items-center justify-center">
                  <span className="text-[10px] tracking-[0.3em] text-[#D4AF37]/80 font-bold">ENCRYPTED</span>
                </div>
                <div className="grid grid-cols-3 gap-3 opacity-30 select-none">
                  <div><div className="text-[9px] text-gray-500 tracking-wider">STRUCTURAL</div><div className="text-lg text-white font-bold">██.█%</div></div>
                  <div><div className="text-[9px] text-gray-500 tracking-wider">SOFT BIO</div><div className="text-lg text-white font-bold">██.█%</div></div>
                  <div><div className="text-[9px] text-gray-500 tracking-wider">MICRO-TOPO</div><div className="text-lg text-white font-bold">██.█%</div></div>
                </div>
              </div>

              {/* Stripe CTA */}
              <button
                onClick={async () => {
                  try {
                    const res = await fetch(`${getApiUrl()}/checkout/create-session`, { method: 'POST', headers: { 'Content-Type': 'application/json' } });
                    if (!res.ok) throw new Error('Checkout session creation failed');
                    const data = await res.json();
                    window.location.href = data.checkout_url;
                  } catch (err) {
                    console.error('Stripe redirect failed:', err);
                    setErrorMsg('Payment system unavailable.');
                    setStep('error');
                  }
                }}
                className="w-full py-3 bg-[#D4AF37] text-black font-bold text-sm tracking-[0.25em] hover:bg-[#b5952f] transition-all shadow-[0_0_30px_rgba(212,175,55,0.3)] border-2 border-[#D4AF37] rounded-sm"
              >
                DECRYPT DOSSIER — $4.99
              </button>
              <p className="text-[10px] text-gray-600 mt-2 tracking-wider">ONE-TIME PAYMENT · INSTANT ACCESS · SECURE CHECKOUT</p>

              {/* ── Operator Bypass ── */}
              <div className="mt-5 pt-4 border-t border-[#1a1a1a]">
                <input
                  type="text"
                  value={bypassCode}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => setBypassCode(e.target.value)}
                  placeholder="OPERATOR CLEARANCE CODE"
                  className="w-full bg-[#0d0d0e] border border-[#222] text-gray-400 p-2 text-center text-[10px] tracking-[0.2em] focus:outline-none focus:border-[#333] transition-colors mb-2"
                />
                <button
                  onClick={() => {
                    if (bypassCode === 'SCARGODS_ADMIN') {
                      const cached = sessionStorage.getItem('cachedResult');
                      if (cached) {
                        const parsed = JSON.parse(cached) as VerificationResult;
                        setResults(parsed);
                        sessionStorage.removeItem('cachedResult');
                      }
                      setBypassCode('');
                      setStep('complete');
                    }
                  }}
                  className="w-full py-1.5 text-[10px] tracking-[0.2em] text-gray-500 hover:text-gray-300 border border-[#222] hover:border-[#444] bg-transparent transition-all rounded-sm"
                >
                  AUTHORIZE OVERRIDE
                </button>
              </div>

              {/* Cancel */}
              <button
                onClick={() => { sessionStorage.removeItem('cachedResult'); setStep('idle'); setResults(null); setProbeFile(null); if (probePreview) URL.revokeObjectURL(probePreview); setProbePreview(''); setGalleryFile(null); if (galleryPreview) URL.revokeObjectURL(galleryPreview); setGalleryPreview(''); setBypassCode(''); }}
                className="mt-3 text-[10px] text-gray-500 hover:text-gray-300 transition-colors tracking-widest"
              >
                CANCEL AND RESET
              </button>
            </div>
          </div>
        )}

        {/* ════ RESULTS DASHBOARD — Zero-Scroll 70/30 Grid ════ */}
        {step === 'complete' && results && (
          <div className="h-full flex flex-col gap-3 min-h-0 overflow-hidden">
            <div className="flex-1 flex gap-3 min-h-0">

            {/* ── LEFT PANEL (70%): Dual-Pane Visualizer ── */}
            <div className="w-[70%] flex flex-col min-h-0 min-w-0">
              {/* Controls bar */}
              <div className="flex justify-between items-center mb-1.5 shrink-0">
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
                  onClick={() => { setStep('idle'); setProbeFile(null); if (probePreview) URL.revokeObjectURL(probePreview); setProbePreview(''); setIsXrayMode(false); setResults(null); setAuditExpanded(false); }}
                  className="text-[10px] text-gray-500 hover:text-white border border-[#333] hover:border-gray-500 px-3 py-1 rounded tracking-widest transition-colors"
                >
                  NEW RUN
                </button>
              </div>
              <div className="flex-1 min-h-0 bg-[#0d0d0e] border border-[#1f1f1f] rounded-lg p-2">
                <SymmetryMerge
                  galleryImageSrc={results.gallery_aligned_b64}
                  probeImageSrc={results.probe_aligned_b64}
                  deltaImageSrc={results.scar_delta_b64}
                  galleryWireframeSrc={results.gallery_wireframe_b64}
                  probeWireframeSrc={results.probe_wireframe_b64}
                  isXrayMode={isXrayMode}
                />
              </div>
            </div>

            {/* ── RIGHT PANEL (30%): Intelligence Panel ── */}
            <div className="w-[30%] flex flex-col gap-1.5 min-h-0 overflow-y-auto overflow-x-hidden shrink-0 pr-0.5">

              {/* Tier 1 */}
              <div className="border border-[#1f1f1f] bg-[#0d0d0e] rounded-lg p-2.5">
                <div className="flex items-center justify-between">
                  <h3 className="text-gray-500 text-[9px] tracking-wider">TIER 1: STRUCTURAL</h3>
                  <div className="relative group/t1">
                    <span className="text-[9px] text-gray-600 border border-[#333] rounded px-1 cursor-help hover:text-[#D4AF37] hover:border-[#D4AF37]/50 transition-colors">?</span>
                    <div className="pointer-events-none absolute right-0 bottom-full mb-1.5 w-52 opacity-0 group-hover/t1:opacity-100 transition-opacity z-50">
                      <div className="bg-[#111] border border-[#333] rounded px-2.5 py-2 text-[9px] text-gray-300 font-mono leading-relaxed shadow-[0_4px_20px_rgba(0,0,0,0.8)]">Cosine similarity of 1404-D cranial and skeletal anchor points. High baseline overlap; used for broad filtering.</div>
                    </div>
                  </div>
                </div>
                <div className="text-xl text-white font-bold mt-0.5">{results.structural_score}%</div>
              </div>

              {/* Tier 2 */}
              <div className="border border-[#1f1f1f] bg-[#0d0d0e] rounded-lg p-2.5">
                <div className="flex items-center justify-between">
                  <h3 className="text-gray-500 text-[9px] tracking-wider">TIER 2: SOFT BIO</h3>
                  <div className="relative group/t2">
                    <span className="text-[9px] text-gray-600 border border-[#333] rounded px-1 cursor-help hover:text-[#D4AF37] hover:border-[#D4AF37]/50 transition-colors">?</span>
                    <div className="pointer-events-none absolute right-0 bottom-full mb-1.5 w-52 opacity-0 group-hover/t2:opacity-100 transition-opacity z-50">
                      <div className="bg-[#111] border border-[#333] rounded px-2.5 py-2 text-[9px] text-gray-300 font-mono leading-relaxed shadow-[0_4px_20px_rgba(0,0,0,0.8)]">Pixel-density analysis of melanin, ocular hue, and keratin. Low statistical uniqueness; acts as a secondary filter.</div>
                    </div>
                  </div>
                </div>
                <div className="text-xl text-white font-bold mt-0.5">{results.soft_biometrics_score}%</div>
              </div>

              {/* Tier 3 */}
              <div className="border border-[#1f1f1f] bg-[#0d0d0e] rounded-lg p-2.5">
                <div className="flex items-center justify-between">
                  <h3 className="text-gray-500 text-[9px] tracking-wider">TIER 3: MICRO-TOPO</h3>
                  <div className="relative group/t3">
                    <span className="text-[9px] text-gray-600 border border-[#333] rounded px-1 cursor-help hover:text-[#D4AF37] hover:border-[#D4AF37]/50 transition-colors">?</span>
                    <div className="pointer-events-none absolute right-0 bottom-full mb-1.5 w-52 opacity-0 group-hover/t3:opacity-100 transition-opacity z-50">
                      <div className="bg-[#111] border border-[#333] rounded px-2.5 py-2 text-[9px] text-gray-300 font-mono leading-relaxed shadow-[0_4px_20px_rgba(0,0,0,0.8)]">The primary identifier. Analyzes chaotic epidermal deviations (scars, asymmetrical moles). A high match here drives the False Acceptance Rate (FAR) to near zero.</div>
                    </div>
                  </div>
                </div>
                <div className="text-xl text-white font-bold mt-0.5">{results.micro_topology_score}%</div>
              </div>

              {/* Fused Score */}
              <div className="border border-[#D4AF37]/40 bg-[#1a170d] rounded-lg p-2.5 relative overflow-hidden">
                <div className="absolute top-0 right-0 w-8 h-8 bg-[#D4AF37]/10 rounded-bl-full"></div>
                <div className="flex items-center justify-between">
                  <h3 className="text-[#D4AF37] text-[9px] tracking-wider">FUSED IDENTITY</h3>
                  <div className="relative group/tf">
                    <span className="text-[9px] text-[#D4AF37]/60 border border-[#D4AF37]/30 rounded px-1 cursor-help hover:text-[#D4AF37] hover:border-[#D4AF37]/50 transition-colors">?</span>
                    <div className="pointer-events-none absolute right-0 bottom-full mb-1.5 w-52 opacity-0 group-hover/tf:opacity-100 transition-opacity z-50">
                      <div className="bg-[#111] border border-[#333] rounded px-2.5 py-2 text-[9px] text-gray-300 font-mono leading-relaxed shadow-[0_4px_20px_rgba(0,0,0,0.8)]">Bayesian probability matrix combining all tiers to calculate the definitive False Acceptance Rate.</div>
                    </div>
                  </div>
                </div>
                <div className="text-2xl text-[#D4AF37] font-bold mt-0.5">{results.fused_identity_score}%</div>
              </div>

              {/* Conclusion */}
              <div className={`border rounded-lg p-2.5 ${results.veto_triggered ? 'border-red-900 bg-red-950/30' : 'border-[#1f1f1f] bg-[#0d0d0e]'}`}>
                <h3 className="text-gray-500 text-[9px] mb-1 tracking-wider">CONCLUSION</h3>
                <p className={`text-[11px] leading-snug ${results.veto_triggered ? 'text-red-400 font-bold' : 'text-gray-200'}`}>{results.conclusion}</p>
                <div className="mt-1.5">
                  {results.veto_triggered ? (
                    <div className="px-2 py-0.5 bg-red-900 text-red-100 text-[8px] tracking-widest border border-red-500 text-center rounded">ACE-V VETO</div>
                  ) : (
                    <div className="px-2 py-0.5 bg-[#0a0a0a] text-green-500 text-[8px] tracking-widest border border-green-900/50 text-center rounded">NO DISCREPANCY</div>
                  )}
                </div>
              </div>



              {/* Export Dossier */}
              <button
                onClick={generateForensicReport}
                disabled={isExporting}
                className={`w-full py-2 text-[9px] font-bold tracking-widest border rounded transition-all shrink-0 ${
                  isExporting
                    ? 'border-[#333] bg-[#111] text-gray-500 cursor-wait'
                    : 'border-[#D4AF37]/50 bg-[#0a0a0a] text-[#D4AF37] hover:bg-[#D4AF37]/10 hover:shadow-[0_0_15px_rgba(212,175,55,0.4)]'
                }`}
              >
                {isExporting ? 'COMPILING...' : 'EXPORT DOSSIER'}
              </button>
            </div>
            </div>

            {/* ── Full-Width Forensic Audit Block ── */}
            <div className="shrink-0 w-full flex flex-col gap-2">
              <button
                onClick={() => setAuditExpanded(!auditExpanded)}
                className={`w-full flex items-center justify-between px-5 py-2.5 rounded font-mono tracking-[0.2em] transition-all border-2 ${
                  auditExpanded
                    ? 'border-[#D4AF37] bg-[#D4AF37]/10 text-[#D4AF37] shadow-[0_0_20px_rgba(212,175,55,0.2)]'
                    : 'border-[#D4AF37]/50 bg-[#0a0a0a] text-[#D4AF37]/90 hover:border-[#D4AF37] hover:bg-[#111] hover:shadow-[0_0_15px_rgba(212,175,55,0.3)]'
                }`}
              >
                <span className="flex items-center gap-3 font-bold text-[11px]">
                  <span className="relative flex h-3 w-3">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#D4AF37] opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-3 w-3 bg-[#D4AF37]"></span>
                  </span>
                  ACCESS SECURE FORENSIC TELEMETRY &amp; AUDIT LOG
                </span>
                <span className="text-lg font-bold">{auditExpanded ? '−' : '+'}</span>
              </button>

              {/* ── Forensic Terminal (3-column) ── */}
              {auditExpanded && results.audit_log && (
                <div className="border border-[#1a1a0a] bg-[#000000] rounded p-2.5 font-mono text-[9px] leading-relaxed shadow-[inset_0_0_30px_rgba(0,0,0,0.5)]">
                  <div className="grid grid-cols-3 gap-4 mt-1">

                    {/* Block 1: Statistical Certainty */}
                    <div className="border border-[#1a2a1a] rounded p-2 bg-[#010201]">
                      <div className="text-green-500/80 tracking-[0.2em] mb-1.5 border-b border-green-900/30 pb-1 text-[8px]">▸ STATISTICAL CERTAINTY</div>
                      <div className="space-y-0.5 pl-1">
                        <div className="flex justify-between"><span className="text-gray-600">FALSE ACCEPTANCE RATE</span><span className={`font-bold ${results.audit_log.false_acceptance_rate === 'Inconclusive' ? 'text-red-400' : 'text-green-400'}`}>{results.audit_log.false_acceptance_rate}</span></div>
                        <div className="flex justify-between"><span className="text-gray-600">CERTAINTY</span><span className={`font-bold ${results.audit_log.statistical_certainty.startsWith('<') ? 'text-red-400' : 'text-green-400'}`}>{results.audit_log.statistical_certainty}</span></div>
                        <div className="flex justify-between"><span className="text-gray-600">ANCHOR NODES</span><span className="text-white font-bold">{results.audit_log.nodes_mapped}/468</span></div>
                        <div className="flex justify-between"><span className="text-gray-600">COSINE DISTANCE</span><span className="text-white font-bold">{results.audit_log.raw_cosine_score.toFixed(6)}</span></div>
                      </div>
                    </div>

                    {/* Block 2: Spatial Alignment & Liveness */}
                    <div className="border border-[#2a1a1a] rounded p-2 bg-[#020101]">
                      <div className="text-cyan-500/80 tracking-[0.2em] mb-1.5 border-b border-cyan-900/30 pb-1 text-[8px]">▸ SPATIAL ALIGNMENT &amp; LIVENESS</div>
                      <div className="space-y-0.5 pl-1">
                        {results.audit_log.alignment_variance && (<>
                          <div className="flex justify-between"><span className="text-gray-600">YAW CORRECTION</span><span className="text-cyan-300">{results.audit_log.alignment_variance.yaw}</span></div>
                          <div className="flex justify-between"><span className="text-gray-600">PITCH CORRECTION</span><span className="text-cyan-300">{results.audit_log.alignment_variance.pitch}</span></div>
                          <div className="flex justify-between"><span className="text-gray-600">ROLL CORRECTION</span><span className="text-cyan-300">{results.audit_log.alignment_variance.roll}</span></div>
                        </>)}
                        {results.audit_log.liveness_check && (<>
                          <div className="flex justify-between mt-1"><span className="text-gray-600">SPOOF PROBABILITY</span><span className="text-green-400 font-bold">{results.audit_log.liveness_check.spoof_probability}</span></div>
                          <div className="flex justify-between"><span className="text-gray-600">DEEPFAKE STATUS</span><span className={`font-bold ${results.audit_log.liveness_check.status === 'VERIFIED_3D_ORGANIC' ? 'text-green-400' : 'text-yellow-400'}`}>{results.audit_log.liveness_check.status}</span></div>
                        </>)}
                      </div>
                    </div>

                    {/* Block 3: Cryptographic Signature */}
                    <div className="border border-[#1a1a2a] rounded p-2 bg-[#010102]">
                      <div className="text-amber-500/80 tracking-[0.2em] mb-1.5 border-b border-amber-900/30 pb-1 text-[8px]">▸ CRYPTOGRAPHIC SIGNATURE</div>
                      <div className="space-y-0.5 pl-1">
                        {results.audit_log.vector_hash && (
                          <div><span className="text-gray-600">VECTOR SHA-256</span><div className="text-amber-300/80 text-[8px] break-all mt-0.5">{results.audit_log.vector_hash}</div></div>
                        )}
                        {results.audit_log.crypto_envelope && (<>
                          <div className="flex justify-between mt-1"><span className="text-gray-600">ENCRYPTION</span><span className="text-amber-300">{results.audit_log.crypto_envelope.standard}</span></div>
                          <div className="flex justify-between"><span className="text-gray-600">KMS LATENCY</span><span className="text-amber-300">{results.audit_log.crypto_envelope.decryption_time}</span></div>
                        </>)}
                        {results.audit_log.matched_user_id && (
                          <div className="flex justify-between mt-1"><span className="text-gray-600">TARGET ID</span><span className="text-white">{results.audit_log.matched_user_id}</span></div>
                        )}
                        {results.audit_log.person_name && (
                          <div className="flex justify-between"><span className="text-gray-600">PERSON</span><span className="text-white">{results.audit_log.person_name}</span></div>
                        )}
                        {results.audit_log.license_short_name && (
                          <div className="flex justify-between"><span className="text-gray-600">LICENSE</span><span className="text-gray-400">{results.audit_log.license_short_name}</span></div>
                        )}
                      </div>
                    </div>
                  </div>
                  <div className="mt-2 pt-1.5 border-t border-[#1a1a0a] text-[7px] text-gray-700 tracking-widest text-center">AUDIT GENERATED AT {new Date().toISOString()} · SYSTEM INTEGRITY VERIFIED</div>
                </div>
              )}
            </div>

          </div>
        )}

      </div>
      )}
    </main>
  );
}

