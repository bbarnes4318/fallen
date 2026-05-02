'use client';

import React, { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import Image from 'next/image';
import html2canvas from 'html2canvas';
import { jsPDF } from 'jspdf';
import SymmetryMerge from '@/components/SymmetryMerge';

import { VerificationResult } from '@/types/verification';

const IdentityGraph = dynamic(() => import('@/components/IdentityGraph'), { ssr: false });

function escapeHtml(unsafe: string | number | null | undefined): string {
  if (unsafe === null || unsafe === undefined) return '';
  return String(unsafe)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function safeImgSrc(src: string | null | undefined): string {
  if (!src) return '';
  if (src.startsWith('data:image/') || src.startsWith('http://') || src.startsWith('https://')) return src;
  return '';
}

function getApiUrl(): string {
  return '/api';
}

const PIPELINE_STEPS = [
  "[sys] Initializing secure TLS enclave...",
  "[sys] Injecting payload into volatile memory...",
  "[tier_1] Verifying image integrity and EXIF metadata...",
  "[tier_1] Normalizing cross-spectral variants...",
  "[tier_1] Executing MTCNN face detection...",
  "[tier_1] Extracting 512-D neural embeddings (ArcFace)...",
  "[tier_2] Extracting 468-point 3D facial mesh...",
  "[tier_2] Executing 3D Procrustes rigid alignment...",
  "[tier_2] Computing geometric ratios and soft biometrics...",
  "[tier_3] Mapping micro-topology LBP textures...",
  "[tier_3] Scanning for synthetic GAN anomalies...",
  "[tier_4] Extracting localized facial marks and scars...",
  "[tier_4] Querying population frequency database...",
  "[tier_4] Calculating Bayesian Likelihood Ratios...",
  "[sys] Fusing independent identity scores...",
  "[sys] Finalizing audit trail..."
];

function TelemetryLoader() {
  const [logs, setLogs] = useState<string[]>([]);
  const [hash, setHash] = useState<string>('');
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    let currentIndex = 0;
    const interval = setInterval(() => {
      if (currentIndex < PIPELINE_STEPS.length) {
        setLogs((prev: string[]) => {
          const newLogs = [...prev, PIPELINE_STEPS[currentIndex]];
          if (newLogs.length > 5) return newLogs.slice(newLogs.length - 5);
          return newLogs;
        });
        setProgress(Math.min(99, (currentIndex / PIPELINE_STEPS.length) * 100 + Math.random() * 5));
        currentIndex++;
      } else {
        setLogs((prev: string[]) => {
          const newLogs = [...prev, "[sys] Awaiting server response..."];
          if (newLogs.length > 5) return newLogs.slice(newLogs.length - 5);
          return newLogs;
        });
        setProgress(99);
      }
    }, 450);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const hashInterval = setInterval(() => {
      const randomHex = Array.from({length: 64}, () => Math.floor(Math.random()*16).toString(16)).join('');
      setHash(randomHex);
    }, 50);
    return () => clearInterval(hashInterval);
  }, []);

  return (
    <div className="h-full flex flex-col items-center justify-center p-4 w-full">
      <div className="w-full max-w-2xl border border-gray-700 bg-[#050505] flex flex-col p-6 font-mono relative overflow-hidden shadow-[0_0_30px_rgba(0,0,0,0.8)]">
        {/* Header */}
        <div className="flex justify-between items-center border-b border-gray-800 pb-3 mb-4 relative z-10">
          <div className="flex flex-col">
            <span className="text-[#D4AF37] text-xs tracking-[0.2em] font-bold">ACTIVE TELEMETRY</span>
            <span className="text-gray-600 text-[9px] tracking-widest mt-0.5">INSTITUTIONAL PIPELINE STATUS</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-red-500 animate-[pulse_0.5s_infinite] rounded-full"></div>
            <span className="text-red-500 text-[10px] tracking-widest font-bold">PROCESSING</span>
          </div>
        </div>
        
        {/* Cryptographic Visual (Hex Dump) */}
        <div className="bg-[#0a0a0a] p-3 border border-gray-800 mb-4 relative z-10">
          <div className="text-[8px] text-gray-500 tracking-[0.2em] mb-1.5 flex justify-between">
            <span>KMS ENVELOPE DECRYPTION [SHA-256]</span>
            <span className="text-gray-600">SECURE ENCLAVE</span>
          </div>
          <div className="text-[10px] text-emerald-500/80 break-all font-bold tracking-widest leading-relaxed">
            {hash}
          </div>
        </div>

        {/* Terminal Feed */}
        <div className="h-28 flex flex-col justify-end text-[10px] text-gray-400 gap-1.5 mb-4 relative z-10">
          {logs.map((log: string, i: number) => (
            <div key={i} className="flex gap-2 items-start">
              <span className="text-gray-600 shrink-0">{'>'}</span>
              <span className={i === logs.length - 1 ? 'text-gray-200' : 'text-gray-500'}>{log}</span>
            </div>
          ))}
          <div className="flex gap-2 items-start text-[#D4AF37] animate-pulse">
            <span className="shrink-0">{'>'}</span>
            <span>_</span>
          </div>
        </div>

        {/* Progress Architecture */}
        <div className="relative z-10">
          <div className="flex justify-between text-[9px] text-gray-500 mb-1.5 tracking-widest font-bold">
            <span>PIPELINE LOAD</span>
            <span className="text-[#D4AF37]">{Math.floor(progress)}%</span>
          </div>
          <div className="w-full h-2 bg-[#111] border border-gray-700 overflow-hidden">
            <div 
              className="h-full bg-gradient-to-r from-gray-700 via-gray-400 to-white transition-all duration-300 ease-out" 
              style={{ width: `${progress}%` }}
            ></div>
          </div>
        </div>
      </div>
    </div>
  );
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

  /** Format LR in compact scientific notation */
  function formatLRSci(lr: number | null | undefined): string {
    if (lr == null) return 'N/A';
    if (lr >= 1e6) return lr.toExponential(2);
    if (lr >= 100) return lr.toLocaleString(undefined, { maximumFractionDigits: 1 });
    return lr.toFixed(4);
  }

  const [results, setResults] = useState<VerificationResult | null>(null);
  const [lockedJob, setLockedJob] = useState<{job_id: string, preview: Record<string, unknown>} | null>(null);
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
    const jobId = params.get('job_id');
    const sessionId = params.get('session_id');

    if (params.get('success') === 'true' && jobId && sessionId) {
      fetch(`${getApiUrl()}/verify/result/${jobId}?session_id=${sessionId}`)
        .then(res => {
          if (!res.ok) throw new Error('Unlock failed');
          return res.json();
        })
        .then(data => {
          setResults(data);
          setStep('complete');
          sessionStorage.removeItem('lockedJob');
        })
        .catch(err => {
          console.error(err);
          setErrorMsg('Payment verification failed. Contact support.');
          setStep('error');
        });
      window.history.replaceState({}, '', window.location.pathname);
    } else if (params.get('canceled') === 'true') {
      const cached = sessionStorage.getItem('lockedJob');
      if (cached) {
        try {
          const parsed = JSON.parse(cached);
          setTimeout(() => {
            setLockedJob(parsed);
            setStep('paywall');
          }, 0);
        } catch {}
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
      if (data.locked) {
        sessionStorage.setItem('lockedJob', JSON.stringify(data));
        setLockedJob(data);
        setStep('paywall');
      } else {
        setResults(data);
        setStep('complete');
      }
      
    } catch (err: unknown) {
      setErrorMsg(err instanceof Error ? err.message : 'An unknown error occurred');
      setStep('error');
    }
  };

  // ── Graph-to-Comparison bridge: receives URLs from IdentityGraph ──
  const handleGraphCompare = async (galleryUrl: string, probeUrl: string) => {
    try {
      setViewMode('acquisition');
      setMode('compare');
      setStep('frontalizing');
      setGalleryPreview(galleryUrl);
      setProbePreview(probeUrl);

      await new Promise(resolve => setTimeout(resolve, 1200));
      setStep('calculating');

      const verifyRes = await fetch(`${getApiUrl()}/verify/fuse`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
        body: JSON.stringify({ gallery_url: galleryUrl, probe_url: probeUrl })
      });
      if (verifyRes.status === 401) { handleLogout(); return; }
      if (!verifyRes.ok) {
        const errBody = await verifyRes.json().catch(() => null);
        throw new Error(errBody?.detail || 'Verification pipeline failed.');
      }
      const data = await verifyRes.json();

      if (data.locked) {
        sessionStorage.setItem('lockedJob', JSON.stringify(data));
        setLockedJob(data);
        setStep('paywall');
      } else {
        setResults(data);
        setStep('complete');
      }
    } catch (err: unknown) {
      setErrorMsg(err instanceof Error ? err.message : 'Graph comparison failed.');
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
                <div style="font-size:8px;color:#D4AF37;letter-spacing:5px;margin-bottom:2px;">▓▓ REPORT ▓▓</div>
                <div style="font-size:16px;font-weight:bold;color:white;letter-spacing:3px;">BIOMETRIC <span style="color:#D4AF37;">SIMILARITY REPORT</span></div>
              </div>
              <div style="text-align:right;">
                <div style="font-size:7px;color:#555;letter-spacing:2px;">GENERATED ${escapeHtml(ts)} UTC</div>
                <div style="font-size:7px;color:#555;letter-spacing:2px;">DOC ID: ${escapeHtml(docId)} · EXPERIMENTAL ANALYSIS</div>
              </div>
            </div>
          </div>

          <!-- VISUAL EVIDENCE -->
          <div style="display:flex;gap:8px;margin-bottom:10px;flex-shrink:0;">
            <div style="flex:1;border:1px solid #333;padding:4px;background:#0a0a0a;text-align:center;">
              <img src="${safeImgSrc(results.probe_aligned_b64)}" style="width:100%;height:180px;object-fit:contain;display:block;" />
              <div style="font-size:7px;color:#666;letter-spacing:3px;margin-top:4px;">PROBE (UNKNOWN TARGET)</div>
            </div>
            <div style="flex:1;border:1px solid #333;padding:4px;background:#0a0a0a;text-align:center;">
              <img src="${safeImgSrc(results.gallery_aligned_b64)}" style="width:100%;height:180px;object-fit:contain;display:block;" />
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
              <div style="font-size:7px;color:#D4AF37;letter-spacing:2px;">POSTERIOR PROB</div>
              <div style="font-size:22px;color:#D4AF37;font-weight:bold;margin:2px 0;">${results.fused_identity_score}%</div>
              <div style="font-size:7px;color:#997a1d;line-height:1.3;">LR<sub>total</sub>: ${audit?.lr_total != null ? formatLRSci(audit.lr_total) : 'N/A'}</div>
            </div>
          </div>

          <!-- CONCLUSION -->
          <div style="border:1px solid ${results.fused_identity_score < 40.0 ? '#7f1d1d' : (results.veto_triggered && results.fused_identity_score >= 40.0) ? '#92400e' : '#333'};padding:8px 12px;background:${results.fused_identity_score < 40.0 ? '#1a0505' : (results.veto_triggered && results.fused_identity_score >= 40.0) ? '#291102' : '#0a0a0a'};margin-bottom:10px;flex-shrink:0;display:flex;justify-content:space-between;align-items:center;">
            <div>
              <div style="font-size:7px;color:#666;letter-spacing:2px;margin-bottom:2px;">CONCLUSION</div>
              <div style="font-size:11px;color:${results.fused_identity_score < 40.0 ? '#f87171' : (results.veto_triggered && results.fused_identity_score >= 40.0) ? '#fbbf24' : '#e5e5e5'};font-weight:bold;">${results.conclusion}</div>
            </div>
            <div style="padding:3px 10px;background:${results.fused_identity_score < 40.0 ? '#7f1d1d' : (results.veto_triggered && results.fused_identity_score >= 40.0) ? '#78350f' : '#0a0a0a'};color:${results.fused_identity_score < 40.0 ? '#fee2e2' : (results.veto_triggered && results.fused_identity_score >= 40.0) ? '#fef3c7' : '#22c55e'};font-size:8px;letter-spacing:2px;border:1px solid ${results.fused_identity_score < 40.0 ? '#ef4444' : (results.veto_triggered && results.fused_identity_score >= 40.0) ? '#d97706' : 'rgba(34,197,94,0.3)'};">${results.fused_identity_score < 40.0 ? 'ACE-V VETO' : (results.veto_triggered && results.fused_identity_score >= 40.0) ? 'VETO OVERRIDDEN' : 'NO DISCREPANCY'}</div>
          </div>

          <!-- CRYPTOGRAPHIC AUDIT LOG (Terminal Block) -->
          <div style="flex:1;border:1px solid #1a1a0a;background:#000;padding:10px 14px;font-size:8px;line-height:1.6;overflow:hidden;">
            <div style="color:#D4AF37;font-size:8px;letter-spacing:3px;margin-bottom:8px;border-bottom:1px solid #1a1a0a;padding-bottom:4px;font-weight:bold;">▸ CRYPTOGRAPHIC AUDIT LOG</div>

            <div style="display:flex;gap:16px;">
              <!-- Col 1: Statistical Certainty -->
              <div style="flex:1;">
                <div style="color:#22c55e;font-size:7px;letter-spacing:2px;margin-bottom:4px;">STATISTICAL CERTAINTY</div>
              <div style="display:flex;justify-content:space-between;margin-bottom:2px;"><span style="color:#555;">FALSE ACCEPT RATE</span><span style="color:#4ade80;font-weight:bold;">${escapeHtml(audit?.false_acceptance_rate) || 'N/A'}</span></div>
                <div style="display:flex;justify-content:space-between;margin-bottom:2px;"><span style="color:#555;">CERTAINTY</span><span style="color:#4ade80;font-weight:bold;">${escapeHtml(audit?.statistical_certainty) || 'N/A'}</span></div>
                <div style="display:flex;justify-content:space-between;margin-bottom:2px;"><span style="color:#555;">NODES MAPPED</span><span style="color:#fff;font-weight:bold;">${escapeHtml(audit?.nodes_mapped) || 468}/468</span></div>
                <div style="display:flex;justify-content:space-between;"><span style="color:#555;">COSINE DIST</span><span style="color:#fff;font-weight:bold;">${escapeHtml(audit?.raw_cosine_score?.toFixed(6)) || 'N/A'}</span></div>
              </div>

              <!-- Col 2: Spatial Alignment & Liveness -->
              <div style="flex:1;border-left:1px solid #1a1a0a;padding-left:16px;">
                <div style="color:#06b6d4;font-size:7px;letter-spacing:2px;margin-bottom:4px;">SPATIAL ALIGNMENT</div>
                <div style="display:flex;justify-content:space-between;margin-bottom:2px;"><span style="color:#555;">YAW</span><span style="color:#67e8f9;">${escapeHtml(audit?.alignment_variance?.yaw) || 'N/A'}</span></div>
                <div style="display:flex;justify-content:space-between;margin-bottom:2px;"><span style="color:#555;">PITCH</span><span style="color:#67e8f9;">${escapeHtml(audit?.alignment_variance?.pitch) || 'N/A'}</span></div>
                <div style="display:flex;justify-content:space-between;margin-bottom:2px;"><span style="color:#555;">ROLL</span><span style="color:#67e8f9;">${escapeHtml(audit?.alignment_variance?.roll) || 'N/A'}</span></div>
                <div style="color:#06b6d4;font-size:7px;letter-spacing:2px;margin:4px 0 2px;">LIVENESS</div>
                <div style="display:flex;justify-content:space-between;margin-bottom:2px;"><span style="color:#555;">SPOOF PROB</span><span style="color:#4ade80;font-weight:bold;">${escapeHtml(audit?.liveness_check?.spoof_probability) || 'N/A'}</span></div>
                <div style="display:flex;justify-content:space-between;"><span style="color:#555;">STATUS</span><span style="color:#4ade80;font-weight:bold;">${escapeHtml(audit?.liveness_check?.status) || 'N/A'}</span></div>
              </div>

              <!-- Col 3: Cryptography & Source -->
              <div style="flex:1;border-left:1px solid #1a1a0a;padding-left:16px;">
                <div style="color:#f59e0b;font-size:7px;letter-spacing:2px;margin-bottom:4px;">CRYPTOGRAPHY</div>
                <div style="margin-bottom:2px;"><span style="color:#555;">VECTOR SHA-256</span></div>
                <div style="color:#fbbf24;font-size:6px;word-break:break-all;margin-bottom:3px;">${escapeHtml(audit?.vector_hash) || 'N/A'}</div>
                <div style="display:flex;justify-content:space-between;margin-bottom:2px;"><span style="color:#555;">ENCRYPTION</span><span style="color:#fbbf24;">${escapeHtml(audit?.crypto_envelope?.standard) || 'N/A'}</span></div>
                <div style="display:flex;justify-content:space-between;margin-bottom:2px;"><span style="color:#555;">KMS LATENCY</span><span style="color:#fbbf24;">${escapeHtml(audit?.crypto_envelope?.decryption_time) || 'N/A'}</span></div>
                ${audit?.matched_user_id ? `<div style="color:#f59e0b;font-size:7px;letter-spacing:2px;margin:4px 0 2px;">SOURCE</div><div style="display:flex;justify-content:space-between;margin-bottom:2px;"><span style="color:#555;">TARGET</span><span style="color:#fff;">${escapeHtml(audit.matched_user_id)}</span></div>` : ''}
                ${audit?.person_name ? `<div style="display:flex;justify-content:space-between;margin-bottom:2px;"><span style="color:#555;">PERSON</span><span style="color:#fff;">${escapeHtml(audit.person_name)}</span></div>` : ''}
                ${audit?.license_short_name ? `<div style="display:flex;justify-content:space-between;"><span style="color:#555;">LICENSE</span><span style="color:#888;">${escapeHtml(audit.license_short_name)}</span></div>` : ''}
              </div>
            </div>
          </div>

          <!-- FOOTER -->
          <div style="margin-top:8px;padding-top:6px;border-top:1px solid #1a1a0a;display:flex;justify-content:space-between;flex-shrink:0;">
            <div style="font-size:6px;color:#333;letter-spacing:2px;">GENERATED REPORT</div>
            <div style="font-size:6px;color:#333;letter-spacing:2px;">DOC ${escapeHtml(docId)} · ${escapeHtml(ts)}</div>
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
      pdf.save(`FALLEN_DOSSIER_${fileTs}.pdf`);
    } catch (err) {
      console.error('PDF export failed:', err);
    } finally {
      setIsExporting(false);
    }
  };

  // ─── State for login overlay on landing page ─────────
  const [showLoginOverlay, setShowLoginOverlay] = useState(false);

  // ─── IDENTITY GATEWAY (Landing Page) ───────
  if (!token) {
    return (
      <main className="min-h-screen w-full bg-[#050A10] text-[#E0E0E0] selection:bg-[#D4AF37] selection:text-black overflow-y-auto overflow-x-hidden">

        {/* ═══════════════════════════════════════════════
            SECTION 1: HERO — Identity Gateway
            ═══════════════════════════════════════════════ */}
        <section className="relative min-h-screen flex flex-col">

          {/* ── Top Navigation Bar ── */}
          <nav className="shrink-0 flex items-center justify-between px-8 py-5 relative z-20">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 border-2 border-[#D4AF37] flex items-center justify-center">
                <div className="w-3 h-3 bg-[#D4AF37]"></div>
              </div>
              <div>
                <span className="text-white font-bold text-sm tracking-[0.25em]">FAL<span className="text-[#D4AF37]">LEN</span></span>
                <span className="hidden sm:inline text-gray-600 text-[9px] ml-3 tracking-[0.15em]">BIOMETRIC INTELLIGENCE</span>
              </div>
            </div>
            <button
              onClick={() => setShowLoginOverlay(true)}
              className="px-5 py-2 text-[10px] tracking-[0.2em] font-bold text-gray-400 border border-[#1F2937] hover:border-[#D4AF37]/50 hover:text-[#D4AF37] bg-transparent transition-all"
            >
              OPERATOR LOGIN
            </button>
          </nav>

          {/* ── Hero Content ── */}
          <div className="flex-1 flex flex-col items-center justify-center px-6 text-center relative z-10 pb-16">
            {/* Background radial glow */}
            <div className="absolute inset-0 pointer-events-none overflow-hidden">
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-[#D4AF37]/[0.03] rounded-full blur-[120px]"></div>
              <div className="absolute top-1/4 right-1/4 w-[300px] h-[300px] bg-[#D4AF37]/[0.02] rounded-full blur-[80px]"></div>
            </div>

            {/* Decorative corner brackets */}
            <div className="absolute top-28 left-8 w-10 h-10 border-t border-l border-[#D4AF37]/20 pointer-events-none"></div>
            <div className="absolute top-28 right-8 w-10 h-10 border-t border-r border-[#D4AF37]/20 pointer-events-none"></div>
            <div className="absolute bottom-28 left-8 w-10 h-10 border-b border-l border-[#D4AF37]/20 pointer-events-none"></div>
            <div className="absolute bottom-28 right-8 w-10 h-10 border-b border-r border-[#D4AF37]/20 pointer-events-none"></div>

            <div className="relative z-10 max-w-3xl">
              <div className="text-[#D4AF37]/60 text-[10px] tracking-[0.5em] font-mono mb-6 flex items-center justify-center gap-3">
                <div className="w-8 h-[1px] bg-[#D4AF37]/30"></div>
                FALLEN FACIAL INTELLIGENCE
                <div className="w-8 h-[1px] bg-[#D4AF37]/30"></div>
              </div>

              <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-white leading-[1.1] tracking-tight mb-6" style={{ fontFamily: "'Inter', 'Helvetica Neue', sans-serif" }}>
                Securely Match and Verify Identities{' '}
                <span className="text-[#D4AF37]">in the Vault.</span>
              </h1>

              <p className="text-gray-400 text-base sm:text-lg leading-relaxed max-w-2xl mx-auto mb-10" style={{ fontFamily: "'Inter', 'Helvetica Neue', sans-serif" }}>
                Search over 1 million secure facial records to find duplicate entries, verify an identity, or prevent fraud. Faster. More Accurate. Institutional Grade.
              </p>

              <button
                onClick={() => setShowLoginOverlay(true)}
                className="group relative px-12 py-4 bg-[#D4AF37] text-black font-bold text-base tracking-[0.3em] hover:bg-[#b5952f] transition-all shadow-[0_0_40px_rgba(212,175,55,0.25)] hover:shadow-[0_0_60px_rgba(212,175,55,0.4)] border-2 border-[#D4AF37]"
              >
                <span className="relative z-10">SCAN THE VAULT</span>
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent translate-x-[-200%] group-hover:translate-x-[200%] transition-transform duration-700"></div>
              </button>

              <p className="text-gray-600 text-[10px] tracking-[0.2em] font-mono mt-5">
                256-BIT ENCRYPTION · SUB-SECOND MATCHING · HIGH-RELIABILITY METRICS
              </p>
            </div>
          </div>

          {/* Scroll indicator */}
          <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 z-10">
            <div className="w-[1px] h-8 bg-gradient-to-b from-transparent to-[#D4AF37]/40 animate-pulse"></div>
          </div>
        </section>


        {/* ═══════════════════════════════════════════════
            SECTION 2: TRUST HUD — Social Proof Band
            ═══════════════════════════════════════════════ */}
        <section className="border-y border-[#1F2937] bg-[#0A0E17]/80 py-10 px-6">
          <div className="max-w-5xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-8">
            {/* Metric 1 */}
            <div className="flex items-center gap-4">
              <div className="shrink-0 w-12 h-12 border border-[#D4AF37]/30 flex items-center justify-center bg-[#D4AF37]/5">
                <svg className="w-5 h-5 text-[#D4AF37]" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"></path></svg>
              </div>
              <div>
                <div className="text-2xl font-bold text-white font-mono tracking-wider">1.2M+</div>
                <div className="text-[10px] text-gray-500 tracking-[0.15em]">MATCHES VERIFIED</div>
              </div>
            </div>
            {/* Metric 2 */}
            <div className="flex items-center gap-4">
              <div className="shrink-0 w-12 h-12 border border-[#D4AF37]/30 flex items-center justify-center bg-[#D4AF37]/5">
                <svg className="w-5 h-5 text-[#D4AF37]" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
              </div>
              <div>
                <div className="text-2xl font-bold text-white font-mono tracking-wider">&lt;0.001%</div>
                <div className="text-[10px] text-gray-500 tracking-[0.15em]">FALSE POSITIVE RATE</div>
              </div>
            </div>
            {/* Metric 3 */}
            <div className="flex items-center gap-4">
              <div className="shrink-0 w-12 h-12 border border-[#D4AF37]/30 flex items-center justify-center bg-[#D4AF37]/5">
                <svg className="w-5 h-5 text-[#D4AF37]" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"></path></svg>
              </div>
              <div>
                <div className="text-2xl font-bold text-white font-mono tracking-wider">AES-256</div>
                <div className="text-[10px] text-gray-500 tracking-[0.15em]">GCM ENCRYPTION</div>
              </div>
            </div>
          </div>
        </section>


        {/* ═══════════════════════════════════════════════
            SECTION 3: VAULT IN-ACTION — Feature Grid
            ═══════════════════════════════════════════════ */}
        <section className="py-20 px-6">
          <div className="max-w-5xl mx-auto">
            <div className="text-center mb-14">
              <div className="text-[#D4AF37]/50 text-[10px] tracking-[0.4em] font-mono mb-3">HOW IT WORKS</div>
              <h2 className="text-2xl sm:text-3xl font-bold text-white tracking-tight" style={{ fontFamily: "'Inter', 'Helvetica Neue', sans-serif" }}>
                Intelligence-Grade Biometric Verification
              </h2>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Card 1: Fraud Prevention */}
              <div className="border border-[#1F2937] bg-[#0A0E17] p-6 group hover:border-[#D4AF37]/30 transition-all">
                <div className="mb-5 h-44 border border-[#1F2937] bg-[#050A10] flex items-center justify-center overflow-hidden relative">
                  {/* Stylized comparison mockup */}
                  <div className="flex items-center gap-4">
                    <div className="w-24 h-28 border border-[#1F2937] bg-[#0d1117] flex flex-col items-center justify-center relative">
                      <div className="w-12 h-14 rounded-sm bg-gradient-to-b from-[#1F2937] to-[#111] mb-1.5"></div>
                      <div className="text-[7px] text-gray-600 tracking-wider">UPLOAD</div>
                      <div className="absolute -top-1 -left-1 w-2 h-2 border-t border-l border-[#D4AF37]/40"></div>
                      <div className="absolute -top-1 -right-1 w-2 h-2 border-t border-r border-[#D4AF37]/40"></div>
                      <div className="absolute -bottom-1 -left-1 w-2 h-2 border-b border-l border-[#D4AF37]/40"></div>
                      <div className="absolute -bottom-1 -right-1 w-2 h-2 border-b border-r border-[#D4AF37]/40"></div>
                    </div>
                    <div className="flex flex-col items-center gap-1">
                      <div className="text-[8px] font-mono text-[#D4AF37]/80 tracking-wider">99.4%</div>
                      <div className="flex items-center gap-1">
                        <div className="w-4 h-[1px] bg-[#D4AF37]/50"></div>
                        <svg className="w-3 h-3 text-[#D4AF37]" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 7l5 5m0 0l-5 5m5-5H6"></path></svg>
                        <div className="w-4 h-[1px] bg-[#D4AF37]/50"></div>
                      </div>
                      <div className="text-[6px] text-red-400/80 tracking-wider font-mono">⚠ DUPLICATE</div>
                    </div>
                    <div className="w-24 h-28 border border-red-900/40 bg-[#0d1117] flex flex-col items-center justify-center relative">
                      <div className="w-12 h-14 rounded-sm bg-gradient-to-b from-[#1F2937] to-[#111] mb-1.5"></div>
                      <div className="text-[7px] text-gray-600 tracking-wider">VAULT MATCH</div>
                      <div className="absolute -top-1 -left-1 w-2 h-2 border-t border-l border-red-500/40"></div>
                      <div className="absolute -top-1 -right-1 w-2 h-2 border-t border-r border-red-500/40"></div>
                      <div className="absolute -bottom-1 -left-1 w-2 h-2 border-b border-l border-red-500/40"></div>
                      <div className="absolute -bottom-1 -right-1 w-2 h-2 border-b border-r border-red-500/40"></div>
                    </div>
                  </div>
                  {/* Scan line animation */}
                  <div className="absolute inset-0 pointer-events-none overflow-hidden">
                    <div className="absolute left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-[#D4AF37]/20 to-transparent top-0 animate-[scan_4s_ease-in-out_infinite]"></div>
                  </div>
                </div>
                <h3 className="text-white font-bold text-lg tracking-wider mb-2" style={{ fontFamily: "'Inter', 'Helvetica Neue', sans-serif" }}>Fraud Prevention</h3>
                <p className="text-gray-400 text-sm leading-relaxed">
                  Spot duplicate IDs instantly. The network identified a 99.4% match in the vault that used a different legal name. Catch synthetic identities before they cause damage.
                </p>
              </div>

              {/* Card 2: Verifiable Accuracy */}
              <div className="border border-[#1F2937] bg-[#0A0E17] p-6 group hover:border-[#D4AF37]/30 transition-all">
                <div className="mb-5 h-44 border border-[#1F2937] bg-[#050A10] flex items-center justify-center overflow-hidden relative">
                  {/* Stylized mesh graphic */}
                  <svg className="w-32 h-32 opacity-60" viewBox="0 0 120 120" fill="none" xmlns="http://www.w3.org/2000/svg">
                    {/* Face outline */}
                    <ellipse cx="60" cy="58" rx="35" ry="42" stroke="#D4AF37" strokeWidth="0.5" opacity="0.4"/>
                    {/* Mesh grid lines */}
                    <line x1="60" y1="16" x2="60" y2="100" stroke="#D4AF37" strokeWidth="0.3" opacity="0.2"/>
                    <line x1="25" y1="58" x2="95" y2="58" stroke="#D4AF37" strokeWidth="0.3" opacity="0.2"/>
                    {/* Eye regions */}
                    <circle cx="45" cy="48" r="6" stroke="#D4AF37" strokeWidth="0.5" opacity="0.5"/>
                    <circle cx="75" cy="48" r="6" stroke="#D4AF37" strokeWidth="0.5" opacity="0.5"/>
                    <circle cx="45" cy="48" r="2" fill="#D4AF37" opacity="0.3"/>
                    <circle cx="75" cy="48" r="2" fill="#D4AF37" opacity="0.3"/>
                    {/* Nose */}
                    <path d="M55 55 L60 68 L65 55" stroke="#D4AF37" strokeWidth="0.5" opacity="0.4" fill="none"/>
                    {/* Mouth */}
                    <path d="M48 75 Q60 82 72 75" stroke="#D4AF37" strokeWidth="0.5" opacity="0.4" fill="none"/>
                    {/* Mesh connection points */}
                    {[
                      [40,30],[50,25],[60,23],[70,25],[80,30],
                      [32,40],[88,40],[28,55],[92,55],[30,70],[90,70],
                      [35,82],[45,88],[55,92],[65,92],[75,88],[85,82],
                      [60,35],[50,40],[70,40],[42,60],[78,60],
                      [48,70],[72,70],[55,80],[65,80],
                    ].map(([cx, cy], i) => (
                      <circle key={i} cx={cx} cy={cy} r="1.2" fill="#D4AF37" opacity="0.5"/>
                    ))}
                    {/* Connection lines */}
                    {[
                      [[40,30],[50,25]],[[50,25],[60,23]],[[60,23],[70,25]],[[70,25],[80,30]],
                      [[40,30],[32,40]],[[80,30],[88,40]],[[32,40],[28,55]],[[88,40],[92,55]],
                      [[28,55],[30,70]],[[92,55],[90,70]],[[30,70],[35,82]],[[90,70],[85,82]],
                      [[35,82],[45,88]],[[45,88],[55,92]],[[55,92],[65,92]],[[65,92],[75,88]],[[75,88],[85,82]],
                      [[50,40],[45,48]],[[70,40],[75,48]],[[42,60],[48,70]],[[78,60],[72,70]],
                      [[55,80],[60,23]],[[65,80],[60,23]],
                    ].map(([[x1,y1],[x2,y2]], i) => (
                      <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke="#D4AF37" strokeWidth="0.3" opacity="0.25"/>
                    ))}
                  </svg>
                  <div className="absolute bottom-3 right-3 text-[8px] font-mono text-[#D4AF37]/50 tracking-wider">468 NODES MAPPED</div>
                </div>
                <h3 className="text-white font-bold text-lg tracking-wider mb-2" style={{ fontFamily: "'Inter', 'Helvetica Neue', sans-serif" }}>Verifiable Accuracy</h3>
                <p className="text-gray-400 text-sm leading-relaxed">
                  Our system maps the micro-topology of the skin surface to establish unique biometric identity, not just a 2D face comparison. Scars, pore density, and wrinkle patterns are all analyzed.
                </p>
              </div>
            </div>
          </div>
        </section>


        {/* ═══════════════════════════════════════════════
            SECTION 4: FOOTER — Minimal Institutional
            ═══════════════════════════════════════════════ */}
        <footer className="border-t border-[#1F2937] py-8 px-6">
          <div className="max-w-5xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <span className="text-white font-bold text-sm tracking-[0.15em]">FAL<span className="text-[#D4AF37]">LEN</span></span>
              <span className="text-gray-700 text-xs">|</span>
              <span className="text-gray-500 text-[10px] tracking-wider">Secure Facial Biometrics</span>
            </div>
            <div className="flex items-center gap-6">
              <a href="#" className="text-gray-600 text-[10px] tracking-wider hover:text-gray-400 transition-colors">TERMS</a>
              <a href="/privacy" className="text-gray-600 text-[10px] tracking-wider hover:text-gray-400 transition-colors">PRIVACY</a>
              <a href="#" className="text-gray-600 text-[10px] tracking-wider hover:text-gray-400 transition-colors">API DOCS</a>
            </div>
          </div>
        </footer>


        {/* ═══════════════════════════════════════════════
            LOGIN OVERLAY — Triggered by "Operator Login"
            ═══════════════════════════════════════════════ */}
        {showLoginOverlay && (
          <div className="fixed inset-0 z-50 flex items-center justify-center">
            {/* Backdrop */}
            <div
              className="absolute inset-0 bg-black/80 backdrop-blur-sm"
              onClick={() => setShowLoginOverlay(false)}
            ></div>
            {/* Modal */}
            <div className="relative w-full max-w-sm p-8 bg-[#0A0E17] border border-[#1F2937] shadow-[0_0_60px_rgba(0,0,0,0.8)]">
              {/* Close button */}
              <button
                onClick={() => setShowLoginOverlay(false)}
                className="absolute top-3 right-3 text-gray-600 hover:text-gray-300 transition-colors text-lg"
              >
                ✕
              </button>
              <div className="text-center mb-6">
                <div className="w-12 h-12 mx-auto border-2 border-[#D4AF37]/40 flex items-center justify-center mb-4 bg-[#D4AF37]/5">
                  <svg className="w-5 h-5 text-[#D4AF37]" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"></path></svg>
                </div>
                <h2 className="text-lg font-bold tracking-[0.25em] text-white mb-1">
                  OPERATOR <span className="text-[#D4AF37]">LOGIN</span>
                </h2>
                <p className="text-gray-600 text-[10px] tracking-[0.15em]">AUTHENTICATED ACCESS REQUIRED</p>
              </div>

              <form onSubmit={handleLogin} className="space-y-4">
                <input
                  type="password"
                  value={password}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => setPassword(e.target.value)}
                  placeholder="ENTER PASSPHRASE"
                  className="w-full bg-[#050A10] border border-[#1F2937] text-white p-3 text-center text-sm tracking-widest focus:outline-none focus:border-[#D4AF37] transition-colors font-mono"
                  autoFocus
                />

                {loginError && (
                  <div className="text-red-500 text-[10px] text-center tracking-wider">{loginError}</div>
                )}

                <button
                  type="submit"
                  className="w-full py-3 bg-[#D4AF37] text-black font-bold text-sm tracking-[0.25em] hover:bg-[#b5952f] transition-all shadow-[0_0_20px_rgba(212,175,55,0.2)] border border-[#D4AF37]"
                >
                  AUTHENTICATE
                </button>
              </form>

              <p className="text-gray-700 text-[9px] text-center mt-4 tracking-wider">ZERO-TRUST · END-TO-END ENCRYPTED</p>
            </div>
          </div>
        )}

        {/* ── Keyframe for scan animation ── */}
        <style jsx>{`
          @keyframes scan {
            0%, 100% { top: 0%; }
            50% { top: 100%; }
          }
        `}</style>
      </main>
    );
  }

  // ─── MAIN TERMINAL ────────────────────────────────────
  return (
    <main className="h-screen w-full overflow-hidden bg-[#0A0A0B] text-[#E0E0E0] font-mono selection:bg-[#D4AF37] selection:text-black flex flex-col">
      
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
          <a
            href="/architecture"
            className="text-[10px] tracking-widest font-bold text-gray-500 hover:text-[#D4AF37] transition-colors border border-[#1f1f1f] hover:border-[#D4AF37]/30 px-3 py-1 bg-[#0d0d0e]"
          >
            ARCHITECTURE
          </a>
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
          <IdentityGraph onCompare={handleGraphCompare} />
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
                  <input type="file" accept="image/*" aria-label="Upload Target Image" onChange={handleFileChange} className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10" />
                  <div className="absolute top-4 left-4 w-4 h-4 border-t border-l border-[#D4AF37]/50"></div>
                  <div className="absolute top-4 right-4 w-4 h-4 border-t border-r border-[#D4AF37]/50"></div>
                  <div className="absolute bottom-4 left-4 w-4 h-4 border-b border-l border-[#D4AF37]/50"></div>
                  <div className="absolute bottom-4 right-4 w-4 h-4 border-b border-r border-[#D4AF37]/50"></div>
                  {probePreview ? (
                    <Image src={probePreview} alt="Target" width={400} height={200} className="max-h-[200px] w-auto object-contain rounded shadow-[0_0_20px_rgba(212,175,55,0.15)] z-0 relative" unoptimized />
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
                  <input type="file" accept="image/*" aria-label="Upload Probe Image" onChange={handleFileChange} className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10" />
                  <div className="absolute top-3 left-3 w-3 h-3 border-t border-l border-[#D4AF37]/50"></div>
                  <div className="absolute top-3 right-3 w-3 h-3 border-t border-r border-[#D4AF37]/50"></div>
                  <div className="absolute bottom-3 left-3 w-3 h-3 border-b border-l border-[#D4AF37]/50"></div>
                  <div className="absolute bottom-3 right-3 w-3 h-3 border-b border-r border-[#D4AF37]/50"></div>
                  {probePreview ? (
                    <Image src={probePreview} alt="Probe" width={400} height={160} className="max-h-[160px] w-auto object-contain rounded shadow-[0_0_15px_rgba(212,175,55,0.1)] z-0 relative" unoptimized />
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
                  <input type="file" accept="image/*" aria-label="Upload Gallery Image" onChange={handleGalleryChange} className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10" />
                  <div className="absolute top-3 left-3 w-3 h-3 border-t border-l border-gray-700/50"></div>
                  <div className="absolute top-3 right-3 w-3 h-3 border-t border-r border-gray-700/50"></div>
                  <div className="absolute bottom-3 left-3 w-3 h-3 border-b border-l border-gray-700/50"></div>
                  <div className="absolute bottom-3 right-3 w-3 h-3 border-b border-r border-gray-700/50"></div>
                  {galleryPreview ? (
                    <Image src={galleryPreview} alt="Gallery" width={400} height={160} className="max-h-[160px] w-auto object-contain rounded shadow-[0_0_15px_rgba(255,255,255,0.05)] z-0 relative" unoptimized />
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
            <div className="mt-4 text-center max-w-lg px-4">
              <p className="text-[10px] text-gray-500 mb-2 leading-relaxed">
                <strong className="text-gray-400">Disclaimer:</strong> This tool provides experimental biometric similarity analysis only. Results may be inaccurate and do not prove identity. Do not use as the sole basis for legal, employment, financial, medical, or law-enforcement decisions. Users must have appropriate rights/permission to upload images.
              </p>
              <p className="text-[10px] text-gray-500 leading-relaxed">
                By uploading, you agree that images are processed solely for experimental similarity analysis. <a href="/privacy" className="underline hover:text-gray-300">Privacy Policy</a>
              </p>
            </div>
          </div>
        )}

        {/* ════ LOADING ════ */}
        {['uploading', 'frontalizing', 'calculating'].includes(step) && (
          <TelemetryLoader />
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
        {step === 'paywall' && lockedJob && (
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
                  <div><div className="text-[9px] text-gray-500 tracking-wider">GEOMETRIC</div><div className="text-lg text-white font-bold">██.█%</div></div>
                  <div><div className="text-[9px] text-gray-500 tracking-wider">MICRO-TOPO</div><div className="text-lg text-white font-bold">██.█%</div></div>
                </div>
              </div>

              {/* Stripe CTA */}
              <button
                onClick={async () => {
                  try {
                    const res = await fetch(`${getApiUrl()}/checkout/create-session`, { 
                      method: 'POST', 
                      headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify({ job_id: lockedJob.job_id })
                    });
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
                  onClick={async () => {
                    try {
                      const res = await fetch(`${getApiUrl()}/verify/result/${lockedJob.job_id}?bypass_code=${bypassCode}`);
                      if (!res.ok) throw new Error('Bypass failed');
                      const data = await res.json();
                      setResults(data);
                      sessionStorage.removeItem('lockedJob');
                      setBypassCode('');
                      setStep('complete');
                    } catch (err) {
                      console.error(err);
                      alert('Invalid operator clearance code.');
                    }
                  }}
                  className="w-full py-1.5 text-[10px] tracking-[0.2em] text-gray-500 hover:text-gray-300 border border-[#222] hover:border-[#444] bg-transparent transition-all rounded-sm"
                >
                  AUTHORIZE OVERRIDE
                </button>
              </div>

              {/* Cancel */}
              <button
                onClick={() => { sessionStorage.removeItem('lockedJob'); setStep('idle'); setResults(null); setLockedJob(null); setProbeFile(null); if (probePreview) URL.revokeObjectURL(probePreview); setProbePreview(''); setGalleryFile(null); if (galleryPreview) URL.revokeObjectURL(galleryPreview); setGalleryPreview(''); setBypassCode(''); }}
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
              <div className={`flex-1 min-h-0 bg-[#0d0d0e] border rounded-lg p-2 ${(results.fused_identity_score < 40.0) ? 'border-red-900/50 shadow-[0_0_20px_rgba(180,0,30,0.15)]' : 'border-[#D4AF37]/30 shadow-[0_0_20px_rgba(212,175,55,0.08)]'}`}>
                <SymmetryMerge
                  results={results}
                  isXrayMode={isXrayMode}
                />
              </div>
            </div>

            {/* ── RIGHT PANEL (30%): Intelligence Panel — Human-Readable ── */}
            <div className="w-[30%] flex flex-col gap-2 min-h-0 overflow-y-auto overflow-x-hidden shrink-0 min-w-0 break-words pr-0.5">

              {/* ═══ OVERALL MATCH — Hero Score ═══ */}
              <div className={`relative overflow-hidden rounded-lg p-4 border-2 ${(results.fused_identity_score < 40.0) ? 'border-red-700/60 bg-gradient-to-br from-[#1a0505] to-[#0d0d0e]' : 'border-[#D4AF37]/50 bg-gradient-to-br from-[#1a170d] to-[#0d0d0e]'}`}>
                <div className={`absolute -top-6 -right-6 w-24 h-24 rounded-full ${(results.fused_identity_score < 40.0) ? 'bg-red-500/5' : 'bg-[#D4AF37]/5'}`}></div>
                <div className={`absolute -bottom-4 -left-4 w-16 h-16 rounded-full ${(results.fused_identity_score < 40.0) ? 'bg-red-500/5' : 'bg-[#D4AF37]/5'}`}></div>
                <div className="relative z-10">
                  <div className={`text-[8px] tracking-[0.3em] mb-1 ${(results.fused_identity_score < 40.0) ? 'text-red-400/70' : 'text-[#D4AF37]/70'}`}>FUSED SIMILARITY SCORE</div>
                  <div className="flex items-baseline gap-1.5 flex-wrap overflow-hidden min-w-0 w-full">
                    <span className={`text-4xl font-bold tabular-nums ${(results.fused_identity_score < 40.0) ? 'text-red-400' : 'text-[#D4AF37]'}`}>{results.fused_identity_score}</span>
                    <span className={`text-lg font-bold ${(results.fused_identity_score < 40.0) ? 'text-red-400/60' : 'text-[#D4AF37]/60'}`}>%</span>
                  </div>
                  {/* LR_total context */}
                  {results.audit_log?.lr_total != null && (
                    <div className="mt-1.5 flex items-center gap-2 max-w-full">
                      <span className="text-[8px] text-gray-500 tracking-wider shrink-0">LR<sub>total</sub></span>
                      <span className={`text-[11px] font-bold tabular-nums truncate ${(results.fused_identity_score < 40.0) ? 'text-red-400/80' : 'text-[#D4AF37]/90'}`}>{formatLRSci(results.audit_log.lr_total)}</span>
                    </div>
                  )}
                  {/* Score bar */}
                  <div className="mt-2 h-1.5 w-full bg-[#111] rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-700 ${results.fused_identity_score > 80 ? 'bg-gradient-to-r from-emerald-600 to-emerald-400' : results.fused_identity_score > 60 ? 'bg-gradient-to-r from-amber-600 to-amber-400' : 'bg-gradient-to-r from-red-700 to-red-500'}`}
                      style={{ width: `${Math.min(100, results.fused_identity_score)}%` }}
                    />
                  </div>
                  {/* Human-readable interpretation */}
                  <div className={`text-[10px] mt-2 font-medium ${(results.fused_identity_score < 40.0) ? 'text-red-300/80' : results.fused_identity_score > 80 ? 'text-emerald-300/80' : results.fused_identity_score > 60 ? 'text-amber-300/80' : 'text-red-300/80'}`}>
                    {results.fused_identity_score > 99 ? 'Extremely strong similarity detected' : results.fused_identity_score > 85 ? 'Very strong facial similarity detected' : results.fused_identity_score > 70 ? 'Moderate facial similarity detected' : results.fused_identity_score > 50 ? 'Weak similarity detected' : 'Very low similarity detected'}
                  </div>
                  <div className={`text-[8px] mt-1 ${(results.fused_identity_score < 40.0) ? 'text-red-400/40' : 'text-[#D4AF37]/40'}`}>Bayesian fusion of {results.marks_matched ? '4' : '3'} independent evidence channels below</div>
                </div>
              </div>

              {/* ═══ HOW WE SCORED THIS — Breakdown ═══ */}
              <div className="border border-[#1f1f1f] bg-[#0d0d0e] rounded-lg overflow-hidden">
                <div className="px-2.5 py-1.5 border-b border-[#1a1a1a] bg-[#111]">
                  <span className="text-[9px] text-gray-400 tracking-wider font-bold">HOW WE ANALYZED THIS</span>
                </div>

                {/* Tier 1: Face Shape & Identity */}
                <div className="p-2.5 border-b border-[#1a1a1a]">
                  <div className="flex items-baseline justify-between">
                    <h3 className="text-gray-300 text-[9px] tracking-wider font-bold">FACE RECOGNITION</h3>
                    <span className={`text-lg font-bold tabular-nums ${results.structural_score > 80 ? 'text-emerald-400' : results.structural_score > 60 ? 'text-amber-400' : 'text-red-400'}`}>{results.structural_score}%</span>
                  </div>
                  <div className="mt-1 h-1 w-full bg-[#111] rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${results.structural_score > 80 ? 'bg-emerald-500/70' : results.structural_score > 60 ? 'bg-amber-500/70' : 'bg-red-500/70'}`}
                      style={{ width: `${Math.min(100, results.structural_score)}%` }}
                    />
                  </div>
                  <p className="text-[9px] text-gray-500 mt-1.5 leading-relaxed">Do these faces belong to the same person? This is the primary test — an AI model maps each face into a numerical fingerprint and measures how similar they are.</p>
                  <div className="mt-1.5 flex items-center gap-1.5">
                    <div className={`w-1 h-1 rounded-full ${results.structural_score > 80 ? 'bg-emerald-500' : results.structural_score > 60 ? 'bg-amber-500' : 'bg-red-500'}`}></div>
                    <span className={`text-[8px] italic ${results.structural_score > 80 ? 'text-emerald-500/70' : results.structural_score > 60 ? 'text-amber-500/70' : 'text-red-500/70'}`}>
                      {results.structural_score > 85 ? 'Strong match — very likely the same person' : results.structural_score > 70 ? 'Possible match — further review recommended' : results.structural_score > 50 ? 'Unlikely match — faces differ significantly' : 'No match — these are different people'}
                    </span>
                  </div>
                  <div className="text-[7px] text-gray-700 mt-1 tracking-wide">60% of overall score · ArcFace 512-D CNN</div>
                </div>

                {/* Tier 2: Face Proportions */}
                <div className="p-2.5 border-b border-[#1a1a1a]">
                  <div className="flex items-baseline justify-between">
                    <h3 className="text-gray-300 text-[9px] tracking-wider font-bold">FACE PROPORTIONS</h3>
                    <span className={`text-lg font-bold tabular-nums ${
                      results.geometry_status && results.geometry_status !== 'OK'
                        ? 'text-gray-500'
                        : results.soft_biometrics_score > 80 ? 'text-emerald-400' : results.soft_biometrics_score > 60 ? 'text-amber-400' : 'text-red-400'
                    }`}>
                      {results.geometry_status && results.geometry_status !== 'OK' ? 'N/A' : `${results.soft_biometrics_score}%`}
                    </span>
                  </div>
                  <div className="mt-1 h-1 w-full bg-[#111] rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${
                        results.geometry_status && results.geometry_status !== 'OK'
                          ? 'bg-gray-700/50'
                          : results.soft_biometrics_score > 80 ? 'bg-emerald-500/70' : results.soft_biometrics_score > 60 ? 'bg-amber-500/70' : 'bg-red-500/70'
                      }`}
                      style={{ width: `${results.geometry_status && results.geometry_status !== 'OK' ? 0 : Math.min(100, results.soft_biometrics_score)}%` }}
                    />
                  </div>
                  <p className="text-[9px] text-gray-500 mt-1.5 leading-relaxed">Are the facial measurements similar? Compares the distances between eyes, nose width, jawline angle, and brow spacing — like a ruler measuring each face.</p>
                  <div className="mt-1.5 flex items-center gap-1.5">
                    <div className={`w-1 h-1 rounded-full ${
                      results.geometry_status && results.geometry_status !== 'OK'
                        ? 'bg-gray-500'
                        : results.soft_biometrics_score > 80 ? 'bg-emerald-500' : results.soft_biometrics_score > 60 ? 'bg-amber-500' : results.soft_biometrics_score > 0 ? 'bg-red-500' : 'bg-yellow-500'
                    }`}></div>
                    <span className={`text-[8px] italic ${
                      results.geometry_status && results.geometry_status !== 'OK'
                        ? 'text-gray-500/70'
                        : results.soft_biometrics_score > 80 ? 'text-emerald-500/70' : results.soft_biometrics_score > 60 ? 'text-amber-500/70' : results.soft_biometrics_score > 0 ? 'text-red-500/70' : 'text-yellow-500/70'
                    }`}>
                      {results.geometry_status && results.geometry_status !== 'OK'
                        ? (results.geometry_status === 'INVALID_IOD' ? 'Could not measure — eyes not clearly visible' : 'Could not measure — face angle or quality too low')
                        : results.soft_biometrics_score > 80 ? 'Proportions closely match' : results.soft_biometrics_score > 60 ? 'Proportions partially align' : results.soft_biometrics_score > 0 ? 'Proportions do not match' : 'Could not measure — face angle or quality too low'}
                    </span>
                  </div>
                  <div className="text-[7px] text-gray-700 mt-1 tracking-wide">25% of overall score · 12-point landmark geometry</div>
                </div>

                {/* Tier 3: Skin Texture */}
                <div className="p-2.5">
                  <div className="flex items-baseline justify-between">
                    <h3 className="text-gray-300 text-[9px] tracking-wider font-bold">SKIN TEXTURE</h3>
                    <span className={`text-lg font-bold tabular-nums ${results.micro_topology_score > 80 ? 'text-emerald-400' : results.micro_topology_score > 60 ? 'text-amber-400' : 'text-red-400'}`}>{results.micro_topology_score}%</span>
                  </div>
                  <div className="mt-1 h-1 w-full bg-[#111] rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${results.micro_topology_score > 80 ? 'bg-emerald-500/70' : results.micro_topology_score > 60 ? 'bg-amber-500/70' : 'bg-red-500/70'}`}
                      style={{ width: `${Math.min(100, results.micro_topology_score)}%` }}
                    />
                  </div>
                  <p className="text-[9px] text-gray-500 mt-1.5 leading-relaxed">Does the skin look similar? Analyzes pore patterns, wrinkle depth, scars, and surface texture. High scores can occur between people of similar age and ethnicity — this alone does not confirm identity.</p>
                  <div className="mt-1.5 flex items-center gap-1.5">
                    <div className={`w-1 h-1 rounded-full ${results.micro_topology_score > 80 ? 'bg-emerald-500' : results.micro_topology_score > 60 ? 'bg-amber-500' : 'bg-red-500'}`}></div>
                    <span className={`text-[8px] italic ${results.micro_topology_score > 80 ? 'text-emerald-500/70' : results.micro_topology_score > 60 ? 'text-amber-500/70' : 'text-red-500/70'}`}>
                      {results.micro_topology_score > 80 ? 'Similar skin texture detected' : results.micro_topology_score > 60 ? 'Partial texture similarity' : 'Skin textures differ'}
                    </span>
                  </div>
                  <div className="text-[7px] text-gray-700 mt-1 tracking-wide">15% of overall score · LBP texture analysis</div>
                </div>

                {/* Tier 4: Mark Correspondence — Bayesian LR (Only if marks found) */}
                {results.marks_matched !== undefined && results.marks_matched > 0 && (
                  <div className="p-2.5 border-t border-[#1a1a1a]">
                    <div className="flex items-baseline justify-between">
                      <h3 className="text-[#D4AF37] text-[9px] tracking-wider font-bold">MARK EVIDENCE (LR)</h3>
                      <span className="text-lg font-bold tabular-nums text-[#D4AF37] break-all whitespace-normal overflow-hidden">{formatLRSci(results.audit_log?.lr_marks)}</span>
                    </div>
                    {/* LR magnitude bar — log-scaled */}
                    <div className="mt-1 h-1 w-full bg-[#111] rounded-full overflow-hidden border border-[#D4AF37]/20">
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-[#D4AF37]/60 to-[#D4AF37] transition-all duration-700"
                        style={{ width: `${Math.min(100, results.audit_log?.lr_marks != null ? Math.min(100, Math.log10(Math.max(1, results.audit_log.lr_marks)) * 10) : 0)}%` }}
                      />
                    </div>
                    <p className="text-[8px] break-words text-[#D4AF37]/70 mt-1.5 leading-relaxed">Bayesian Likelihood Ratio from {results.marks_matched} matching scars, moles, and birthmarks. Values {'>'} 1 support visual similarity hypothesis; values {'>'} 10,000 constitute extremely strong similarity.</p>
                    {/* Individual mark LR breakdown */}
                    {results.audit_log?.lr_arcface != null && (
                      <div className="mt-1.5 flex items-center gap-3 flex-wrap">
                        <div className="flex items-center gap-1">
                          <span className="text-[7px] text-gray-500">LR<sub>arcface</sub></span>
                          <span className="text-[8px] font-bold text-[#D4AF37]/80 tabular-nums break-all whitespace-normal overflow-hidden">{formatLRSci(results.audit_log.lr_arcface)}</span>
                        </div>
                        <span className="text-[7px] text-gray-600">×</span>
                        <div className="flex items-center gap-1">
                          <span className="text-[7px] text-gray-500">LR<sub>marks</sub></span>
                          <span className="text-[8px] font-bold text-[#D4AF37]/80 tabular-nums break-all whitespace-normal overflow-hidden">{formatLRSci(results.audit_log.lr_marks)}</span>
                        </div>
                        <span className="text-[7px] text-gray-600">=</span>
                        <div className="flex items-center gap-1">
                          <span className="text-[7px] text-gray-500">LR<sub>total</sub></span>
                          <span className="text-[8px] font-bold text-[#D4AF37] tabular-nums break-all whitespace-normal overflow-hidden">{formatLRSci(results.audit_log.lr_total)}</span>
                        </div>
                      </div>
                    )}
                    <div className="mt-1.5 flex items-center gap-1.5">
                      <div className="w-1 h-1 rounded-full bg-[#D4AF37]"></div>
                      <span className="text-[8px] italic text-[#D4AF37]/80">
                        {results.marks_matched} corresponding marks matched across both faces
                      </span>
                    </div>
                    <div className="text-[7px] text-[#D4AF37]/50 mt-1 tracking-wide">Bayesian LR fusion · P(same|evidence) = LR/(LR+1)</div>
                  </div>
                )}
              </div>

              {/* ═══ VERDICT ═══ */}
              <div className={`rounded-lg overflow-hidden border-2 ${(results.fused_identity_score < 40.0) ? 'border-red-700/60' : (results.veto_triggered && results.fused_identity_score >= 40.0) ? 'border-amber-700/50' : 'border-emerald-700/40'}`}>
                <div className={`px-3 py-1.5 text-[9px] tracking-[0.15em] font-bold ${(results.fused_identity_score < 40.0) ? 'bg-red-900/40 text-red-300' : (results.veto_triggered && results.fused_identity_score >= 40.0) ? 'bg-amber-900/40 text-amber-400' : 'bg-emerald-900/30 text-emerald-300'}`}>
                  {(results.fused_identity_score < 40.0) ? '✗ VERDICT: NOT A MATCH' : (results.veto_triggered && results.fused_identity_score >= 40.0) ? '⚠ VERDICT: CONDITIONAL MATCH' : '✓ VERDICT: MATCH DETECTED'}
                </div>
                <div className={`px-3 py-3 ${(results.fused_identity_score < 40.0) ? 'bg-red-950/20' : 'bg-[#0d0d0e]'}`}>
                  <p className={`text-[11px] leading-relaxed break-all whitespace-normal overflow-hidden ${(results.fused_identity_score < 40.0) ? 'text-red-300/90' : 'text-gray-200'}`}>
                    {results.conclusion}
                  </p>
                  {(results.fused_identity_score < 40.0 && results.veto_triggered) && (
                    <div className="mt-2 px-2 py-1.5 bg-red-950/30 rounded border border-red-900/30">
                      <div className="flex items-center gap-1.5 mb-1">
                        <div className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse"></div>
                        <span className="text-[9px] text-red-500/80 tracking-wider font-bold">ARCFACE VETO TRIGGERED</span>
                      </div>
                      <p className="text-[8px] text-red-400/70 leading-relaxed break-words">Structural similarity fell below threshold. This is an automatic exclusion. Any subsequent Bayesian mark evidence has been overruled.</p>
                    </div>
                  )}
                  {(results.fused_identity_score < 40.0 && !results.veto_triggered) && (
                    <div className="mt-2 px-2 py-1.5 bg-red-950/30 rounded border border-red-900/30">
                      <p className="text-[8px] text-red-400/70 leading-relaxed break-words">The face recognition AI returned a similarity of {results.structural_score}%, which is below the 40% minimum required to consider a potential match. This is an automatic exclusion.</p>
                    </div>
                  )}
                  {(!results.veto_triggered && results.fused_identity_score >= 40.0) && (
                    <div className="mt-2 flex items-center gap-1.5">
                      <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></div>
                      <span className="text-[9px] text-emerald-500/70 tracking-wider">No discrepancies found across structural tests</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Export Dossier */}
              <button
                onClick={generateForensicReport}
                disabled={isExporting}
                className={`w-full py-2.5 text-[10px] font-bold tracking-[0.2em] border-2 rounded-lg transition-all shrink-0 ${
                  isExporting
                    ? 'border-[#333] bg-[#111] text-gray-500 cursor-wait'
                    : 'border-[#D4AF37]/50 bg-[#0a0a0a] text-[#D4AF37] hover:bg-[#D4AF37]/10 hover:border-[#D4AF37] hover:shadow-[0_0_20px_rgba(212,175,55,0.4)]'
                }`}
              >
                {isExporting ? 'COMPILING...' : '↓ DOWNLOAD FULL REPORT'}
              </button>
            </div>
            </div>

            {/* ── Full-Width Technical Details Block ── */}
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
                  VIEW DETAILED TECHNICAL BREAKDOWN
                </span>
                <span className="text-lg font-bold">{auditExpanded ? '−' : '+'}</span>
              </button>

              {/* ── Technical Details (3-column) ── */}
              {auditExpanded && results.audit_log && (
                <div className="border border-[#1a1a0a] bg-[#000000] rounded p-2.5 font-mono text-[9px] leading-relaxed shadow-[inset_0_0_30px_rgba(0,0,0,0.5)]">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-1 w-full min-w-0">

                    {/* Block 1: Confidence & Accuracy */}
                    <div className="border border-[#1a2a1a] rounded p-2 bg-[#010201] min-w-0">
                      <div className="text-green-500/80 tracking-[0.2em] mb-1 border-b border-green-900/30 pb-1 text-[8px]">▸ CONFIDENCE &amp; ACCURACY</div>
                      <p className="text-[7px] text-gray-600 mb-1.5 leading-relaxed">How confident is the system in this result? Lower error rates mean higher reliability.</p>
                      <div className="space-y-0.5 pl-1">
                        <div className="flex justify-between"><span className="text-gray-500">Error Probability</span><span className={`font-bold ${results.audit_log.false_acceptance_rate === 'UNCALIBRATED' || results.audit_log.false_acceptance_rate === 'Inconclusive' ? 'text-yellow-400' : results.audit_log.false_acceptance_rate === 'DIFFERENT IDENTITIES' ? 'text-red-400' : 'text-green-400'}`}>{results.audit_log.false_acceptance_rate}</span></div>
                        <div className="flex justify-between"><span className="text-gray-500">Confidence Level</span><span className={`font-bold ${results.audit_log.statistical_certainty === 'UNCALIBRATED' ? 'text-yellow-400' : results.audit_log.statistical_certainty.startsWith('<') || results.audit_log.statistical_certainty.startsWith('0%') ? 'text-red-400' : 'text-green-400'}`}>{results.audit_log.statistical_certainty}</span></div>
                        <div className="flex justify-between group relative"><span className="text-gray-500">Face Points Mapped</span><span className="text-white font-bold">{results.audit_log.nodes_mapped}/468</span><div className="pointer-events-none absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 opacity-0 group-hover:opacity-100 transition-opacity duration-200 z-50"><div className="bg-[#111] border border-[#333] rounded px-3 py-2 text-[8px] text-gray-300 font-mono leading-relaxed shadow-[0_4px_20px_rgba(0,0,0,0.8)]">The system maps up to 468 points on each face to measure geometry. More points = more accurate comparison.<div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-[#333]"></div></div></div></div>
                        <div className="flex justify-between"><span className="text-gray-500">Raw Similarity Score</span><span className="text-white font-bold">{results.audit_log.raw_cosine_score.toFixed(6)}</span></div>
                        {results.audit_log.calibration_benchmark && (
                          <div className="flex justify-between mt-0.5"><span className="text-gray-500">Tested Against</span><span className={`font-bold ${results.audit_log.calibration_benchmark === 'N/A' ? 'text-yellow-400' : 'text-green-400'}`}>{results.audit_log.calibration_benchmark}{results.audit_log.calibration_pairs ? ` (${results.audit_log.calibration_pairs.toLocaleString()} pairs)` : ''}</span></div>
                        )}
                      </div>
                    </div>

                    {/* Block 2: Image Quality & Authenticity */}
                    <div className="border border-[#2a1a1a] rounded p-2 bg-[#020101] min-w-0">
                      <div className="text-cyan-500/80 tracking-[0.2em] mb-1 border-b border-cyan-900/30 pb-1 text-[8px]">▸ IMAGE QUALITY &amp; AUTHENTICITY</div>
                      <p className="text-[7px] text-gray-600 mb-1.5 leading-relaxed">How were the photos corrected for comparison, and are they real photographs?</p>
                      <div className="space-y-0.5 pl-1">
                        {results.audit_log.alignment_variance && (<>
                          <div className="flex justify-between"><span className="text-gray-500">Left-Right Correction</span><span className="text-cyan-300">{results.audit_log.alignment_variance.yaw}</span></div>
                          <div className="flex justify-between"><span className="text-gray-500">Up-Down Correction</span><span className="text-cyan-300">{results.audit_log.alignment_variance.pitch}</span></div>
                          <div className="flex justify-between"><span className="text-gray-500">Tilt Correction</span><span className="text-cyan-300">{results.audit_log.alignment_variance.roll}</span></div>
                        </>)}
                        {results.audit_log.liveness_check && (<>
                          <div className="flex justify-between mt-1"><span className="text-gray-500">Detection Method</span><span className="text-cyan-300 font-bold">{results.audit_log.liveness_check.method}</span></div>
                          <div className="flex justify-between"><span className="text-gray-500">Fake Photo Risk</span><span className={`font-bold ${results.audit_log.liveness_check.status.includes('PASSED') || results.audit_log.liveness_check.status.includes('VERIFIED') || results.audit_log.liveness_check.status.includes('LIVE') ? 'text-green-400' : 'text-red-400'}`}>{results.audit_log.liveness_check.spoof_probability}</span></div>
                          <div className="flex justify-between"><span className="text-gray-500">Authenticity</span><span className={`font-bold ${results.audit_log.liveness_check.status.includes('PASSED') || results.audit_log.liveness_check.status.includes('VERIFIED') || results.audit_log.liveness_check.status.includes('LIVE') ? 'text-green-400' : 'text-red-400'}`}>{results.audit_log.liveness_check.status}</span></div>
                          {results.audit_log.liveness_check.laplacian_variance != null && (
                            <div className="flex justify-between"><span className="text-gray-500">Image Sharpness</span><span className="text-cyan-300">{results.audit_log.liveness_check.laplacian_variance}</span></div>
                          )}
                        </>)}
                      </div>
                    </div>

                    {/* Block 3: Security & Data Integrity */}
                    <div className="border border-[#1a1a2a] rounded p-2 bg-[#010102] min-w-0">
                      <div className="text-amber-500/80 tracking-[0.2em] mb-1 border-b border-amber-900/30 pb-1 text-[8px]">▸ SECURITY &amp; DATA INTEGRITY</div>
                      <p className="text-[7px] text-gray-600 mb-1.5 leading-relaxed">Cryptographic proof that the biometric data was not tampered with during analysis.</p>
                      <div className="space-y-0.5 pl-1">
                        {results.audit_log.vector_hash && (
                          <div><span className="text-gray-500">Digital Fingerprint</span><div className="text-amber-300/80 text-[8px] break-all whitespace-pre-wrap w-full min-w-0 mt-0.5">{results.audit_log.vector_hash}</div></div>
                        )}
                        {results.audit_log.crypto_envelope && (<>
                          <div className="flex justify-between items-start gap-2 break-words w-full min-w-0 mt-1"><span className="text-gray-500">Encryption Standard</span><span className="text-amber-300">{results.audit_log.crypto_envelope.standard}</span></div>
                          <div className="flex justify-between items-start gap-2 break-words w-full min-w-0"><span className="text-gray-500">Decryption Speed</span><span className="text-amber-300">{results.audit_log.crypto_envelope.decryption_time}</span></div>
                        </>)}
                        {results.audit_log.matched_user_id && (
                          <div className="flex justify-between items-start gap-2 break-words w-full min-w-0 mt-1"><span className="text-gray-500">Matched Profile ID</span><span className="text-white">{results.audit_log.matched_user_id}</span></div>
                        )}
                        {results.audit_log.person_name && (
                          <div className="flex justify-between items-start gap-2 break-words w-full min-w-0"><span className="text-gray-500">Matched Name</span><span className="text-white">{results.audit_log.person_name}</span></div>
                        )}
                        {results.audit_log.license_short_name && (
                          <div className="flex justify-between items-start gap-2 break-words w-full min-w-0"><span className="text-gray-500">Image License</span><span className="text-gray-400">{results.audit_log.license_short_name}</span></div>
                        )}
                      </div>
                    </div>

                    {/* Block 4: Bayesian Evidence — Forensic Trail */}
                    <div className="border border-[#2a1a2a] rounded p-2 bg-[#020102] min-w-0">
                      <div className="text-purple-400/80 tracking-[0.2em] mb-1 border-b border-purple-900/30 pb-1 text-[8px]">▸ BAYESIAN EVIDENCE</div>
                      <p className="text-[8px] break-words text-gray-600 mb-1.5 leading-relaxed">Likelihood Ratios quantifying the strength of evidence for visual similarity.</p>
                      <div className="space-y-0.5 pl-1">
                        <div className="flex justify-between"><span className="text-gray-500">LR<sub>arcface</sub></span><span className="text-purple-300 font-bold break-all whitespace-normal overflow-hidden">{formatLRSci(results.audit_log.lr_arcface)}</span></div>
                        <div className="flex justify-between"><span className="text-gray-500">LR<sub>marks</sub></span><span className="text-purple-300 font-bold break-all whitespace-normal overflow-hidden">{formatLRSci(results.audit_log.lr_marks)}</span></div>
                        <div className="flex justify-between"><span className="text-gray-500">LR<sub>total</sub></span><span className="text-[#D4AF37] font-bold break-all whitespace-normal overflow-hidden">{formatLRSci(results.audit_log.lr_total)}</span></div>
                        <div className="flex justify-between mt-1 pt-1 border-t border-purple-900/20"><span className="text-gray-500">Similarity Probability</span><span className="text-[#D4AF37] font-bold">{results.audit_log.posterior_probability != null ? `${(results.audit_log.posterior_probability * 100).toFixed(6)}%` : 'N/A'}</span></div>
                        {results.audit_log.mark_lrs && results.audit_log.mark_lrs.length > 0 && (
                          <div className="mt-1 pt-1 border-t border-purple-900/20">
                            <div className="text-gray-500 mb-0.5">Individual Mark LRs ({results.audit_log.mark_lrs.length})</div>
                            <div className="flex flex-wrap gap-1">
                              {results.audit_log.mark_lrs.map((lr, i) => (
                                <span key={i} className={`text-[7px] px-1 py-0.5 rounded ${lr > 10 ? 'bg-[#D4AF37]/15 text-[#D4AF37]' : 'bg-gray-800 text-gray-400'}`}>{lr.toFixed(1)}</span>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                  <div className="mt-2 pt-1.5 border-t border-[#1a1a0a] text-[7px] text-gray-700 tracking-widest text-center">REPORT GENERATED AT {new Date().toISOString()}</div>
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

