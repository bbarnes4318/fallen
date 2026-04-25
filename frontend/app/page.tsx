'use client';

import React, { useState, useRef, useEffect } from 'react';
import SymmetryMerge from '@/components/SymmetryMerge';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function Home() {
  const [galleryFile, setGalleryFile] = useState<File | null>(null);
  const [probeFile, setProbeFile] = useState<File | null>(null);
  const [galleryPreview, setGalleryPreview] = useState<string>('');
  const [probePreview, setProbePreview] = useState<string>('');

  const [step, setStep] = useState<'idle' | 'uploading' | 'frontalizing' | 'calculating' | 'complete' | 'error'>('idle');
  const [errorMsg, setErrorMsg] = useState('');

  const [results, setResults] = useState<any>(null);
  const [galleryUrl, setGalleryUrl] = useState('');
  const [probeUrl, setProbeUrl] = useState('');
  const [isXrayMode, setIsXrayMode] = useState(false);

  const [token, setToken] = useState<string | null>(null);
  const [password, setPassword] = useState('');
  const [loginError, setLoginError] = useState('');

  useEffect(() => {
    const savedToken = localStorage.getItem('operator_token');
    if (savedToken) setToken(savedToken);
  }, []);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoginError('');
    try {
      const res = await fetch(`${API_URL}/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password })
      });
      if (!res.ok) throw new Error('Invalid credentials');
      const data = await res.json();
      setToken(data.access_token);
      localStorage.setItem('operator_token', data.access_token);
    } catch (err: any) {
      setLoginError(err.message || 'Login failed');
    }
  };

  const handleLogout = () => {
    setToken(null);
    localStorage.removeItem('operator_token');
    setStep('idle');
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>, type: 'gallery' | 'probe') => {
    const file = e.target.files?.[0];
    if (file) {
      if (type === 'gallery') {
        setGalleryFile(file);
        setGalleryPreview(URL.createObjectURL(file));
      } else {
        setProbeFile(file);
        setProbePreview(URL.createObjectURL(file));
      }
    }
  };

  const startSequence = async () => {
    if (!galleryFile || !probeFile) return;
    try {
      setStep('uploading');
      
      // 1. Get Pre-Signed URLs from FastAPI (Pass Content-Types dynamically)
      const urlRes = await fetch(`${API_URL}/generate-upload-urls`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          gallery_content_type: galleryFile.type,
          probe_content_type: probeFile.type
        })
      });
      if (urlRes.status === 401 || urlRes.status === 403) {
        handleLogout();
        return;
      }
      if (!urlRes.ok) throw new Error("Failed to secure upload channels.");
      const { gallery_upload_url, probe_upload_url, gallery_gs_uri, probe_gs_uri } = await urlRes.json();
      
      // 2. Direct Client Upload to GCS
      await Promise.all([
        fetch(gallery_upload_url, {
          method: 'PUT',
          body: galleryFile,
          headers: { 'Content-Type': galleryFile.type }
        }),
        fetch(probe_upload_url, {
          method: 'PUT',
          body: probeFile,
          headers: { 'Content-Type': probeFile.type }
        })
      ]);
      
      setGalleryUrl(gallery_gs_uri);
      setProbeUrl(probe_gs_uri);
      
      // UX Pacing
      setStep('frontalizing');
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      setStep('calculating');
      
      // 3. The Verification Call
      const verifyRes = await fetch(`${API_URL}/verify/fuse`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          gallery_url: gallery_gs_uri,
          probe_url: probe_gs_uri
        })
      });
      
      if (verifyRes.status === 401 || verifyRes.status === 403) {
        handleLogout();
        return;
      }
      if (!verifyRes.ok) throw new Error("Verification pipeline failed.");
      const data = await verifyRes.json();
      
      setResults(data);
      setStep('complete');
      
    } catch (err: any) {
      setErrorMsg(err.message || 'An unknown error occurred');
      setStep('error');
    }
  };

  if (!token) {
    return (
      <main className="min-h-screen bg-[#020202] text-gray-200 font-sans flex items-center justify-center selection:bg-[#D4AF37] selection:text-black">
        <div className="w-full max-w-md p-8 bg-[#0a0a0a] border border-[#1f1f1f] rounded-xl shadow-2xl">
          <div className="text-center mb-8">
            <h1 className="text-2xl font-bold font-mono tracking-widest text-white mb-2">
              ZERO-TRUST <span className="text-[#D4AF37]">PORTAL</span>
            </h1>
            <p className="text-gray-500 font-mono text-xs">AWAITING OPERATOR CREDENTIALS</p>
          </div>
          
          <form onSubmit={handleLogin} className="space-y-6">
            <div>
              <input 
                type="password" 
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="ENTER PASSPHRASE"
                className="w-full bg-[#111] border border-[#333] text-white p-3 font-mono text-center tracking-widest focus:outline-none focus:border-[#D4AF37] transition-colors"
              />
            </div>
            
            {loginError && (
              <div className="text-red-500 text-xs font-mono text-center">{loginError}</div>
            )}
            
            <button 
              type="submit"
              className="w-full py-3 bg-[#D4AF37] text-black font-mono font-bold tracking-widest hover:bg-[#b5952f] transition-colors shadow-[0_0_15px_rgba(212,175,55,0.2)]"
            >
              AUTHENTICATE
            </button>
          </form>
        </div>
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-[#020202] text-gray-200 font-sans p-8 selection:bg-[#D4AF37] selection:text-black">
      <div className="max-w-6xl mx-auto">
        <header className="mb-12 border-b border-[#333] pb-6 flex justify-between items-end">
          <div>
            <h1 className="text-3xl font-bold font-mono tracking-widest text-white">
              BIOMETRIC VERIFICATION <span className="text-[#D4AF37]">ENGINE</span>
            </h1>
            <p className="text-gray-500 font-mono text-sm mt-2">Level 3 Topology & 3DMM Frontalization Pipeline</p>
          </div>
          <button 
            onClick={handleLogout}
            className="text-xs font-mono text-gray-500 hover:text-red-400 transition-colors border border-transparent hover:border-red-900/50 px-3 py-1 rounded"
          >
            END SESSION
          </button>
        </header>

        {step === 'idle' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
            {/* Gallery Upload */}
            <div className="border border-dashed border-[#444] rounded-xl p-8 flex flex-col items-center justify-center bg-[#0a0a0a] hover:border-[#D4AF37] transition-colors relative">
              <input 
                type="file" 
                accept="image/*" 
                onChange={(e) => handleFileChange(e, 'gallery')}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
              {galleryPreview ? (
                <img src={galleryPreview} alt="Gallery Preview" className="h-48 object-cover rounded shadow-lg" />
              ) : (
                <>
                  <div className="w-12 h-12 rounded-full bg-[#111] border border-[#333] flex items-center justify-center mb-4 text-[#D4AF37]">
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4"></path></svg>
                  </div>
                  <span className="font-mono text-gray-400">UPLOAD GALLERY (REFERENCE)</span>
                </>
              )}
            </div>

            {/* Probe Upload */}
            <div className="border border-dashed border-[#444] rounded-xl p-8 flex flex-col items-center justify-center bg-[#0a0a0a] hover:border-[#D4AF37] transition-colors relative">
              <input 
                type="file" 
                accept="image/*" 
                onChange={(e) => handleFileChange(e, 'probe')}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
              {probePreview ? (
                <img src={probePreview} alt="Probe Preview" className="h-48 object-cover rounded shadow-lg" />
              ) : (
                <>
                  <div className="w-12 h-12 rounded-full bg-[#111] border border-[#333] flex items-center justify-center mb-4 text-[#D4AF37]">
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4"></path></svg>
                  </div>
                  <span className="font-mono text-gray-400">UPLOAD PROBE (TARGET)</span>
                </>
              )}
            </div>
          </div>
        )}

        {step === 'idle' && (
          <div className="flex justify-center">
            <button 
              onClick={startSequence}
              disabled={!galleryFile || !probeFile}
              className={`px-8 py-3 font-mono font-bold tracking-widest rounded-sm border ${galleryFile && probeFile ? 'bg-[#D4AF37] text-black border-[#D4AF37] hover:bg-[#b5952f] shadow-[0_0_20px_rgba(212,175,55,0.3)]' : 'bg-[#111] text-gray-500 border-[#333] cursor-not-allowed'}`}
            >
              INITIALIZE VERIFICATION
            </button>
          </div>
        )}

        {/* Loading States */}
        {['uploading', 'frontalizing', 'calculating'].includes(step) && (
          <div className="flex flex-col items-center justify-center h-64 border border-[#1f1f1f] bg-[#0a0a0a] rounded-xl shadow-2xl">
            <div className="w-16 h-16 border-4 border-[#333] border-t-[#D4AF37] rounded-full animate-spin mb-6"></div>
            <p className="font-mono text-lg text-gray-300 tracking-widest animate-pulse">
              {step === 'uploading' && "UPLOADING TO SECURE VAULT..."}
              {step === 'frontalizing' && "RUNNING 3DMM FRONTALIZATION..."}
              {step === 'calculating' && "CALCULATING FUSED IDENTITY SCORE..."}
            </p>
          </div>
        )}

        {step === 'error' && (
          <div className="p-6 border border-red-900 bg-red-950/30 text-red-400 rounded-xl font-mono text-center">
            <p className="font-bold mb-2">SYSTEM ERROR</p>
            <p>{errorMsg}</p>
            <button onClick={() => setStep('idle')} className="mt-4 px-4 py-2 bg-red-900/50 hover:bg-red-900/80 rounded border border-red-700">RESET</button>
          </div>
        )}

        {/* Results Dashboard */}
        {step === 'complete' && results && (
          <div className="space-y-8 animate-in fade-in zoom-in duration-500">
            {/* Action Bar / Toggle */}
            <div className="flex justify-end mb-2">
              <button
                onClick={() => setIsXrayMode(!isXrayMode)}
                className={`flex items-center space-x-3 px-4 py-2 border rounded-md font-mono text-xs tracking-widest transition-all ${
                  isXrayMode 
                    ? 'border-[#D4AF37] bg-[#D4AF37]/10 text-[#D4AF37] shadow-[0_0_15px_rgba(212,175,55,0.2)]' 
                    : 'border-[#333] bg-[#111] text-gray-500 hover:text-gray-300 hover:border-[#444]'
                }`}
              >
                <div className={`w-3 h-3 rounded-full ${isXrayMode ? 'bg-[#D4AF37] animate-pulse' : 'bg-gray-600'}`}></div>
                <span>FORENSIC X-RAY MODE</span>
              </button>
            </div>

            {/* The Canvas Symmetry Merge using local Object URLs for instant rendering or XAI Heatmaps */}
            <SymmetryMerge 
              galleryImageSrc={isXrayMode ? results.gallery_heatmap_b64 : galleryPreview} 
              probeImageSrc={isXrayMode ? results.probe_heatmap_b64 : probePreview} 
            />

            {/* The Scoring Dashboard */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="col-span-1 border border-[#1f1f1f] bg-[#0a0a0a] rounded-xl p-6">
                <h3 className="text-gray-500 font-mono text-xs mb-4">TIER 1: STRUCTURAL</h3>
                <div className="text-4xl font-mono text-white">{results.structural_score}%</div>
              </div>
              <div className="col-span-1 border border-[#1f1f1f] bg-[#0a0a0a] rounded-xl p-6">
                <h3 className="text-gray-500 font-mono text-xs mb-4">TIER 2: SOFT BIOMETRICS</h3>
                <div className="text-4xl font-mono text-white">{results.soft_biometrics_score}%</div>
              </div>
              <div className="col-span-1 border border-[#1f1f1f] bg-[#0a0a0a] rounded-xl p-6">
                <h3 className="text-gray-500 font-mono text-xs mb-4">TIER 3: MICRO-TOPOLOGY</h3>
                <div className="text-4xl font-mono text-white">{results.micro_topology_score}%</div>
              </div>
              <div className="col-span-1 border border-[#D4AF37]/50 bg-[#1a170d] rounded-xl p-6 relative overflow-hidden">
                <div className="absolute top-0 right-0 w-16 h-16 bg-[#D4AF37]/10 rounded-bl-full"></div>
                <h3 className="text-[#D4AF37] font-mono text-xs mb-4">FUSED IDENTITY SCORE</h3>
                <div className="text-4xl font-mono text-[#D4AF37] font-bold">{results.fused_identity_score}%</div>
              </div>
            </div>

            {/* Veto Logic and Conclusion */}
            <div className={`p-6 border rounded-xl font-mono flex flex-col md:flex-row justify-between items-center ${results.veto_triggered ? 'border-red-900 bg-red-950/30' : 'border-[#1f1f1f] bg-[#0a0a0a]'}`}>
              <div>
                <h3 className="text-gray-500 text-xs mb-1">FORENSIC CONCLUSION</h3>
                <p className={`text-xl ${results.veto_triggered ? 'text-red-400 font-bold' : 'text-gray-200'}`}>
                  {results.conclusion}
                </p>
              </div>
              {results.veto_triggered && (
                <div className="px-4 py-2 mt-4 md:mt-0 bg-red-900 text-red-100 text-sm tracking-widest border border-red-500">
                  ACE-V VETO TRIGGERED
                </div>
              )}
              {!results.veto_triggered && (
                <div className="px-4 py-2 mt-4 md:mt-0 bg-[#0a0a0a] text-green-500 text-sm tracking-widest border border-green-900/50">
                  NO BIOLOGICAL DISCREPANCY
                </div>
              )}
            </div>
            
            <div className="flex justify-center mt-8">
              <button 
                onClick={() => {
                  setStep('idle');
                  setGalleryFile(null);
                  setProbeFile(null);
                  setGalleryPreview('');
                  setProbePreview('');
                  setIsXrayMode(false);
                  setResults(null);
                }}
                className="px-6 py-2 border border-[#333] text-gray-400 hover:text-white hover:border-gray-500 rounded font-mono text-sm transition-colors"
              >
                RUN NEW COMPARISON
              </button>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
