import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Fallen — Biometric Architecture",
  description:
    "High-reliability similarity analysis fusing 512-D deep neural embeddings with Bayesian Likelihood Ratio metrics. Tier 1–4 pipeline architecture, AES-256-GCM envelope encryption.",
};

/* ── Design Tokens ── */
const C = {
  bg: "#13233A",
  bgDeep: "#0D1A2C",
  bgPanel: "#182C46",
  bgPanelHover: "#1E3552",
  gold: "#D0A85C",
  goldMuted: "#A8894A",
  goldDim: "rgba(208,168,92,0.12)",
  text: "#E7ECF4",
  textMuted: "#8A9BB5",
  textDim: "#5B6E88",
  border: "#2A3E58",
  borderGold: "rgba(208,168,92,0.25)",
  red: "#E05252",
  green: "#3DDC84",
  cyan: "#4AC8DB",
  purple: "#A78BDB",
} as const;

/* ── Tier Data ── */
const TIERS = [
  {
    id: "01",
    label: "TIER 1",
    title: "Structural Identity",
    subtitle: "ArcFace Deep Neural Embedding",
    color: C.green,
    borderColor: "rgba(61,220,132,0.3)",
    bgAccent: "rgba(61,220,132,0.06)",
    metric: "Deep",
    metricLabel: "Embedding Vector",
    method: "Cosine Similarity",
    threshold: "Distance Threshold",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-5 h-5">
        <path d="M12 2a7 7 0 017 7v1a7 7 0 01-14 0V9a7 7 0 017-7z" />
        <path d="M5.5 21a8.38 8.38 0 0113 0" />
        <circle cx="9" cy="10" r="1" fill="currentColor" />
        <circle cx="15" cy="10" r="1" fill="currentColor" />
      </svg>
    ),
    body: "The primary gatekeeper. Each face is projected through a deep neural network into a multidimensional hypersphere. The cosine distance between two embeddings is the core structural similarity measure. Scores below a proprietary threshold trigger an automatic exclusion — the biometric equivalent of an ACE-V elimination.",
    details: [
      "Model: Proprietary Neural Embedding Network",
      "Embedding normalization: L2-normalized unit vectors",
      "Distance metric: 1 − cos(θ) mapped to similarity scale",
      "Calibrated against large-scale biometric benchmarks",
    ],
  },
  {
    id: "02",
    label: "TIER 2",
    title: "Geometric Biometrics",
    subtitle: "Scale-Invariant Facial Ratios",
    color: C.cyan,
    borderColor: "rgba(74,200,219,0.3)",
    bgAccent: "rgba(74,200,219,0.06)",
    metric: "Key",
    metricLabel: "Facial Ratios",
    method: "L2 Distance",
    threshold: "Euclidean Norm",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-5 h-5">
        <rect x="3" y="3" width="18" height="18" />
        <line x1="3" y1="12" x2="21" y2="12" />
        <line x1="12" y1="3" x2="12" y2="21" />
        <line x1="3" y1="3" x2="21" y2="21" opacity="0.4" />
      </svg>
    ),
    body: "Anatomically-grounded facial ratios — inter-pupillary distance normalized against face width, nose length to face height, jaw width to cheekbone span — are extracted from dense facial landmarks. All ratios are scale-invariant, immune to distance-from-camera artifacts. The L2 (Euclidean) distance between ratio vectors quantifies geometric divergence.",
    details: [
      "Landmark source: Dense topometric mesh",
      "Ratio set: Proprietary craniometric proportions (Procrustes-aligned)",
      "Invariance: scale, rotation, and translation normalized",
      "Contributes to weighted heuristic fusion",
    ],
  },
  {
    id: "03",
    label: "TIER 3",
    title: "Micro-Topology",
    subtitle: "Epidermal Texture Analysis",
    color: C.purple,
    borderColor: "rgba(167,139,219,0.3)",
    bgAccent: "rgba(167,139,219,0.06)",
    metric: "LBP",
    metricLabel: "Texture Descriptor",
    method: "Chi-Squared Distance",
    threshold: "χ² Divergence",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-5 h-5">
        <circle cx="6" cy="6" r="1.5" />
        <circle cx="12" cy="4" r="1" />
        <circle cx="18" cy="7" r="1.5" />
        <circle cx="8" cy="12" r="1" />
        <circle cx="15" cy="13" r="2" />
        <circle cx="5" cy="18" r="1.5" />
        <circle cx="12" cy="19" r="1" />
        <circle cx="19" cy="17" r="1.5" />
      </svg>
    ),
    body: "Local Binary Patterns (LBP) extract micro-textural signatures from the skin surface — pore density, wrinkle topology, and pigmentation gradients. The Chi-Squared distance between LBP histograms captures differences invisible to neural embeddings: aging patterns, sun damage, and dermatological conditions that persist across years.",
    details: [
      "Algorithm: Uniform Local Binary Patterns",
      "Histogram: Rotation-invariant LBP distribution",
      "Distance: χ² divergence",
      "Contributes to weighted heuristic fusion",
    ],
  },
  {
    id: "04",
    label: "TIER 4",
    title: "Bayesian Mark Correspondence",
    subtitle: "Likelihood Ratio",
    color: C.gold,
    borderColor: C.borderGold,
    bgAccent: C.goldDim,
    metric: "LR",
    metricLabel: "Likelihood Ratio",
    method: "Bayesian Fusion",
    threshold: "P(same) = LR/(LR+1)",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-5 h-5">
        <path d="M4 20l4-12 4 8 4-16 4 8" />
        <line x1="2" y1="20" x2="22" y2="20" opacity="0.3" />
      </svg>
    ),
    body: "The crown of the pipeline. Scars, moles, birthmarks, and surgical marks are detected, localized in Procrustes-normalized face space, and matched via Hungarian optimization. Each matched mark pair produces an individual Likelihood Ratio by comparing observed delta vectors against the background population (H_d) and intra-person (H_p) distributions. The product of all mark LRs is fused with LR_arcface to yield a final posterior probability.",
    details: [
      "Population model: Multivariate Statistical Distributions",
      "Calibration: Institutional-scale calibration dataset",
      "Mark features: Spatial, textural, and geometric deltas",
      "Fusion: LR_total = LR_arcface × Π(LR_markᵢ) → Posterior = LR/(LR+1)",
    ],
  },
];

const SECURITY_ITEMS = [
  {
    title: "Envelope Encryption",
    standard: "AES-256-GCM",
    detail: "Application-level envelope encryption protects biometric embeddings at rest and in transit. Data Encryption Keys (DEKs) are generated per-vector and encrypted by a root Key Encryption Key (KEK) managed in Google Cloud KMS. Decryption requires both the encrypted DEK and KMS access — compromise of either alone yields nothing.",
    specs: ["Algorithm: AES-256-GCM (AEAD)", "KEK: GCP Cloud KMS (HSM-backed)", "Per-vector DEK rotation", "Zero plaintext persistence"],
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-6 h-6">
        <rect x="5" y="11" width="14" height="10" />
        <path d="M12 3a4 4 0 014 4v4H8V7a4 4 0 014-4z" />
        <circle cx="12" cy="16" r="1.5" fill="currentColor" />
      </svg>
    ),
  },
  {
    title: "Presentation Attack Detection",
    standard: "PAD / ISO 30107",
    detail: "Multi-layered anti-spoofing gate combining frequency-domain analysis, Laplacian variance sharpness testing, and statistical artifact detection. Screens for printed photos, digital displays, 3D masks, and deepfake injection before biometric processing begins.",
    specs: ["Laplacian variance (blur detection)", "Frequency spectrum analysis", "JPEG artifact quantization", "Status: PASS required to proceed"],
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-6 h-6">
        <path d="M12 2l8 4v6c0 5.25-3.5 10-8 11-4.5-1-8-5.75-8-11V6l8-4z" />
        <path d="M9 12l2 2 4-4" />
      </svg>
    ),
  },
  {
    title: "Secure Chain of Custody",
    standard: "SHA-256 / RFC 3161",
    detail: "Pre-decode binary SHA-256 hashes of all evidentiary images are computed before any processing begins, establishing an immutable chain of custody. Pipeline version, dependency manifests, and all intermediate computations are logged to a PostgreSQL audit ledger with server-authoritative timestamps.",
    specs: ["Pre-decode SHA-256 image hashing", "Pipeline version pinning", "Dependency manifest locking", "Immutable PostgreSQL audit ledger"],
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-6 h-6">
        <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8l-6-6z" />
        <polyline points="14,2 14,8 20,8" />
        <line x1="9" y1="13" x2="15" y2="13" />
        <line x1="9" y1="17" x2="15" y2="17" />
      </svg>
    ),
  },
];

export default function ArchitecturePage() {
  return (
    <main
      className="min-h-screen font-[family-name:var(--font-geist-mono)]"
      style={{ background: C.bg, color: C.text }}
    >
      {/* ═══════════════════════════════════════════════════════════
          HERO
      ═══════════════════════════════════════════════════════════ */}
      <section className="relative overflow-hidden">
        {/* Grid pattern overlay */}
        <div
          className="absolute inset-0 pointer-events-none opacity-[0.03]"
          style={{
            backgroundImage: `linear-gradient(${C.gold} 1px, transparent 1px), linear-gradient(90deg, ${C.gold} 1px, transparent 1px)`,
            backgroundSize: "60px 60px",
          }}
        />
        {/* Diagonal accent */}
        <div
          className="absolute -top-32 -right-32 w-[500px] h-[500px] pointer-events-none"
          style={{
            background: `linear-gradient(135deg, rgba(208,168,92,0.08) 0%, transparent 60%)`,
          }}
        />

        <div className="relative max-w-[1200px] mx-auto px-6 pt-16 pb-20">
          {/* Breadcrumb */}
          <div className="flex items-center gap-2 mb-10" style={{ color: C.textDim }}>
            <a href="/" className="text-[11px] tracking-[0.2em] uppercase hover:underline" style={{ color: C.goldMuted }}>
              Fallen
            </a>
            <span className="text-[10px]">/</span>
            <span className="text-[11px] tracking-[0.2em] uppercase">Architecture</span>
          </div>

          {/* Classification banner */}
          <div
            className="inline-flex items-center gap-2 px-3 py-1.5 mb-6 text-[10px] tracking-[0.3em] uppercase font-bold"
            style={{
              background: C.goldDim,
              color: C.gold,
              border: `1px solid ${C.borderGold}`,
            }}
          >
            <span className="w-1.5 h-1.5 inline-block" style={{ background: C.gold }} />
            TECHNICAL REFERENCE
          </div>

          <h1
            className="text-[clamp(28px,4vw,48px)] font-bold leading-[1.1] tracking-tight mb-6"
            style={{ color: C.text }}
          >
            High-Reliability<br />
            <span style={{ color: C.gold }}>Biometric Architecture</span>
          </h1>

          <p
            className="text-[15px] leading-[1.8] max-w-[680px] mb-10"
            style={{ color: C.textMuted }}
          >
            Fallen fuses deep neural network embeddings with Bayesian
            statistical inference to produce high-reliability similarity
            analysis. Four independent evidence channels — structural,
            geometric, textural, and physical mark correspondence — are
            combined through a rigorous Likelihood Ratio
            framework to yield highly reliable posterior probabilities.
          </p>

          {/* Pipeline stats bar */}
          <div
            className="flex gap-0"
            style={{ border: `1px solid ${C.border}` }}
          >
            {[
              { val: "4", label: "Evidence Tiers" },
              { val: "512-D", label: "Neural Embedding" },
              { val: "468", label: "Facial Landmarks" },
              { val: "LR", label: "Bayesian Fusion" },
              { val: "AES-256", label: "Envelope Encryption" },
            ].map((s, i) => (
              <div
                key={i}
                className="flex-1 px-4 py-3 text-center"
                style={{
                  background: C.bgDeep,
                  borderRight: i < 4 ? `1px solid ${C.border}` : "none",
                }}
              >
                <div className="text-[18px] font-bold" style={{ color: C.gold }}>
                  {s.val}
                </div>
                <div className="text-[9px] tracking-[0.15em] uppercase mt-0.5" style={{ color: C.textDim }}>
                  {s.label}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════
          PIPELINE HEADER
      ═══════════════════════════════════════════════════════════ */}
      <section style={{ background: C.bgDeep }}>
        <div className="max-w-[1200px] mx-auto px-6 py-10">
          <div className="flex items-center gap-4 mb-2">
            <div className="h-px flex-1" style={{ background: C.border }} />
            <h2
              className="text-[11px] tracking-[0.4em] uppercase font-bold"
              style={{ color: C.gold }}
            >
              Verification Pipeline
            </h2>
            <div className="h-px flex-1" style={{ background: C.border }} />
          </div>
          <p className="text-center text-[12px] max-w-[600px] mx-auto" style={{ color: C.textDim }}>
            Each tier operates as an independent evidence channel. Tiers 1–3 contribute weighted
            heuristic scores. Tier 4 applies Bayesian Likelihood Ratio fusion for highly reliable inference.
          </p>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════
          THE FOUR TIERS
      ═══════════════════════════════════════════════════════════ */}
      <section style={{ background: C.bgDeep }}>
        <div className="max-w-[1200px] mx-auto px-6 pb-16">
          <div className="flex flex-col gap-4">
            {TIERS.map((tier) => (
              <article
                key={tier.id}
                className="relative"
                style={{
                  background: C.bgPanel,
                  border: `1px solid ${tier.borderColor}`,
                }}
              >
                {/* Top accent bar */}
                <div className="h-[2px]" style={{ background: tier.color }} />

                <div className="p-6 md:p-8">
                  {/* Header row */}
                  <div className="flex items-start justify-between gap-6 mb-5">
                    <div className="flex items-start gap-4">
                      {/* Tier number */}
                      <div
                        className="flex items-center justify-center w-10 h-10 shrink-0 text-[10px] tracking-[0.2em] font-bold"
                        style={{
                          border: `1px solid ${tier.borderColor}`,
                          color: tier.color,
                          background: tier.bgAccent,
                        }}
                      >
                        {tier.id}
                      </div>
                      <div>
                        <div className="text-[9px] tracking-[0.3em] uppercase font-bold mb-0.5" style={{ color: tier.color }}>
                          {tier.label}
                        </div>
                        <h3 className="text-[20px] font-bold leading-tight" style={{ color: C.text }}>
                          {tier.title}
                        </h3>
                        <div className="text-[11px] mt-0.5" style={{ color: C.textMuted }}>
                          {tier.subtitle}
                        </div>
                      </div>
                    </div>

                    {/* Right-side metrics */}
                    <div className="hidden md:flex items-center gap-4 shrink-0">
                      <div className="text-right">
                        <div className="text-[22px] font-bold" style={{ color: tier.color }}>
                          {tier.metric}
                        </div>
                        <div className="text-[8px] tracking-[0.2em] uppercase" style={{ color: C.textDim }}>
                          {tier.metricLabel}
                        </div>
                      </div>
                      <div className="w-px h-10" style={{ background: C.border }} />
                      <div className="text-right">
                        <div className="text-[13px] font-bold" style={{ color: C.textMuted }}>
                          {tier.method}
                        </div>
                        <div className="text-[8px] tracking-[0.2em] uppercase" style={{ color: C.textDim }}>
                          {tier.threshold}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Body */}
                  <p className="text-[13px] leading-[1.8] mb-5 max-w-[800px]" style={{ color: C.textMuted }}>
                    {tier.body}
                  </p>

                  {/* Technical specifications */}
                  <div
                    className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-1 px-4 py-3"
                    style={{ background: C.bgDeep, border: `1px solid ${C.border}` }}
                  >
                    {tier.details.map((d, i) => (
                      <div key={i} className="flex items-start gap-2 py-1">
                        <span className="w-1 h-1 mt-[7px] shrink-0" style={{ background: tier.color }} />
                        <span className="text-[11px] leading-[1.6]" style={{ color: C.textDim }}>
                          {d}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </article>
            ))}
          </div>

          {/* Fusion equation callout */}
          <div
            className="mt-6 px-6 py-5 text-center"
            style={{
              background: C.bgPanel,
              border: `1px solid ${C.borderGold}`,
            }}
          >
            <div className="text-[9px] tracking-[0.3em] uppercase font-bold mb-3" style={{ color: C.gold }}>
              Bayesian Fusion Equation
            </div>
            <div className="text-[18px] md:text-[22px] font-bold tracking-wide" style={{ color: C.text }}>
              LR<sub className="text-[14px]">total</sub>{" "}
              <span style={{ color: C.textDim }}>=</span>{" "}
              LR<sub className="text-[14px]">arcface</sub>{" "}
              <span style={{ color: C.textDim }}>×</span>{" "}
              <span style={{ color: C.gold }}>
                Π
              </span>
              (LR<sub className="text-[14px]">mark_i</sub>)
              <span className="mx-4" style={{ color: C.textDim }}>→</span>
              P<sub className="text-[14px]">same</sub>{" "}
              <span style={{ color: C.textDim }}>=</span>{" "}
              <span style={{ color: C.gold }}>LR / (LR + 1)</span>
            </div>
            <p className="text-[11px] mt-3 max-w-[500px] mx-auto" style={{ color: C.textDim }}>
              The similarity probability given all observed evidence.
              Values approaching 1.0 indicate extremely strong similarity.
            </p>
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════
          SECURITY INFRASTRUCTURE
      ═══════════════════════════════════════════════════════════ */}
      <section style={{ background: C.bg }}>
        <div className="max-w-[1200px] mx-auto px-6 py-16">
          <div className="flex items-center gap-4 mb-10">
            <div className="h-px flex-1" style={{ background: C.border }} />
            <h2
              className="text-[11px] tracking-[0.4em] uppercase font-bold"
              style={{ color: C.gold }}
            >
              Security Infrastructure
            </h2>
            <div className="h-px flex-1" style={{ background: C.border }} />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {SECURITY_ITEMS.map((item, i) => (
              <article
                key={i}
                className="flex flex-col"
                style={{
                  background: C.bgPanel,
                  border: `1px solid ${C.border}`,
                }}
              >
                {/* Gold top accent */}
                <div className="h-[2px]" style={{ background: `linear-gradient(90deg, ${C.gold}, transparent)` }} />

                <div className="p-5 flex flex-col flex-1">
                  {/* Header */}
                  <div className="flex items-start justify-between mb-4">
                    <div style={{ color: C.gold }}>{item.icon}</div>
                    <div
                      className="px-2 py-1 text-[9px] tracking-[0.15em] font-bold"
                      style={{
                        color: C.gold,
                        background: C.goldDim,
                        border: `1px solid ${C.borderGold}`,
                      }}
                    >
                      {item.standard}
                    </div>
                  </div>

                  <h3 className="text-[15px] font-bold mb-2" style={{ color: C.text }}>
                    {item.title}
                  </h3>

                  <p className="text-[12px] leading-[1.7] mb-4 flex-1" style={{ color: C.textMuted }}>
                    {item.detail}
                  </p>

                  {/* Specs list */}
                  <div
                    className="px-3 py-2.5"
                    style={{ background: C.bgDeep, border: `1px solid ${C.border}` }}
                  >
                    {item.specs.map((spec, j) => (
                      <div key={j} className="flex items-center gap-2 py-0.5">
                        <span className="w-1 h-1 shrink-0" style={{ background: C.gold }} />
                        <span className="text-[10px]" style={{ color: C.textDim }}>{spec}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </article>
            ))}
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════
          FOOTER
      ═══════════════════════════════════════════════════════════ */}
      <footer
        className="py-8 px-6 text-center"
        style={{
          background: C.bgDeep,
          borderTop: `1px solid ${C.border}`,
        }}
      >
        <div className="max-w-[1200px] mx-auto">
          <div className="flex items-center justify-center gap-3 mb-3">
            <div className="h-px w-12" style={{ background: C.borderGold }} />
            <span className="text-[10px] tracking-[0.3em] uppercase font-bold" style={{ color: C.gold }}>
              Fallen
            </span>
            <div className="h-px w-12" style={{ background: C.borderGold }} />
          </div>
          <p className="text-[10px] leading-[1.8]" style={{ color: C.textDim }}>
            Similarity Analysis · Verification Engine · v3.0
          </p>
          <p className="text-[9px] mt-2" style={{ color: C.textDim }}>
            This document constitutes a technical reference for qualified auditors and institutional evaluators.
          </p>
        </div>
      </footer>
    </main>
  );
}
