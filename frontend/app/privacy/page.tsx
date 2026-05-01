export default function PrivacyPolicy() {
  return (
    <div className="min-h-screen bg-black text-[#E0E0E0] p-8 md:p-16 lg:p-24 font-mono selection:bg-[#FF3366] selection:text-white">
      <div className="max-w-4xl mx-auto space-y-12">
        <header className="space-y-4">
          <h1 className="text-4xl md:text-5xl font-bold tracking-tighter text-white">PRIVACY & USAGE POLICY</h1>
          <p className="text-[#FF3366] text-sm md:text-base font-semibold tracking-widest uppercase">
            FALLEN EXPERIMENTAL PLATFORM
          </p>
        </header>

        <section className="space-y-6 text-sm md:text-base leading-relaxed">
          <h2 className="text-2xl font-bold text-white border-b border-[#333333] pb-2">1. EXPERIMENTAL NATURE</h2>
          <p>
            The Fallen biometric analysis platform is provided strictly for experimental, research, and entertainment purposes. It does not provide legally binding, forensic-grade, or &quot;Daubert compliant&quot; identity verification. 
          </p>
          <p>
            The similarity metrics presented by this system are statistically generated estimates and must never be used as the sole basis for legal, financial, or employment decisions.
          </p>
        </section>

        <section className="space-y-6 text-sm md:text-base leading-relaxed">
          <h2 className="text-2xl font-bold text-white border-b border-[#333333] pb-2">2. DATA RETENTION & SECURITY</h2>
          <p>
            Uploaded images are processed in a secure environment. Image processing occurs via ephemeral cloud instances. Resulting biometric embeddings are encrypted using industry-standard KMS cryptography before being written to the vault index.
          </p>
          <p>
            You retain the right to request the deletion of your biometric footprint from the Fallen ledger at any time.
          </p>
        </section>

        <section className="space-y-6 text-sm md:text-base leading-relaxed">
          <h2 className="text-2xl font-bold text-white border-b border-[#333333] pb-2">3. LIMITATION OF LIABILITY</h2>
          <p>
            The operators of this platform assume no liability for the misinterpretation, misuse, or overreliance on the similarity scores or &quot;Delta&quot; visualizations provided. By utilizing this system, you explicitly acknowledge that biometric similarity does not definitively prove identity.
          </p>
        </section>

        <footer className="pt-12 border-t border-[#333333] text-xs text-[#666666]">
          <p>Last Updated: {new Date().toLocaleDateString()}</p>
          <p className="mt-2">For inquiries, contact legal@scargods.com</p>
        </footer>
      </div>
    </div>
  )
}
