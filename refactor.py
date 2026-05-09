import re

with open('C:\\Users\\jimbo\\OneDrive\\Documents\\facial\\frontend\\app\\page.tsx', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Replace getResultLabel
old_get_result = """  /** Get the result classification label */
  function getResultLabel(r: VerificationResult): string {
    if (r.failed_provenance_veto) {
      return 'Synthetic Provenance Veto Triggered';
    }
    if (r.veto_triggered && !r.veto_override_applied) {
      return 'Inconclusive — Limited by Face-Model Threshold';
    }
    if (r.fused_identity_score >= 90) return 'Strongly Supports Common Source';
    if (r.fused_identity_score >= 75) return 'Supports Common Source';
    return 'Inconclusive — Insufficient Evidence';
  }"""

new_get_result = """  /** Get the result classification label */
  function getResultLabel(r: VerificationResult): string {
    if (r.failed_provenance_veto) {
      return 'Synthetic Provenance Veto Triggered';
    }
    if (r.veto_triggered && r.veto_override_applied) {
      return 'Conflicting Evidence — Human Review Needed';
    }
    if (r.veto_triggered && !r.veto_override_applied) {
      return 'Face Similarity Was Too Low to Confirm Same Person';
    }
    if (r.fused_identity_score >= 90) return 'Strongly Supports Same Person';
    if (r.fused_identity_score >= 75) return 'Supports Same Person';
    return 'Not Enough Evidence to Confirm Same Person';
  }"""

content = content.replace(old_get_result, new_get_result)

# Extract SHARED MARK EVIDENCE and HOW WE ANALYZED THIS to move them to Technical Details
start_shared = content.find("{/* ═══ SHARED MARK EVIDENCE — First-Class Panel ═══ */}")
end_shared = content.find("{/* ═══ HOW WE SCORED THIS — Breakdown ═══ */}")
start_how = content.find("{/* ═══ HOW WE SCORED THIS — Breakdown ═══ */}")
end_how = content.find("{/* ═══ ACTIONS — Report + Technical Details ═══ */}")

if start_shared != -1 and end_shared != -1 and end_how != -1:
    shared_mark_panel = content[start_shared:end_shared]
    how_panel = content[start_how:end_how]
    
    # We will insert them into the Technical Details
    tech_details_insert_point = content.find("{/* Block 4: Bayesian Evidence + Decision Policy */}")
    
    # Let's adjust the styling of the moved panels slightly so they fit in the modal
    shared_mark_panel = shared_mark_panel.replace('border border-[#1f1f1f] bg-[#0d0d0e]', 'border border-[#2a1a1a] bg-[#020101]')
    how_panel = how_panel.replace('border border-[#1f1f1f] bg-[#0d0d0e]', 'border border-[#2a1a1a] bg-[#020101]')
    
    new_tech_details = shared_mark_panel + "\n                      " + how_panel + "\n                      {/* Block 4: Bayesian Evidence + Decision Policy */}"
    content = content.replace("{/* Block 4: Bayesian Evidence + Decision Policy */}", new_tech_details)

# Replace the VERDICT and the old panels in the Right Panel
start_verdict = content.find("{/* ═══ VERDICT — Primary User-Facing Result ═══ */}")
end_actions = content.find("{/* ═══ ACTIONS — Report + Technical Details ═══ */}")

new_right_panel = """{/* ═══ VERDICT — Primary User-Facing Result ═══ */}
              <div className="flex flex-col gap-3 p-3 bg-[#0d0d0e] border-2 border-[#1f1f1f] rounded-lg">
                <div className="text-[10px] font-bold text-gray-400 tracking-widest uppercase border-b border-[#1f1f1f] pb-2">Final Result</div>
                <div className="flex flex-col gap-2">
                  <div className={`text-xl font-bold ${
                    results.veto_triggered && results.veto_override_applied ? 'text-amber-400' :
                    results.veto_triggered ? 'text-red-400' :
                    results.fused_identity_score >= 75 ? 'text-emerald-400' : 'text-gray-400'
                  }`}>
                    {getResultLabel(results)}
                  </div>
                  <p className="text-[10px] text-gray-500 leading-relaxed">
                    {results.veto_triggered && results.veto_override_applied 
                      ? "Facial mark evidence strongly supports that these may be the same person. However, the face similarity score is too low to confirm a match automatically. Human review is recommended."
                      : results.veto_triggered 
                      ? "The face similarity score is too low to confirm a match automatically. The facial marks detected do not provide enough evidence to override this safety rule."
                      : results.fused_identity_score >= 75
                      ? "Both the face similarity score and the facial marks strongly suggest these are the same person."
                      : "There is not enough evidence to confirm these are the same person."}
                  </p>
                </div>
              </div>

              {/* ═══ EVIDENCE SUMMARY ═══ */}
              <div className="flex flex-col gap-3 p-3 bg-[#0d0d0e] border-2 border-[#1f1f1f] rounded-lg mb-4">
                <div className="text-[10px] font-bold text-gray-400 tracking-widest uppercase border-b border-[#1f1f1f] pb-2">Evidence Summary</div>
                
                {/* Face Similarity Channel */}
                <div className="flex justify-between items-center border-b border-[#1a1a1a] pb-2">
                  <div className="flex flex-col">
                    <span className="text-[10px] text-gray-300 font-bold tracking-wider">Face Similarity</span>
                    <span className="text-[8px] text-gray-500">Overall face shape and proportions</span>
                  </div>
                  <span className={`text-[10px] font-bold px-2 py-1 rounded ${results.structural_score > 80 ? 'bg-emerald-900/30 text-emerald-400' : results.structural_score > 60 ? 'bg-amber-900/30 text-amber-400' : 'bg-red-900/30 text-red-400'}`}>
                    {results.structural_score > 80 ? 'STRONG' : results.structural_score > 60 ? 'MODERATE' : 'WEAK'}
                  </span>
                </div>

                {/* Facial Mark Evidence Channel */}
                <div className="flex justify-between items-center border-b border-[#1a1a1a] pb-2">
                  <div className="flex flex-col">
                    <span className="text-[10px] text-gray-300 font-bold tracking-wider">Facial Mark Evidence</span>
                    <span className="text-[8px] text-gray-500">Scars, moles, and blemishes</span>
                  </div>
                  <span className={`text-[10px] font-bold px-2 py-1 rounded ${
                    (results.mark_diagnostics?.lr_marks ?? 1) > 100 ? 'bg-emerald-900/30 text-emerald-400' : 
                    (results.mark_diagnostics?.lr_marks ?? 1) > 1 ? 'bg-amber-900/30 text-amber-400' : 
                    'bg-gray-800 text-gray-400'
                  }`}>
                    {(results.mark_diagnostics?.lr_marks ?? 1) > 100 ? 'STRONG' : 
                     (results.mark_diagnostics?.lr_marks ?? 1) > 1 ? 'MODERATE' : 
                     (results.mark_diagnostics?.raw_probe_marks_count === 0 && results.mark_diagnostics?.raw_gallery_marks_count === 0) ? 'NONE DETECTED' : 'WEAK / NEUTRAL'}
                  </span>
                </div>

                {/* Final Safety Rule */}
                <div className="flex justify-between items-center">
                  <div className="flex flex-col">
                    <span className="text-[10px] text-gray-300 font-bold tracking-wider">Final Safety Rule</span>
                    <span className="text-[8px] text-gray-500">Prevents false positive matches</span>
                  </div>
                  <span className={`text-[10px] font-bold px-2 py-1 rounded ${
                    results.veto_triggered && results.veto_override_applied ? 'bg-amber-900/30 text-amber-400' :
                    results.veto_triggered ? 'bg-red-900/30 text-red-400' :
                    'bg-emerald-900/30 text-emerald-400'
                  }`}>
                    {results.veto_triggered && results.veto_override_applied ? 'HUMAN REVIEW NEEDED' :
                     results.veto_triggered ? 'BLOCKED AUTOMATIC CONFIRMATION' :
                     'PASSED'}
                  </span>
                </div>
              </div>

              """

if start_verdict != -1 and end_actions != -1:
    content = content[:start_verdict] + new_right_panel + content[end_actions:]
else:
    print("Could not find start_verdict or end_actions")

with open('C:\\Users\\jimbo\\OneDrive\\Documents\\facial\\frontend\\app\\page.tsx', 'w', encoding='utf-8') as f:
    f.write(content)

print("Done")
