import sys
import re

with open('backend/main.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace Veto String
text = text.replace('conclusion="VETO: Synthetic Media Detected",', 'conclusion="Synthetic Provenance Veto Triggered",')
text = text.replace('"conclusion": "VETO: Synthetic Media Detected",', '"conclusion": "Synthetic Provenance Veto Triggered",')

# Replace fuse conclusions
# For these, I need to use regex because the characters might be unicode
text = re.sub(r'conclusion = "Strongest Support for Common Source \(Bayesian Posterior [^)]+\)"', 'conclusion = "Strongly Supports Common Source"', text)
text = re.sub(r'conclusion = "Moderate Support for Common Source \(Bayesian Posterior [^)]+\)"', 'conclusion = "Supports Common Source"', text)
text = text.replace('conclusion = "Inconclusive: Insufficient Bayesian Evidence for Common Source"', 'conclusion = "Inconclusive — Insufficient Evidence"')

# For Face Model Veto
text = re.sub(r'"Face Model Veto: ArcFace embedding similarity below operating threshold\. "\s*"This veto applies to the face-model channel only and does not constitute "\s*"a validated full biometric exclusion\."', '"Inconclusive — Limited by Face-Model Threshold"', text)

# Replace 1vn conclusions
text = re.sub(r'conclusion = f"Strongest Support for Common Source [^\\]+ Nearest vault candidate: \{best_user_id\} \(Posterior: \{fused_score:\.1f\}%\)"', 'conclusion = f"Strongly Supports Common Source — Nearest vault candidate: {best_user_id}"', text)
text = re.sub(r'conclusion = f"Moderate Support for Common Source [^\\]+ Nearest vault candidate: \{best_user_id\} \(Posterior: \{fused_score:\.1f\}%\)"', 'conclusion = f"Supports Common Source — Nearest vault candidate: {best_user_id}"', text)
text = re.sub(r'conclusion = f"Inconclusive [^\\]+ Nearest vault candidate: \{best_user_id\} \(Posterior: \{fused_score:\.1f\}%\)"', 'conclusion = f"Inconclusive — Insufficient Evidence (Nearest vault candidate: {best_user_id})"', text)

# Also replace the Mark Override conclusion
text = re.sub(r'"Mark Override Applied: ArcFace face-model veto overridden by independent "\s*"mark correspondence evidence\. ArcFace channel did not pass\."', '"Supports Common Source — Face-Model Veto Overridden by Mark Correspondence"', text)
text = re.sub(r'f"Mark Override Applied: ArcFace face-model veto overridden by independent "\s*f"mark correspondence evidence\. ArcFace channel did not pass\. \(\{best_user_id\}\)"', 'f"Supports Common Source — Face-Model Veto Overridden by Mark Correspondence ({best_user_id})"', text)


# Add missing fields in VerificationEvent
rep1 = '''            effective_geometric_ratios_used=effective_ratios,
            receipt_url=receipt_url,
            lr_arcface=finite_or_none(lr_ensemble),'''
new1 = '''            effective_geometric_ratios_used=effective_ratios,
            receipt_url=receipt_url,
            synthetic_anomaly_score=max_anomaly,
            failed_provenance_veto=False,
            lr_arcface=finite_or_none(lr_ensemble),'''
text = text.replace(rep1, new1)

with open('backend/main.py', 'w', encoding='utf-8') as f:
    f.write(text)
print('Done!')
