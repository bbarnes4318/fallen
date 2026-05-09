 = Get-Content -Path 'C:\Users\jimbo\OneDrive\Documents\facial\backend\main.py' -Raw
 = @('fetch_image_from_url', 'apply_clahe', 'align_face_crop', 'extract_ensemble_embeddings', 'compute_ensemble_similarity', 'score_to_lr_ensemble', 'evaluate_mark_veto_override', '_run_mark_evidence_pipeline', 'estimate_age', 'cross_spectral_normalize', 'compute_image_hash', 'calculate_cosine_similarity', 'finite_or_none')

foreach ( in ) {
    if ( -match "(?m)^def \b") {
        Write-Output "Found: "
    } else {
        Write-Output "MISSING: "
    }
}
