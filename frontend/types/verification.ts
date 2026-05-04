export interface AuditLog {
  raw_cosine_score: number;
  raw_arcface_score?: number;
  raw_secondary_score?: number;
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
  liveness_check?: { method: string; spoof_probability: string; status: string; laplacian_variance?: number };
  crypto_envelope?: { standard: string; decryption_time: string };
  calibration_benchmark?: string;
  calibration_pairs?: number;
  lr_arcface?: number | null;
  lr_marks?: number | null;
  lr_total?: number | null;
  posterior_probability?: number | null;
  mark_lrs?: number[] | null;
  bayesian_fused_score?: number | null;
  // Chain of Custody — Source file hashes
  probe_source_file_hash?: string;
  gallery_source_file_hash?: string;
  // Chain of Custody — Decoded & aligned image hashes
  probe_decoded_image_hash?: string;
  gallery_decoded_image_hash?: string;
  probe_aligned_crop_hash_pre_clahe?: string;
  gallery_aligned_crop_hash_pre_clahe?: string;
  probe_aligned_crop_hash_post_clahe?: string;
  gallery_aligned_crop_hash_post_clahe?: string;
  // Image dimensions at each stage
  probe_original_dimensions?: string;
  gallery_original_dimensions?: string;
  probe_decoded_dimensions?: string;
  gallery_decoded_dimensions?: string;
  probe_aligned_dimensions?: string;
  gallery_aligned_dimensions?: string;
  preprocessing_steps?: string[];
  // Model Provenance
  code_commit_hash?: string;
  docker_image_digest?: string;
  arcface_model_name?: string;
  arcface_weight_hash?: string;
  secondary_model_weight_hash?: string;
  mediapipe_version?: string;
  opencv_version?: string;
  deepface_version?: string;
  calibration_file_hash?: string;
  calibration_pair_count?: number;
  receipt_url?: string | null;
  synthetic_anomaly_score?: number | null;
  failed_provenance_veto?: boolean | null;
}

export type RawPoint = 
  | [number, number, number?] 
  | { x: number; y: number; area?: number; lr?: number }
  | { centroid: [number, number]; area?: number; lr?: number };

export interface Correspondence {
  gallery_idx?: number;
  probe_idx?: number;
  gallery_pt?: RawPoint;
  probe_pt?: RawPoint;
  lr: number;
  mark_type?: string;
  face_region?: string;
  gallery_centroid?: [number, number];
  probe_centroid?: [number, number];
  match_quality?: number;
  position_distance?: number;
  area_ratio?: number;
  type_match?: boolean;
  region_match?: boolean;
}

export type MarkType =
  | "dark_mole"
  | "dark_blob"
  | "dark_spot"
  | "light_scar"
  | "light_blob"
  | "light_spot"
  | "linear_scar"
  | "texture_cluster"
  | "blemish"
  | "unknown_mark"
  | "unknown";

/** v2.1 detector status codes */
export type DetectorStatus =
  | "OK"
  | "LOW_CONFIDENCE_CANDIDATES"
  | "NO_CANDIDATES"
  | "FACE_NOT_DETECTED"
  | "LANDMARK_FALLBACK_ROI"
  | "DETECTOR_ERROR"
  | "UNKNOWN";

/** v2.1 detection channel identifiers */
export type MarkChannel =
  | "dark_lesion"
  | "bright_scar"
  | "linear_scar_v2"
  | "texture_anomaly"
  | "dark"
  | "light"
  | "linear_scar"
  | "texture_cluster";

export interface MarkDescriptor {
  index?: number;
  centroid?: [number, number];
  x?: number;
  y?: number;
  area?: number;
  intensity?: number;
  circularity?: number;
  aspect_ratio?: number;
  orientation?: number;
  bbox?: [number, number, number, number];
  contour_area?: number;
  source_side?: "probe" | "gallery";
  mark_type?: MarkType;
  nearest_landmark_index?: number;
  face_region?: string;
  lr?: number;
  rejection_reason?: string;
  // v2.1 fields
  channel?: MarkChannel;
  salience_score?: number;
  confidence?: number;
  contrast_score?: number;
  region_label?: string;
  low_confidence?: boolean;
  fallback_generated?: boolean;
  centroid_px?: [number, number];
  equivalent_radius?: number;
  eccentricity?: number;
  [key: string]: unknown;
}

export interface MarkDebugCorrespondence {
  gallery_idx?: number;
  probe_idx?: number;
  gallery_pt?: RawPoint;
  probe_pt?: RawPoint;
  lr?: number;
  match_cost?: number;
  position_distance?: number;
  area_ratio?: number;
  type_match?: boolean;
  region_match?: boolean;
  rejection_reason?: string;
  [key: string]: unknown;
}

export interface MarkDebugPayload {
  probe_marks_count?: number;
  gallery_marks_count?: number;
  correspondences_count?: number;
  probe_marks_first_20?: MarkDescriptor[];
  gallery_marks_first_20?: MarkDescriptor[];
  correspondences_first_20?: MarkDebugCorrespondence[];
  correspondences?: MarkDebugCorrespondence[];
  unmatched_probe_indices?: number[];
  unmatched_gallery_indices?: number[];
  probe_unmatched_indices?: number[];
  gallery_unmatched_indices?: number[];
  rejected_candidates?: MarkDebugCorrespondence[];
  rejected_probe_marks?: MarkDescriptor[];
  rejected_gallery_marks?: MarkDescriptor[];
  detector_version?: string;
  matcher_version?: string;
  version?: string;
  [key: string]: unknown;
}

/** Typed payload for per-face data returned by the backend */
export interface FaceDataPayload {
  marks?: RawPoint[];
  occluded_regions?: string[];
  age?: number;
  [key: string]: unknown;
}

/** v2.1 per-face detector trace telemetry */
export interface MarkDetectorTrace {
  initial_candidates?: number;
  after_skin_mask?: number;
  after_area_filter?: number;
  after_shape_filter?: number;
  after_region_exclusion?: number;
  after_contrast_filter?: number;
  dark_lesion_initial_candidates?: number;
  bright_scar_initial_candidates?: number;
  linear_scar_initial_candidates?: number;
  texture_anomaly_initial_candidates?: number;
  strict_final_valid_marks?: number;
  fallback_used?: boolean;
  fallback_candidates?: number;
  dedup_removed?: number;
  final_valid_marks?: number;
  detector_status?: DetectorStatus;
  fallback_lr_cap?: number;
  fallback_penalty_applied?: boolean;
}

/** v2.1 debug overlay images (DEBUG_FORENSIC only) */
export interface MarkDebugOverlays {
  face_roi_mask_b64?: string;
  dark_candidate_mask_b64?: string;
  bright_candidate_mask_b64?: string;
  linear_candidate_mask_b64?: string;
  texture_candidate_mask_b64?: string;
  rejected_overlay_b64?: string;
  final_marks_overlay_b64?: string;
}

/** Lightweight always-on mark diagnostics (production-safe) */
export interface MarkDiagnostics {
  raw_probe_marks_count: number;
  raw_gallery_marks_count: number;
  accepted_correspondences_count: number;
  rejected_candidates_count: number;
  detector_status: DetectorStatus | string;
  probe_detector_status?: DetectorStatus | string;
  gallery_detector_status?: DetectorStatus | string;
  matcher_status: string;   // "OK" | "NO_MATCHES" | "INSUFFICIENT_INPUT"
  lr_marks: number | null;
  mark_match_status: string | null;
  rejection_summary: string | null;
  // v2.1 trace
  mark_detector_trace?: {
    probe?: MarkDetectorTrace;
    gallery?: MarkDetectorTrace;
  };
}

/** Bayesian scoring trace — returned only when DEBUG_FORENSIC=true */
export interface ScoringTrace {
  calibration_status?: string;
  calibration_source?: string;
  calibration_benchmark?: string;
  tier4_calibration_status?: string;
  lr_ensemble_raw?: number | null;
  lr_marks_raw?: number | null;
  lr_total_raw?: number | null;
  lr_ensemble_display?: string;
  lr_marks_display?: string;
  lr_total_display?: string;
  posterior_raw?: number | null;
  fused_score_pre_veto?: number | null;
  fused_score_post_veto?: number | null;
  veto_triggered?: boolean;
  veto_reason?: string | null;
  veto_override_applied?: boolean;
  veto_override_reason?: string | null;
  mark_override_eligible?: boolean;
  temporal_delta_years?: number | null;
  ensemble_thresholds_key?: string;
}

export type MarkMatchStatus =
  | "EXACT_SELF_MATCH"
  | "MATCHED"
  | "INSUFFICIENT_MARKS"
  | "NO_MATCHES"
  | "FACE_NOT_DETECTED"
  | "DETECTOR_UNAVAILABLE"
  | "UNKNOWN";

export interface VerificationResult {
  structural_score: number;
  soft_biometrics_score: number;
  micro_topology_score: number;
  geometry_status?: string | null;
  geometric_ratio_distance?: number | null;
  mark_correspondence_score?: number | null;
  marks_detected_gallery?: number;
  marks_detected_probe?: number;
  marks_matched?: number;
  fused_identity_score: number;
  bayesian_fused_score?: number | null;
  veto_reason?: string | null;
  veto_triggered: boolean;
  failed_provenance_veto?: boolean;
  synthetic_anomaly_score?: number;
  veto_override_applied?: boolean;
  veto_override_reason?: string | null;
  scoring_trace?: ScoringTrace | null;
  calibration_status?: string | null;
  conclusion: string;
  gallery_heatmap_b64: string;
  probe_heatmap_b64: string;
  gallery_aligned_b64: string;
  probe_aligned_b64: string;
  receipt_url?: string | null;
  scar_delta_b64: string;  // Backwards compat — prefer edge_delta_b64
  edge_delta_b64?: string | null;  // Forward-compatible field name
  gallery_wireframe_b64: string;
  probe_wireframe_b64: string;
  probe_mark_debug_b64?: string | null;
  gallery_mark_debug_b64?: string | null;
  mark_debug?: MarkDebugPayload | null;  // Full debug payload (DEBUG_FORENSIC only)
  mark_diagnostics?: MarkDiagnostics | null;  // Lightweight always-on diagnostics
  correspondences?: Correspondence[];
  audit_log?: AuditLog;
  probe_data?: FaceDataPayload;
  gallery_data?: FaceDataPayload;
  occluded_regions?: string[];
  occlusion_percentage?: number;
  effective_geometric_ratios_used?: number;
  raw_probe_marks?: RawPoint[];
  raw_gallery_marks?: RawPoint[];
  // Mark evidence metadata (v2.0)
  mark_match_status?: MarkMatchStatus | null;
  lr_marks?: number | null;
  mark_lrs?: number[] | null;
  mark_match_overlay_b64?: string | null;
  mark_detector_version?: string | null;
  mark_matcher_version?: string | null;
  exact_image_match?: boolean;
  // Face-model evidence (explicit decomposition)
  raw_arcface_similarity?: number | null;
  raw_secondary_similarity?: number | null;
  fused_face_model_similarity?: number | null;
  lr_face_model?: number | null;
  // v2.1 debug overlays (DEBUG_FORENSIC only)
  mark_debug_overlays?: MarkDebugOverlays | null;
}

export interface ForensicPoint {
  x: number;
  y: number;
  area: number;
  lr?: number;
  isMatched?: boolean;
}
