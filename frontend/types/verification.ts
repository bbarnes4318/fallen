export interface AuditLog {
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
  liveness_check?: { method: string; spoof_probability: string; status: string; laplacian_variance?: number };
  crypto_envelope?: { standard: string; decryption_time: string };
  calibration_benchmark?: string;
  calibration_pairs?: number;
  lr_arcface?: number | null;
  lr_marks?: number | null;
  lr_total?: number | null;
  posterior_probability?: number | null;
  mark_lrs?: number[] | null;
}

export interface Correspondence {
  gallery_pt: any;
  probe_pt: any;
  lr: number;
}

export interface VerificationResult {
  structural_score: number;
  soft_biometrics_score: number;
  micro_topology_score: number;
  mark_correspondence_score?: number | null;
  marks_detected_gallery?: number;
  marks_detected_probe?: number;
  marks_matched?: number;
  fused_identity_score: number;
  veto_triggered: boolean;
  failed_provenance_veto?: boolean;
  synthetic_anomaly_score?: number;
  conclusion: string;
  gallery_heatmap_b64: string;
  probe_heatmap_b64: string;
  gallery_aligned_b64: string;
  probe_aligned_b64: string;
  scar_delta_b64: string;
  gallery_wireframe_b64: string;
  probe_wireframe_b64: string;
  correspondences?: Correspondence[];
  audit_log?: AuditLog;
  probe_data?: any;
  gallery_data?: any;
  occluded_regions?: string[];
  occlusion_percentage?: number;
  effective_geometric_ratios_used?: number;
  raw_probe_marks?: any[];
  raw_gallery_marks?: any[];
}

export interface ForensicPoint {
  x: number;
  y: number;
  area: number;
  lr?: number;
  isMatched?: boolean;
}
