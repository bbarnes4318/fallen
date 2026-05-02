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
}

export type MarkType =
  | "dark_blob"
  | "light_blob"
  | "linear_scar"
  | "texture_cluster"
  | "unknown";

export interface MarkDescriptor {
  index?: number;
  centroid?: [number, number];
  x?: number;
  y?: number;
  area?: number;
  intensity?: number;
  circularity?: number;
  bbox?: [number, number, number, number];
  contour_area?: number;
  source_side?: "probe" | "gallery";
  mark_type?: MarkType;
  nearest_landmark_index?: number;
  face_region?: string;
  lr?: number;
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
  unmatched_probe_indices?: number[];
  unmatched_gallery_indices?: number[];
  rejected_candidates?: MarkDebugCorrespondence[];
  detector_version?: string;
  matcher_version?: string;
  [key: string]: unknown;
}

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
  probe_mark_debug_b64?: string | null;
  gallery_mark_debug_b64?: string | null;
  mark_debug?: MarkDebugPayload | null;
  correspondences?: Correspondence[];
  audit_log?: AuditLog;
  probe_data?: Record<string, unknown>;
  gallery_data?: Record<string, unknown>;
  occluded_regions?: string[];
  occlusion_percentage?: number;
  effective_geometric_ratios_used?: number;
  raw_probe_marks?: RawPoint[];
  raw_gallery_marks?: RawPoint[];
}

export interface ForensicPoint {
  x: number;
  y: number;
  area: number;
  lr?: number;
  isMatched?: boolean;
}
