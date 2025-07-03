export interface HealthCheck {
  status: 'healthy' | 'degraded' | 'unhealthy'
  timestamp: string
  version: string
  model_status: 'initialized' | 'loading' | 'error'
  uptime_seconds: number
}

export interface ModelInfo {
  status: 'initialized' | 'loading' | 'error'
  architecture: string
  device: string
  classes: string[]
  num_classes: number
  confidence_threshold: number
  road_model_path: string
  defect_model_path: string
}

export interface DefectCounts {
  alligator_cracking: number
  longitudinal_cracking: number
  transverse_cracking: number
  pothole: number
}

export interface InferenceResult {
  image_name: string
  image_shape: [number, number, number]
  road_coverage: number
  total_defect_pixels: number
  defect_counts: DefectCounts
  mean_confidence: number
  processing_status: 'success' | 'failed'
  error_message?: string
  overlay_path?: string
  defect_mask_path?: string
  road_mask_path?: string
}

export interface BatchInferenceResult {
  total_images: number
  successful: number
  failed: number
  results: InferenceResult[]
}

export interface BrowseResultEntry {
  directory: string
  full_path: string
  files: {
    overlays: string[]
    defect_masks: string[]
    road_masks: string[]
    other_files: string[]
  }
  file_count: number
  download_urls: {
    overlays: string[]
    defect_masks: string[]
    road_masks: string[]
  }
}

export interface BrowseResults {
  results: BrowseResultEntry[]
  output_directory: string
  total_directories: number
}

export interface ApiResponse<T = unknown> {
  success: boolean
  message: string
  data?: T
  error?: string
  timestamp?: string
}