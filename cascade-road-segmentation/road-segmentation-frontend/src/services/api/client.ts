import { API_BASE_URL, API_ENDPOINTS } from '@/constants/api'
import type { 
  HealthCheck, 
  ModelInfo, 
  InferenceResult, 
  BatchInferenceResult, 
  BrowseResults,
  ApiResponse 
} from '@/services/types/api'

class ApiClient {
  private readonly baseUrl: string

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl
  }

  private async request<T>(
    endpoint: string, 
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
        ...options,
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      return data
    } catch (error) {
      console.error('API request failed:', error)
      return {
        success: false,
        message: error instanceof Error ? error.message : 'Unknown error occurred',
        error: 'network_error'
      }
    }
  }

  async healthCheck(): Promise<HealthCheck | null> {
    try {
      const response = await fetch(`${this.baseUrl}${API_ENDPOINTS.HEALTH}`)
      if (!response.ok) return null
      return await response.json()
    } catch {
      return null
    }
  }

  async getModelInfo(): Promise<ApiResponse<ModelInfo>> {
    try {
      const response = await fetch(`${this.baseUrl}${API_ENDPOINTS.MODEL_INFO}`)
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      
      // Check if response is already in ApiResponse format or raw model info
      if (data.success !== undefined) {
        return data
      } else {
        // Raw model info, wrap it in ApiResponse format
        return {
          success: true,
          message: 'Model info retrieved successfully',
          data: data
        }
      }
    } catch (error) {
      console.error('Failed to get model info:', error)
      return {
        success: false,
        message: error instanceof Error ? error.message : 'Failed to get model info',
        error: 'network_error'
      }
    }
  }

  async browseResults(): Promise<ApiResponse<BrowseResults>> {
    return this.request<BrowseResults>(API_ENDPOINTS.BROWSE)
  }

  async predictSingle(
    file: File,
    options: {
      saveOutputs?: boolean
      overlayAlpha?: number
      confidenceThreshold?: number
    } = {}
  ): Promise<ApiResponse<InferenceResult>> {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('save_outputs', String(options.saveOutputs ?? true))
    formData.append('overlay_alpha', String(options.overlayAlpha ?? 0.6))
    formData.append('confidence_threshold', String(options.confidenceThreshold ?? 0.6))

    return this.request<InferenceResult>(API_ENDPOINTS.PREDICT_SINGLE, {
      method: 'POST',
      body: formData,
      headers: {} // Remove Content-Type to let browser set it with boundary
    })
  }

  async predictBatch(
    files: File[],
    options: {
      saveOutputs?: boolean
      overlayAlpha?: number
      confidenceThreshold?: number
    } = {}
  ): Promise<ApiResponse<BatchInferenceResult>> {
    const formData = new FormData()
    files.forEach(file => formData.append('files', file))
    formData.append('save_outputs', String(options.saveOutputs ?? true))
    formData.append('overlay_alpha', String(options.overlayAlpha ?? 0.6))
    formData.append('confidence_threshold', String(options.confidenceThreshold ?? 0.6))

    return this.request<BatchInferenceResult>(API_ENDPOINTS.PREDICT_BATCH, {
      method: 'POST',
      body: formData,
      headers: {}
    })
  }

  getResultUrl(filename: string): string {
    return `${this.baseUrl}${API_ENDPOINTS.RESULTS}/${filename}`
  }

  getOverlayUrl(resultId: string): string {
    return `${this.baseUrl}${API_ENDPOINTS.RESULTS}/${resultId}/overlay`
  }

  async downloadResult(filename: string): Promise<Blob | null> {
    try {
      const response = await fetch(`${this.baseUrl}${API_ENDPOINTS.RESULTS}/${filename}`)
      if (!response.ok) return null
      return await response.blob()
    } catch (error) {
      console.error('Download failed:', error)
      return null
    }
  }
}

export const apiClient = new ApiClient()
export default apiClient