export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? 'https://road.sgta.lat'

export const API_ENDPOINTS = {
  HEALTH: '/api/v1/health',
  MODEL_INFO: '/api/v1/model/info',
  PREDICT_SINGLE: '/api/v1/predict/single',
  PREDICT_BATCH: '/api/v1/predict/batch',
  RESULTS: '/api/v1/results',
  BROWSE: '/api/v1/browse'
} as const

export const HEALTH_CHECK_INTERVAL = 5000 // 5 seconds