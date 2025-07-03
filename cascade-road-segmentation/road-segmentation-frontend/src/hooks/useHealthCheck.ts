'use client'

import { useState, useEffect, useCallback } from 'react'
import { apiClient } from '@/services/api/client'
import { HEALTH_CHECK_INTERVAL } from '@/constants/api'
import type { HealthCheck } from '@/services/types/api'

export function useHealthCheck() {
  const [health, setHealth] = useState<HealthCheck | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [lastChecked, setLastChecked] = useState<Date | null>(null)

  const checkHealth = useCallback(async () => {
    try {
      const healthData = await apiClient.healthCheck()
      setHealth(healthData)
      setLastChecked(new Date())
    } catch (error) {
      console.error('Health check failed:', error)
      setHealth(null)
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    checkHealth()
    
    const interval = setInterval(checkHealth, HEALTH_CHECK_INTERVAL)
    
    return () => clearInterval(interval)
  }, [checkHealth])

  const isHealthy = health?.status === 'healthy'
  const isDegraded = health?.status === 'degraded'
  const isUnhealthy = !health || health.status === 'unhealthy'

  return {
    health,
    isLoading,
    lastChecked,
    isHealthy,
    isDegraded,
    isUnhealthy,
    checkHealth
  }
}