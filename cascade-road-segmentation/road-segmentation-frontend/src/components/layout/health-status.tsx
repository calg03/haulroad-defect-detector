'use client'

import { useHealthCheck } from '@/hooks/useHealthCheck'
import { StatusIndicator } from '@/components/ui/status-indicator'
import { Activity, Clock, Server } from 'lucide-react'

export function HealthStatus() {
  const { health, isLoading, isHealthy, isDegraded } = useHealthCheck()

  const getStatus = () => {
    if (isLoading) return 'loading'
    if (isHealthy) return 'healthy'
    if (isDegraded) return 'degraded'
    return 'unhealthy'
  }

  const formatUptime = (seconds: number) => {
    const days = Math.floor(seconds / 86400)
    const hours = Math.floor((seconds % 86400) / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    
    if (days > 0) {
      return `${days}d ${hours}h ${minutes}m`
    }
    return `${hours}h ${minutes}m`
  }

  return (
    <div className="flex items-center gap-3 px-4 py-2 bg-white/20 backdrop-blur-sm rounded-lg border border-white/30">
      <div className="flex items-center gap-2">
        <StatusIndicator status={getStatus()} size="sm" />
        <span className="text-sm font-medium text-slate-700">
          {isLoading ? 'Checking...' : health?.status ?? 'Unknown'}
        </span>
      </div>
      
      {health && (
        <div className="flex items-center gap-4 text-xs text-slate-600">
          <div className="flex items-center gap-1">
            <Server className="w-3 h-3" />
            <span>v{health.version}</span>
          </div>
          <div className="flex items-center gap-1">
            <Activity className="w-3 h-3" />
            <span>Uptime: {formatUptime(health.uptime_seconds)}</span>
          </div>
          <div className="flex items-center gap-1">
            <Clock className="w-3 h-3" />
            <span>Model: {health.model_status}</span>
          </div>
        </div>
      )}
    </div>
  )
}