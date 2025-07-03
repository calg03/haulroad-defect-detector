'use client'

import { useState, useEffect } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Brain, Cpu, Zap, Activity } from 'lucide-react'
import { apiClient } from '@/services/api/client'
import type { ModelInfo } from '@/services/types/api'

export function ModelStats() {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchModelInfo = async () => {
      try {
        const response = await apiClient.getModelInfo()
        if (response.success && response.data) {
          setModelInfo(response.data)
        }
      } catch (error) {
        console.error('Failed to fetch model info:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchModelInfo()
  }, [])

  if (loading) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center p-8">
          <div className="text-center">
            <div className="w-8 h-8 border-2 border-gray-300 border-t-blue-600 rounded-full animate-spin mx-auto mb-2" />
            <p className="text-sm text-gray-600">Loading model information...</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!modelInfo) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="w-5 h-5" />
            Model Information
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-4">
            <p className="text-gray-600">Unable to load model information</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'initialized':
      case 'ready':
        return 'default'
      case 'loading':
        return 'secondary'
      case 'error':
      case 'failed':
        return 'destructive'
      default:
        return 'outline'
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Brain className="w-5 h-5" />
          Model Information
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          {/* Status Overview */}
          <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-gray-600" />
              <span className="font-medium">Status</span>
            </div>
            <Badge variant={getStatusColor(modelInfo.status)}>
              {modelInfo.status}
            </Badge>
          </div>

          {/* Performance Metrics */}
          <div className="grid grid-cols-2 gap-4">
            <div className="text-center p-4 border border-gray-200 rounded-lg">
              <Cpu className="w-6 h-6 text-blue-600 mx-auto mb-2" />
              <p className="text-sm text-gray-600">Device</p>
              <p className="font-semibold text-gray-900">{modelInfo.device}</p>
            </div>

          </div>

          {/* Road Model Details */}
          <div className="space-y-3">
            <h3 className="font-semibold text-gray-900 flex items-center gap-2">
              <Zap className="w-4 h-4" />
              Road Segmentation Model
            </h3>

          </div>

          {/* Defect Model Details */}
          <div className="space-y-3">
            <h3 className="font-semibold text-gray-900 flex items-center gap-2">
              <Zap className="w-4 h-4" />
              Defect Detection Model
            </h3>

          </div>
        </div>
      </CardContent>
    </Card>
  )
}