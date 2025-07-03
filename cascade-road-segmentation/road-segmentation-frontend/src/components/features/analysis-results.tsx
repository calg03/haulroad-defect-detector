'use client'

import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { 
  BarChart3, 
  Download, 
  Eye, 
  Route, 
  AlertTriangle,
  Zap,
  MapPin
} from 'lucide-react'
import { cn } from '@/utils/cn'
import type { InferenceResult } from '@/services/types/api'
import { apiClient } from '@/services/api/client'

interface AnalysisResultsProps {
  results: InferenceResult[]
  className?: string
}

export function AnalysisResults({ results, className }: AnalysisResultsProps) {
  if (results.length === 0) return null

  const totalDefects = results.reduce((sum, result) => 
    sum + result.total_defect_pixels, 0)
  
  const avgConfidence = results.reduce((sum, result) => 
    sum + result.mean_confidence, 0) / results.length

  const defectTypes = results.reduce((acc, result) => {
    Object.entries(result.defect_counts).forEach(([type, count]) => {
      acc[type] = (acc[type] || 0) + count
    })
    return acc
  }, {} as Record<string, number>)

  const defectTypeLabels: Record<string, string> = {
    alligator_cracking: 'Alligator Cracking',
    longitudinal_cracking: 'Longitudinal Cracking',
    transverse_cracking: 'Transverse Cracking',
    pothole: 'Potholes'
  }

  return (
    <div className={cn('space-y-4', className)}>
      {/* Summary Statistics */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="w-5 h-5" />
            Analysis Summary
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">{results.length}</div>
              <div className="text-sm text-gray-500">Images Analyzed</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">
                {totalDefects.toLocaleString()}
              </div>
              <div className="text-sm text-gray-500">Defect Pixels</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {(avgConfidence * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-gray-500">Avg Confidence</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">
                {Object.values(defectTypes).reduce((a, b) => a + b, 0)}
              </div>
              <div className="text-sm text-gray-500">Total Defects</div>
            </div>
          </div>

          {/* Defect Type Breakdown */}
          <div className="mt-6">
            <h4 className="text-sm font-medium mb-3">Defect Types Detected</h4>
            <div className="space-y-2">
              {Object.entries(defectTypes).map(([type, count]) => (
                <div key={type} className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">
                    {defectTypeLabels[type] || type}
                  </span>
                  <span className="font-medium">{count}</span>
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Individual Results */}
      <div className="grid gap-4">
        {results.map((result, index) => (
          <Card key={index} >
            <CardContent>
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    <Route className="w-4 h-4 text-blue-500" />
                    <h3 className="font-medium">{result.image_name}</h3>
                    <span className={cn(
                      'px-2 py-1 text-xs rounded-full',
                      result.processing_status === 'success' 
                        ? 'bg-green-100 text-green-800' 
                        : 'bg-red-100 text-red-800'
                    )}>
                      {result.processing_status}
                    </span>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                    <div className="flex items-center gap-1">
                      <MapPin className="w-3 h-3 text-gray-400" />
                      <span className="text-gray-600">
                        {result.image_shape[0]} × {result.image_shape[1]}
                      </span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Route className="w-3 h-3 text-blue-400" />
                      <span className="text-gray-600">
                        {(result.road_coverage * 100).toFixed(1)}% road
                      </span>
                    </div>
                    <div className="flex items-center gap-1">
                      <AlertTriangle className="w-3 h-3 text-orange-400" />
                      <span className="text-gray-600">
                        {result.total_defect_pixels} defects
                      </span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Zap className="w-3 h-3 text-green-400" />
                      <span className="text-gray-600">
                        {(result.mean_confidence * 100).toFixed(1)}% conf
                      </span>
                    </div>
                  </div>

                  {result.error_message && (
                    <div className="mt-2 p-2 bg-red-50 border border-red-200 rounded text-sm text-red-700">
                      {result.error_message}
                    </div>
                  )}
                </div>

                <div className="flex gap-2 ml-4">
                  {result.overlay_path && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        const fullUrl = `${apiClient['baseUrl']}${result.overlay_path}`
                        window.open(fullUrl, '_blank')
                      }}
                    >
                      <Eye className="w-3 h-3 mr-1" />
                      View
                    </Button>
                  )}
                  {result.overlay_path && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        const fullUrl = `${apiClient['baseUrl']}${result.overlay_path}`
                        const filename = result.overlay_path?.split('/').pop() || 'overlay.png'
                        const a = document.createElement('a')
                        a.href = fullUrl
                        a.download = filename
                        a.click()
                      }}
                    >
                      <Download className="w-3 h-3 mr-1" />
                      Download
                    </Button>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}