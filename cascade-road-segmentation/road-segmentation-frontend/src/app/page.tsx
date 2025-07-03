'use client'

import { useState, useEffect, useRef } from 'react'
import { HealthStatus } from '@/components/layout/health-status'
import { ImageUpload } from '@/components/features/image-upload'
import { AnalysisResults } from '@/components/features/analysis-results'
import { PreviousAnalysis, type PreviousAnalysisRef } from '@/components/features/previous-analysis'
import { RoadMap } from '@/components/maps/road-map'
import { DefectLegend } from '@/components/features/defect-legend'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { 
  Brain, 
  Zap, 
  Shield, 
  TrendingUp,
  GitBranch,
  Server
} from 'lucide-react'
import { apiClient } from '@/services/api/client'
import type { InferenceResult, ModelInfo } from '@/services/types/api'
import layoutStyles from '@/styles/layout.module.css'

export default function Home() {
  const [results, setResults] = useState<InferenceResult[]>([])
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null)
  const [isLoadingModel, setIsLoadingModel] = useState(false)
  const [modelError, setModelError] = useState<string | null>(null)
  const previousAnalysisRef = useRef<PreviousAnalysisRef>(null)

  const handleImageUpload = async (files: File[]) => {
    setIsAnalyzing(true)
    
    try {
      let newResults: InferenceResult[] = []
      
      if (files.length === 1) {
        const response = await apiClient.predictSingle(files[0])
        if (response.success && response.data) {
          newResults = [response.data]
        }
      } else {
        const response = await apiClient.predictBatch(files)
        if (response.success && response.data) {
          newResults = response.data.results
        }
      }
      
      setResults(prev => [...newResults, ...prev])
      // Refresh previous analysis component
      previousAnalysisRef.current?.refresh()
    } catch (error) {
      console.error('Analysis failed:', error)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const loadModelInfo = async () => {
    setIsLoadingModel(true)
    setModelError(null)
    try {
      const response = await apiClient.getModelInfo()
      console.log('Model info response:', response)
      if (response.success && response.data) {
        setModelInfo(response.data)
        console.log('Model info loaded:', response.data)
      } else {
        setModelError(response.message || 'Failed to load model information')
      }
    } catch (error) {
      console.error('Failed to load model info:', error)
      setModelError('Network error: Unable to connect to backend')
    } finally {
      setIsLoadingModel(false)
    }
  }

  // Load model info on mount
  useEffect(() => {
    loadModelInfo()
  }, [])

  return (
    <div className={layoutStyles.mainLayout}>
      {/* Header */}
      <header className={layoutStyles.stickyHeader}>
        <div className={layoutStyles.headerContent}>
          <div className={layoutStyles.headerInner}>
            <div className={layoutStyles.headerLeft}>
              <div className={layoutStyles.headerLogo}>
                <div className={layoutStyles.logoIcon}>
                  <Brain />
                </div>
                <div className={layoutStyles.logoBadge}></div>
              </div>
              <div>
                <h1 className={layoutStyles.headerTitle}>
                  Road Infrastructure Analysis
                </h1>
                <p className={layoutStyles.headerSubtitle}>
                  Computer Vision • Mining Operations • Industry 4.0
                </p>
              </div>
            </div>
            
            <div className={layoutStyles.headerRight}>
              <HealthStatus />
              <Button variant="outline" size="sm" className="glass-subtle border-white/30 hover:bg-white/10 px-4 py-2">
                <GitBranch className="w-4 h-4 mr-2" />
                Repository
              </Button>
            </div>
          </div>
        </div>
      </header>

      <main className={layoutStyles.mainContent}>
        {/* Hero Section */}
        <section className={layoutStyles.heroSection}>
          <div className={layoutStyles.heroBackground}></div>
          <div className={layoutStyles.heroContent}>
            <h2 className={layoutStyles.heroTitle}>
              Automated Road Infrastructure{' '}
              <span className={layoutStyles.heroGradientText}>Analysis</span>
            </h2>

            <div className={layoutStyles.heroFeatures}>
              <div className={layoutStyles.featureItem}>
                <div className={`${layoutStyles.featureIcon} ${layoutStyles.blue}`}>
                  <Zap />
                </div>
                <span className={layoutStyles.featureLabel}>Road Segmentation</span>
              </div>
              <div className={layoutStyles.featureItem}>
                <div className={`${layoutStyles.featureIcon} ${layoutStyles.green}`}>
                  <Shield />
                </div>
                <span className={layoutStyles.featureLabel}>Defect Detection</span>
              </div>
              <div className={layoutStyles.featureItem}>
                <div className={`${layoutStyles.featureIcon} ${layoutStyles.purple}`}>
                  <TrendingUp />
                </div>
                <span className={layoutStyles.featureLabel}>Pixel Analysis</span>
              </div>
            </div>
          </div>
        </section>

        {/* Main Content Grid */}
        <div className={layoutStyles.contentGrid}>
          {/* Left Column - Upload & Results */}
          <div className={layoutStyles.leftColumn}>
            <Card className={layoutStyles.glassCard}>
              <CardContent >
                <ImageUpload 
                  onUpload={handleImageUpload}
                  isUploading={isAnalyzing}
                />
              </CardContent>
            </Card>
            
            {results.length > 0 && (
              <Card className={layoutStyles.glassCard}>
                <CardContent >
                  <AnalysisResults results={results} />
                </CardContent>
              </Card>
            )}

            <Card className={layoutStyles.glassCard}>
              <CardHeader>
                <CardTitle className="flex items-center gap-3">
                  <div className="w-3 h-3 bg-blue-500 rounded-full animate-pulse"></div>
                  Analysis History
                </CardTitle>
              </CardHeader>
              <CardContent>
                <PreviousAnalysis ref={previousAnalysisRef} />
              </CardContent>
            </Card>
          </div>

          {/* Right Column - Map & Info */}
          <div className={layoutStyles.rightColumn}>
            <Card className={`${layoutStyles.glassCard} overflow-hidden`}>
              <CardHeader>
                <CardTitle className="flex items-center gap-3">
                  <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                  Operations Map
                </CardTitle>
              </CardHeader>
              <CardContent className="p-3">
                <div className="h-[360px] relative rounded-lg overflow-hidden">
                  <RoadMap className="h-full w-full" />
                </div>
              </CardContent>
            </Card>
            
            {/* Model Information Card - Now below the map */}
            <Card className={layoutStyles.glassCard}>
              <CardHeader>
                <CardTitle className="flex items-center gap-3">
                  <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
                  Model Information
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {isLoadingModel && (
                  <div className="grid grid-cols-1 gap-3">
                    <div className="p-3 bg-slate-50/50 rounded-lg animate-pulse">
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-slate-600">Architecture</span>
                        <div className="w-24 h-4 bg-slate-200 rounded"></div>
                      </div>
                      <div className="flex items-center justify-between mt-2">
                        <span className="text-xs text-slate-600">Status</span>
                        <div className="w-16 h-4 bg-slate-200 rounded"></div>
                      </div>
                    </div>
                  </div>
                )}
                
                {!isLoadingModel && modelInfo && (
                  <div className="space-y-4">
                    {/* Primary Model Info */}
                    <div className="p-3 bg-gradient-to-r from-purple-50 to-indigo-50/50 rounded-lg border border-purple-200/50">
                      <h4 className="text-sm font-semibold text-purple-900 mb-3 flex items-center gap-2">
                        <Server className="w-4 h-4" />
                        {modelInfo.architecture}
                      </h4>
                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <span className="text-xs text-slate-600">Status</span>
                          <div className="text-sm font-medium text-slate-800 capitalize flex items-center gap-2">
                            {(() => {
                              let statusColor = 'bg-red-500'
                              if (modelInfo.status === 'initialized') statusColor = 'bg-green-500'
                              else if (modelInfo.status === 'loading') statusColor = 'bg-yellow-500'
                              return <div className={`w-2 h-2 rounded-full ${statusColor}`}></div>
                            })()}
                            {modelInfo.status}
                          </div>
                        </div>
                        <div>
                          <span className="text-xs text-slate-600">Device</span>
                          <div className="text-sm font-medium text-slate-800 uppercase">{modelInfo.device}</div>
                        </div>
                        <div>
                          <span className="text-xs text-slate-600">Classes</span>
                          <div className="text-sm font-medium text-slate-800">{modelInfo.num_classes}</div>
                        </div>
                        <div>
                          <span className="text-xs text-slate-600">Threshold</span>
                          <div className="text-sm font-medium text-slate-800">{(modelInfo.confidence_threshold * 100).toFixed(0)}%</div>
                        </div>
                      </div>
                    </div>
                    
                    {/* Detection Classes */}
                    <div className="p-3 bg-gradient-to-r from-slate-50 to-blue-50/30 rounded-lg">
                      <h4 className="text-sm font-semibold text-slate-700 mb-3">Detection Classes</h4>
                      <div className="flex flex-wrap gap-2">
                        {modelInfo.classes.map((className) => (
                          <span 
                            key={className}
                            className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-white/80 text-slate-700 border border-slate-200 capitalize"
                          >
                            {className.replace('_', ' ')}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
                
                {!isLoadingModel && !modelInfo && (
                  <div className="p-3 bg-red-50/50 rounded-lg border border-red-200">
                    <div className="text-sm text-red-600 font-medium flex items-center gap-2">
                      <Server className="w-4 h-4" />
                      {modelError ?? 'Failed to load model information'}
                    </div>
                    <div className="text-xs text-red-500 mt-1">
                      Check backend connection and try refreshing
                    </div>
                    <button 
                      onClick={loadModelInfo}
                      className="mt-2 px-3 py-1 bg-red-100 hover:bg-red-200 text-red-700 text-xs rounded transition-colors"
                    >
                      Retry Connection
                    </button>
                  </div>
                )}
              </CardContent>
            </Card>
            
            <Card className={layoutStyles.glassCard}>
              <CardHeader>
                <CardTitle className="flex items-center gap-3">
                  <div className="w-3 h-3 bg-emerald-500 rounded-full"></div>
                  Session Statistics
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between p-3 bg-gradient-to-r from-slate-50 to-blue-50/50 rounded-lg border border-slate-200/50">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg">
                      <span className="text-white text-sm font-bold">{results.length}</span>
                    </div>
                    <div>
                      <span className="text-sm font-medium text-slate-700">Images Analyzed</span>
                      <div className="text-xs text-slate-500">
                        {results.filter(r => r.processing_status === 'success').length} successful
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="flex items-center justify-between p-3 bg-gradient-to-r from-slate-50 to-emerald-50/50 rounded-lg border border-slate-200/50">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-gradient-to-br from-emerald-500 to-teal-600 rounded-xl flex items-center justify-center shadow-lg">
                      <TrendingUp className="w-5 h-5 text-white" />
                    </div>
                    <div>
                      <span className="text-sm font-medium text-slate-700">Avg. Confidence</span>
                      <div className="text-xl font-bold text-slate-900">
                        {results.length > 0 
                          ? `${(results.reduce((sum, r) => sum + r.mean_confidence, 0) / results.length * 100).toFixed(1)}%`
                          : 'N/A'
                        }
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="flex items-center justify-between p-3 bg-gradient-to-r from-slate-50 to-amber-50/50 rounded-lg border border-slate-200/50">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-gradient-to-br from-amber-500 to-orange-600 rounded-xl flex items-center justify-center shadow-lg">
                      <Shield className="w-5 h-5 text-white" />
                    </div>
                    <div>
                      <span className="text-sm font-medium text-slate-700">Defect Pixels</span>
                      <div className="text-xl font-bold text-slate-900">
                        {results.reduce((sum, r) => sum + r.total_defect_pixels, 0).toLocaleString()}
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Defect Legend */}
            <DefectLegend />

          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className={layoutStyles.footer}>
        <div className={layoutStyles.footerContent}>
          <div className={layoutStyles.footerBrand}>
            <div className={layoutStyles.footerLogo}>
              <Brain />
            </div>
            <span className={layoutStyles.footerTitle}>RoadSense AI</span>
          </div>
        </div>
      </footer>
    </div>
  )
}