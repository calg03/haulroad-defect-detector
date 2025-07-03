'use client'

import { useState, useEffect, useImperativeHandle, forwardRef } from 'react'
import Image from 'next/image'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { ImageModal } from '@/components/ui/image-modal'
import { Download, Eye, FolderOpen, FileImage, Layers } from 'lucide-react'
import { apiClient } from '@/services/api/client'
import type { BrowseResultEntry } from '@/services/types/api'

interface ModalState {
  isOpen: boolean
  imageSrc: string
  imageAlt: string
  filename: string
  downloadUrl?: string
}

export interface PreviousAnalysisRef {
  refresh: () => void
}

export const PreviousAnalysis = forwardRef<PreviousAnalysisRef>((props, ref) => {
  const [results, setResults] = useState<BrowseResultEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [modal, setModal] = useState<ModalState>({
    isOpen: false,
    imageSrc: '',
    imageAlt: '',
    filename: ''
  })

  const fetchResults = async () => {
    setLoading(true)
    try {
      const response = await apiClient.browseResults()
      if (response.success && response.data) {
        setResults(response.data.results)
      }
    } catch (error) {
      console.error('Failed to fetch previous results:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchResults()
  }, [])

  useImperativeHandle(ref, () => ({
    refresh: fetchResults
  }))

  const handleViewImage = (url: string, filename: string, type: string) => {
    const fullUrl = `${apiClient['baseUrl']}${url}`
    setModal({
      isOpen: true,
      imageSrc: fullUrl,
      imageAlt: `${type} - ${filename}`,
      filename,
      downloadUrl: url
    })
  }

  const handleDownload = (url: string, filename: string) => {
    const fullUrl = `${apiClient['baseUrl']}${url}`
    const a = document.createElement('a')
    a.href = fullUrl
    a.download = filename
    a.click()
  }

  const closeModal = () => {
    setModal(prev => ({ ...prev, isOpen: false }))
  }

  const handleModalDownload = () => {
    if (modal.downloadUrl) {
      handleDownload(modal.downloadUrl, modal.filename)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-8">
        <div className="text-center">
          <div className="w-10 h-10 border-2 border-slate-300 border-t-blue-600 rounded-full animate-spin mx-auto mb-4" />
          <p className="text-sm text-slate-600 font-medium">Loading previous analysis...</p>
        </div>
      </div>
    )
  }  return (
    <div className="space-y-6">
      {results.length === 0 ? (
        <div className="text-center py-8">
          <div className="w-16 h-16 bg-slate-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
            <FolderOpen className="w-8 h-8 text-slate-400" />
          </div>
          <p className="text-slate-700 font-medium mb-2">No previous analysis found</p>
          <p className="text-sm text-slate-500">Upload and analyze images to see results here</p>
        </div>
      ) : (
        <div className="space-y-6">
            {results.map((result, index) => (
              <div
                key={`${result.directory}-${index}`}
                className="bg-gradient-to-br from-white to-slate-50/50 border border-slate-200/50 rounded-xl hover:border-slate-300/50 hover:shadow-lg transition-all duration-300"
              >
                <div className="flex items-start justify-between mb-6">
                  <div className="flex-1">
                    <h3 className="text-lg font-semibold text-slate-900 mb-2">
                      {result.directory === '.' ? 'Root Analysis' : result.directory}
                    </h3>
                    <div className="flex items-center gap-3 flex-wrap">
                      <Badge variant="secondary" className="bg-slate-100 text-slate-700 border-slate-200">
                        {result.file_count} files
                      </Badge>
                      {result.files.overlays.length > 0 && (
                        <Badge variant="outline" className="border-blue-200 text-blue-700 bg-blue-50">
                          {result.files.overlays.length} overlays
                        </Badge>
                      )}
                      {result.files.road_masks.length > 0 && (
                        <Badge variant="outline" className="border-emerald-200 text-emerald-700 bg-emerald-50">
                          {result.files.road_masks.length} road masks
                        </Badge>
                      )}
                      {result.files.defect_masks.length > 0 && (
                        <Badge variant="outline" className="border-red-200 text-red-700 bg-red-50">
                          {result.files.defect_masks.length} defect masks
                        </Badge>
                      )}
                    </div>
                  </div>
                </div>


                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Overlay Images */}
                  {result.download_urls.overlays.length > 0 && (
                    <div>
                      <h4 className="text-sm font-semibold text-slate-700 mb-3 flex items-center gap-2">
                        <Layers className="w-4 h-4" />
                        Original with Overlay
                      </h4>
                      <div className="space-y-4">
                        {result.download_urls.overlays.slice(0, 2).map((url, idx) => {
                          const filename = url.split('/').pop() ?? `overlay_${idx}`
                          return (
                            <div key={`overlay-${url}-${idx}`} className="group">
                              <button 
                                className="relative aspect-video bg-gradient-to-br from-slate-50 to-slate-100 rounded-xl overflow-hidden border-2 border-slate-200/50 hover:border-blue-300 transition-all duration-200 w-full cursor-pointer"
                                onClick={() => handleViewImage(url, filename, 'Overlay')}
                                type="button"
                                aria-label={`View ${filename} overlay image`}
                              >
                                <Image
                                  src={`${apiClient['baseUrl']}${url}`}
                                  alt={filename}
                                  width={300}
                                  height={200}
                                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-200"
                                  loading="lazy"
                                />
                                <div className="absolute inset-0 bg-black/0 group-hover:bg-black/30 transition-all duration-200 flex items-center justify-center">
                                  <div className="opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                                    <Eye className="w-8 h-8 text-white" />
                                  </div>
                                </div>
                              </button>
                              <div className="flex items-center justify-between mt-2">
                                <p className="text-xs text-slate-600 truncate font-medium flex-1">{filename}</p>
                                <Button
                                  size="sm"
                                  variant="ghost"
                                  onClick={(e) => {
                                    e.stopPropagation()
                                    handleDownload(url, filename)
                                  }}
                                  className="h-6 w-6 text-slate-500 hover:text-blue-600"
                                >
                                  <Download className="w-3 h-3" />
                                </Button>
                              </div>
                            </div>
                          )
                        })}
                      </div>
                    </div>
                  )}

                  {/* Road Masks */}
                  {result.download_urls.road_masks.length > 0 && (
                    <div>
                      <h4 className="text-sm font-semibold text-slate-700 mb-3 flex items-center gap-2">
                        <FileImage className="w-4 h-4" />
                        Road Segmentation
                      </h4>
                      <div className="space-y-4">
                        {result.download_urls.road_masks.slice(0, 2).map((url, idx) => {
                          const filename = url.split('/').pop() ?? `road_mask_${idx}`
                          return (
                            <div key={`road-mask-${url}-${idx}`} className="group">
                              <button 
                                className="relative aspect-video bg-gradient-to-br from-slate-50 to-slate-100 rounded-xl overflow-hidden border-2 border-slate-200/50 hover:border-emerald-300 transition-all duration-200 w-full cursor-pointer "
                                onClick={() => handleViewImage(url, filename, 'Road Segmentation')}
                                type="button"
                                aria-label={`View ${filename} road segmentation image`}
                              >
                                <Image
                                  src={`${apiClient['baseUrl']}${url}`}
                                  alt={filename}
                                  width={300}
                                  height={200}
                                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-200"
                                  loading="lazy"
                                />
                                <div className="absolute inset-0 bg-black/0 group-hover:bg-black/30 transition-all duration-200 flex items-center justify-center">
                                  <div className="opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                                    <Eye className="w-8 h-8 text-white" />
                                  </div>
                                </div>
                              </button>
                              <div className="flex items-center justify-between mt-2">
                                <p className="text-xs text-slate-600 truncate font-medium flex-1">{filename}</p>
                                <Button
                                  size="sm"
                                  variant="ghost"
                                  onClick={(e) => {
                                    e.stopPropagation()
                                    handleDownload(url, filename)
                                  }}
                                  className="h-6 w-6 text-slate-500 hover:text-emerald-600"
                                >
                                  <Download className="w-3 h-3" />
                                </Button>
                              </div>
                            </div>
                          )
                        })}
                      </div>
                    </div>
                  )}
                </div>

                {/* Defect Masks - Full Width Below */}
                {result.download_urls.defect_masks.length > 0 && (
                  <div className="mt-6">
                    <h4 className="text-sm font-semibold text-slate-700 mb-3 flex items-center gap-2">
                      <FileImage className="w-4 h-4" />
                      Defect Analysis ({result.download_urls.defect_masks.length})
                    </h4>
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                      {result.download_urls.defect_masks.slice(0, 3).map((url, idx) => {
                        const filename = url.split('/').pop() ?? `defect_mask_${idx}`
                        return (
                          <div key={`defect-mask-${url}-${idx}`} className="group">
                            <button 
                              className="relative aspect-video bg-gradient-to-br from-slate-50 to-slate-100 rounded-xl overflow-hidden border-2 border-slate-200/50 hover:border-red-300 transition-all duration-200 w-full cursor-pointer"
                              onClick={() => handleViewImage(url, filename, 'Defect Analysis')}
                              type="button"
                              aria-label={`View ${filename} defect analysis image`}
                            >
                              <Image
                                src={`${apiClient['baseUrl']}${url}`}
                                alt={filename}
                                width={300}
                                height={200}
                                className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-200"
                                loading="lazy"
                              />
                              <div className="absolute inset-0 bg-black/0 group-hover:bg-black/30 transition-all duration-200 flex items-center justify-center">
                                <div className="opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                                  <Eye className="w-8 h-8 text-white" />
                                </div>
                              </div>
                            </button>
                            <div className="flex items-center justify-between mt-2">
                              <p className="text-xs text-slate-600 truncate font-medium flex-1">{filename}</p>
                              <Button
                                size="sm"
                                variant="ghost"
                                onClick={(e) => {
                                  e.stopPropagation()
                                  handleDownload(url, filename)
                                }}
                                className="h-6 w-6  text-slate-500 hover:text-red-600"
                              >
                                <Download className="w-3 h-3" />
                              </Button>
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )
      }
      
      {/* Image Modal */}
      <ImageModal
        isOpen={modal.isOpen}
        onClose={closeModal}
        imageSrc={modal.imageSrc}
        imageAlt={modal.imageAlt}
        filename={modal.filename}
        onDownload={modal.downloadUrl ? handleModalDownload : undefined}
      />
    </div>
  )
})

PreviousAnalysis.displayName = 'PreviousAnalysis'