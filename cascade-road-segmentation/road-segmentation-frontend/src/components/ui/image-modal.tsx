'use client'

import { useState, useEffect } from 'react'
import { createPortal } from 'react-dom'
import Image from 'next/image'
import { X, Download, ZoomIn, ZoomOut, RotateCw } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

interface ImageModalProps {
  readonly isOpen: boolean
  readonly onClose: () => void
  readonly imageSrc: string
  readonly imageAlt: string
  readonly filename: string
  readonly onDownload?: () => void
}

export function ImageModal({ 
  isOpen, 
  onClose, 
  imageSrc, 
  imageAlt, 
  filename, 
  onDownload 
}: ImageModalProps) {
  const [zoom, setZoom] = useState(1)
  const [rotation, setRotation] = useState(0)
  const [position, setPosition] = useState({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = useState(false)
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 })

  // Reset state when modal opens/closes
  useEffect(() => {
    if (isOpen) {
      setZoom(1)
      setRotation(0)
      setPosition({ x: 0, y: 0 })
      document.body.style.overflow = 'hidden'
    } else {
      document.body.style.overflow = 'unset'
    }

    return () => {
      document.body.style.overflow = 'unset'
    }
  }, [isOpen])

  // Handle escape key and other keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isOpen) return
      
      switch (e.key) {
        case 'Escape':
          onClose()
          break
        case '=':
        case '+':
          e.preventDefault()
          handleZoomIn()
          break
        case '-':
          e.preventDefault()
          handleZoomOut()
          break
        case 'r':
        case 'R':
          e.preventDefault()
          handleRotate()
          break
        case '0':
          e.preventDefault()
          handleReset()
          break
      }
    }

    if (isOpen) {
      document.addEventListener('keydown', handleKeyDown)
    }

    return () => {
      document.removeEventListener('keydown', handleKeyDown)
    }
  }, [isOpen, onClose, zoom])

  // Handle wheel zoom
  useEffect(() => {
    const handleWheel = (e: WheelEvent) => {
      if (!isOpen) return
      
      e.preventDefault()
      const delta = e.deltaY
      
      if (delta > 0) {
        handleZoomOut()
      } else {
        handleZoomIn()
      }
    }

    if (isOpen) {
      document.addEventListener('wheel', handleWheel, { passive: false })
    }

    return () => {
      document.removeEventListener('wheel', handleWheel)
    }
  }, [isOpen, zoom])

  const handleZoomIn = () => {
    setZoom(prev => Math.min(prev * 1.2, 5))
  }

  const handleZoomOut = () => {
    setZoom(prev => Math.max(prev / 1.2, 0.1))
  }

  const handleRotate = () => {
    setRotation(prev => (prev + 90) % 360)
  }

  const handleReset = () => {
    setZoom(1)
    setRotation(0)
    setPosition({ x: 0, y: 0 })
  }

  const handleMouseDown = (e: React.MouseEvent) => {
    if (zoom > 1) {
      setIsDragging(true)
      setDragStart({
        x: e.clientX - position.x,
        y: e.clientY - position.y
      })
    }
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging && zoom > 1) {
      setPosition({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y
      })
    }
  }

  const handleMouseUp = () => {
    setIsDragging(false)
  }

  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose()
    }
  }

  const getCursorStyle = () => {
    if (zoom > 1) {
      return isDragging ? 'grabbing' : 'grab'
    }
    return 'default'
  }

  if (!isOpen) return null

  const modalContent = (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop with blur */}
      <button
        className="absolute inset-0 bg-black/70 backdrop-blur-sm"
        onClick={handleBackdropClick}
        onKeyDown={(e) => {
          if (e.key === 'Escape') {
            onClose()
          }
        }}
        aria-label="Close modal"
        tabIndex={0}
      />
      
      {/* Modal Content */}
      <div className="relative z-10 w-full max-w-[96vw] max-h-[96vh] bg-white rounded-2xl shadow-2xl overflow-hidden border border-slate-200">
        {/* Header */}
        <div className="flex items-center justify-between p-6 bg-slate-50 border-b border-slate-200">
          <div className="flex-1 min-w-0">
            <h3 className="text-lg font-semibold text-slate-900 truncate mb-1">
              {filename}
            </h3>
            <p className="text-sm text-slate-500">{imageAlt}</p>
          </div>
          
          {/* Controls */}
          <div className="flex items-center gap-1 ml-6 bg-white rounded-xl p-2 shadow-sm border border-slate-200">
            <Button
              variant="ghost"
              size="sm"
              onClick={handleZoomOut}
              disabled={zoom <= 0.1}
              className="h-8 w-8  hover:bg-slate-100 disabled:opacity-30"
              title="Zoom out"
            >
              <ZoomOut className="w-4 h-4" />
            </Button>
            
            <div className="text-sm text-slate-600 min-w-[3.5rem] text-center font-medium px-2">
              {Math.round(zoom * 100)}%
            </div>
            
            <Button
              variant="ghost"
              size="sm"
              onClick={handleZoomIn}
              disabled={zoom >= 5}
              className="h-8 w-8  hover:bg-slate-100 disabled:opacity-30"
              title="Zoom in"
            >
              <ZoomIn className="w-4 h-4" />
            </Button>
            
            <div className="w-px h-5 bg-slate-200 mx-1"></div>
            
            <Button
              variant="ghost"
              size="sm"
              onClick={handleRotate}
              className="h-8 w-8 hover:bg-slate-100"
              title="Rotate 90°"
            >
              <RotateCw className="w-4 h-4" />
            </Button>
            
            <Button
              variant="ghost"
              size="sm"
              onClick={handleReset}
              className="text-xs px-3 h-8 hover:bg-slate-100 font-medium"
              title="Reset view"
            >
              Reset
            </Button>
            
            {onDownload && (
              <>
                <div className="w-px h-5 bg-slate-200 mx-1"></div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={onDownload}
                  className="h-8 px-3 border-slate-200 hover:bg-slate-50 font-medium"
                  title="Download image"
                >
                  <Download className="w-4 h-4 mr-1.5" />
                  Download
                </Button>
              </>
            )}
          </div>
          
          <Button
            variant="ghost"
            size="sm"
            onClick={onClose}
            className="h-8 w-8 ml-3 hover:bg-red-50 hover:text-red-600"
            title="Close modal"
          >
            <X className="w-4 h-4" />
          </Button>
        </div>
        
        {/* Image Container */}
        <div 
          className="relative overflow-hidden bg-slate-100 flex items-center justify-center"
          style={{ 
            height: 'calc(96vh - 160px)', // Account for header, footer, and padding
            minHeight: '400px'
          }}
        >
          <button
            className="absolute inset-0 bg-transparent border-0  outline-none focus:outline-none w-full h-full"
            style={{ 
              cursor: getCursorStyle()
            }}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            aria-label="Draggable image container - use mouse to pan when zoomed"
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault()
              }
            }}
          >
            <Image
              src={imageSrc}
              alt={imageAlt}
              width={800}
              height={600}
              className={cn(
                "absolute top-1/2 left-1/2 max-w-none transition-transform duration-200 select-none",
                isDragging ? "transition-none" : ""
              )}
              style={{
                transform: `translate(-50%, -50%) translate(${position.x}px, ${position.y}px) scale(${zoom}) rotate(${rotation}deg)`,
                transformOrigin: 'center'
              }}
              draggable={false}
              priority
            />
          </button>
        </div>
        
        {/* Footer with shortcuts */}
        <div className="px-6 py-3 bg-slate-50 border-t border-slate-200">
          <div className="flex items-center justify-center">
            <div className="text-xs text-slate-500 flex items-center gap-4">
              <span className="flex items-center gap-1">
                <kbd className="px-1.5 py-0.5 bg-white border border-slate-200 rounded text-xs">Scroll</kbd>
                <span>Zoom</span>
              </span>
              <span className="flex items-center gap-1">
                <kbd className="px-1.5 py-0.5 bg-white border border-slate-200 rounded text-xs">+/-</kbd>
                <span>Zoom In/Out</span>
              </span>
              <span className="flex items-center gap-1">
                <kbd className="px-1.5 py-0.5 bg-white border border-slate-200 rounded text-xs">R</kbd>
                <span>Rotate</span>
              </span>
              <span className="flex items-center gap-1">
                <kbd className="px-1.5 py-0.5 bg-white border border-slate-200 rounded text-xs">0</kbd>
                <span>Reset</span>
              </span>
              <span className="flex items-center gap-1">
                <kbd className="px-1.5 py-0.5 bg-white border border-slate-200 rounded text-xs">Esc</kbd>
                <span>Close</span>
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )

  return typeof window !== 'undefined' 
    ? createPortal(modalContent, document.body)
    : null
}
