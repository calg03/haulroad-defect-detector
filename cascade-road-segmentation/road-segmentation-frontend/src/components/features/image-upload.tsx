'use client'

import { useState, useRef } from 'react'
import Image from 'next/image'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Upload, X, Image as ImageIcon } from 'lucide-react'
import { cn } from '@/utils/cn'

interface ImageFile {
  file: File
  preview: string
  id: string
}

interface ImageUploadProps {
  readonly onUpload: (files: File[]) => void
  readonly maxFiles?: number
  readonly isUploading?: boolean
  readonly className?: string
}

export function ImageUpload({ 
  onUpload, 
  maxFiles = 5, 
  isUploading = false,
  className 
}: ImageUploadProps) {
  const [selectedFiles, setSelectedFiles] = useState<ImageFile[]>([])
  const [dragActive, setDragActive] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFiles = (files: FileList) => {
    const newFiles: ImageFile[] = []
    
    Array.from(files).forEach((file) => {
      if (file.type.startsWith('image/') && selectedFiles.length + newFiles.length < maxFiles) {
        const id = Math.random().toString(36).substring(7)
        const preview = URL.createObjectURL(file)
        newFiles.push({ file, preview, id })
      }
    })
    
    setSelectedFiles(prev => [...prev, ...newFiles])
  }

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    
    if (e.dataTransfer.files?.[0]) {
      handleFiles(e.dataTransfer.files)
    }
  }

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      handleFiles(e.target.files)
    }
  }

  const removeFile = (id: string) => {
    setSelectedFiles(prev => {
      const file = prev.find(f => f.id === id)
      if (file) {
        URL.revokeObjectURL(file.preview)
      }
      return prev.filter(f => f.id !== id)
    })
  }

  const handleUpload = () => {
    const files = selectedFiles.map(f => f.file)
    onUpload(files)
  }

  const clearAll = () => {
    selectedFiles.forEach(file => URL.revokeObjectURL(file.preview))
    setSelectedFiles([])
  }

  return (
    <Card className={cn('w-full', className)}>
      <CardHeader className="text-center pb-6">
        <CardTitle className="flex items-center justify-center gap-3 text-xl text-slate-900">
          <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl flex items-center justify-center">
            <ImageIcon className="w-4 h-4 text-white" />
          </div>
          Road Image Analysis
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <button
          type="button"
          className={cn(
            'w-full border-2 border-dashed rounded-2xl p-8 text-center transition-all duration-200 cursor-pointer',
            dragActive 
              ? 'border-blue-500 bg-blue-50 scale-[1.01] shadow-lg' 
              : 'border-slate-300 hover:border-blue-400',
            'hover:bg-blue-50/50 hover:shadow-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2'
          )}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept="image/*"
            onChange={handleFileInput}
            className="hidden"
          />
          
          <div className="flex flex-col items-center">
            <div className="w-16 h-16 bg-gradient-to-br from-slate-100 to-slate-200 rounded-2xl flex items-center justify-center mb-6 shadow-sm">
              <Upload className="w-8 h-8 text-slate-600" />
            </div>
            <h3 className="text-lg font-semibold mb-3 text-slate-900">
              Upload Road Images
            </h3>
            <p className="text-slate-600 mb-4 max-w-sm leading-relaxed">
              Drag and drop images here, or click to browse your files
            </p>
            <div className="flex items-center gap-6 text-sm text-slate-500">
              <span className="flex items-center gap-2">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                JPG, PNG formats
              </span>
              <span className="flex items-center gap-2">
                <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                Max {maxFiles} files
              </span>
            </div>
          </div>
        </button>

        {selectedFiles.length > 0 && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-slate-900 flex items-center gap-2">
                <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                Selected Images ({selectedFiles.length})
              </h3>
              <Button 
                variant="outline" 
                size="sm" 
                onClick={clearAll}
                disabled={isUploading}
                className="hover:bg-red-50 hover:border-red-300 hover:text-red-700 rounded-xl"
              >
                <X className="w-3 h-3 mr-1" />
                Clear All
              </Button>
            </div>
            
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
              {selectedFiles.map((file) => (
                <div key={file.id} className="relative group">
                  <div className="aspect-square rounded-2xl overflow-hidden bg-slate-100 border-2 border-slate-200 shadow-sm group-hover:shadow-md transition-shadow">
                    <Image
                      src={file.preview}
                      alt={file.file.name}
                      width={200}
                      height={200}
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <button
                    onClick={() => removeFile(file.id)}
                    disabled={isUploading}
                    className="absolute -top-2 -right-2 w-6 h-6 bg-red-500 hover:bg-red-600 text-white rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-all shadow-lg disabled:opacity-50"
                  >
                    <X className="w-3 h-3" />
                  </button>
                  <div className="mt-3">
                    <p className="text-xs text-slate-600 truncate text-center font-medium">
                      {file.file.name}
                    </p>
                    <p className="text-xs text-slate-400 text-center mt-1">
                      {(file.file.size / 1024).toFixed(1)} KB
                    </p>
                  </div>
                </div>
              ))}
            </div>

            <div className="flex justify-center pt-4">
              <Button
                onClick={handleUpload}
                disabled={selectedFiles.length === 0 || isUploading}
                size="lg"
                className="px-10 py-4 bg-gradient-to-r from-slate-800 to-slate-900 hover:from-slate-900 hover:to-black text-white shadow-lg hover:shadow-xl transition-all duration-200 rounded-2xl border-0 font-semibold tracking-wide"
              >
                {isUploading ? (
                  <>
                    <div className="w-4 h-4 mr-3 border-2 border-current border-t-transparent rounded-full animate-spin" />
                    Processing {selectedFiles.length} image{selectedFiles.length !== 1 ? 's' : ''}...
                  </>
                ) : (
                  <>
                    <div className="w-4 h-4 mr-3 bg-white/20 rounded-sm flex items-center justify-center">
                      <div className="w-2 h-2 bg-white rounded-full"></div>
                    </div>
                    Begin Analysis • {selectedFiles.length} Image{selectedFiles.length !== 1 ? 's' : ''}
                  </>
                )}
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}