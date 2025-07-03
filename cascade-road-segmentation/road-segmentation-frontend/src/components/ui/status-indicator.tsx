'use client'

import { cn } from '@/utils/cn'
import { Circle } from 'lucide-react'

interface StatusIndicatorProps {
  status: 'healthy' | 'degraded' | 'unhealthy' | 'loading'
  size?: 'sm' | 'md' | 'lg'
  showText?: boolean
  className?: string
}

export function StatusIndicator({ 
  status, 
  size = 'md', 
  showText = false, 
  className 
}: StatusIndicatorProps) {
  const getStatusConfig = () => {
    switch (status) {
      case 'healthy':
        return {
          color: 'text-green-500',
          bg: 'bg-green-500',
          text: 'Healthy'
        }
      case 'degraded':
        return {
          color: 'text-yellow-500',
          bg: 'bg-yellow-500',
          text: 'Degraded'
        }
      case 'unhealthy':
        return {
          color: 'text-red-500',
          bg: 'bg-red-500',
          text: 'Unhealthy'
        }
      case 'loading':
        return {
          color: 'text-gray-400',
          bg: 'bg-gray-400',
          text: 'Checking...'
        }
    }
  }

  const config = getStatusConfig()
  
  const sizeClasses = {
    sm: 'w-2 h-2',
    md: 'w-3 h-3',
    lg: 'w-4 h-4'
  }

  return (
    <div className={cn('flex items-center gap-2', className)}>
      <div className={cn('relative', sizeClasses[size])}>
        <Circle 
          className={cn(
            'w-full h-full',
            config.bg,
            status === 'loading' && 'animate-pulse-slow'
          )} 
          fill="currentColor"
        />
        {status === 'healthy' && (
          <div className={cn(
            'absolute inset-0 rounded-full animate-ping',
            config.bg,
            'opacity-75'
          )} />
        )}
      </div>
      {showText && (
        <span className={cn('text-sm font-medium', config.color)}>
          {config.text}
        </span>
      )}
    </div>
  )
}