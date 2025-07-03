'use client'

import { useEffect, useState, useRef } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { MapPin, Navigation, Loader2 } from 'lucide-react'
import { cn } from '@/utils/cn'
import styles from './road-map.module.css'
import 'leaflet/dist/leaflet.css'

// Define types for Leaflet objects
type LeafletModule = typeof import('leaflet')
type LeafletMap = import('leaflet').Map
//type LeafletMarker = import('leaflet').Marker

interface RoadLocation {
  id: string
  lat: number
  lng: number
  name: string
  defectCount: number
  severity: 'low' | 'medium' | 'high'
}

interface RoadMapProps {
  readonly locations?: RoadLocation[]
  readonly className?: string
}

// Mock data for demonstration - Mining facilities in Asia
const defaultLocations: RoadLocation[] = [
  {
    id: '1',
    lat: 3.0738,
    lng: 101.5183,
    name: 'Kuala Lumpur Mining Access Road',
    defectCount: 12,
    severity: 'high'
  },
  {
    id: '2',
    lat: 3.1390,
    lng: 101.6869,
    name: 'Gombak Quarry Road',
    defectCount: 8,
    severity: 'medium'
  },
  {
    id: '3',
    lat: 2.9512,
    lng: 101.7981,
    name: 'Industrial Mining Complex',
    defectCount: 4,
    severity: 'low'
  },
  {
    id: '4',
    lat: 3.2079,
    lng: 101.6508,
    name: 'Transport Hub Access',
    defectCount: 6,
    severity: 'medium'
  }
]

export function RoadMap({ locations = defaultLocations, className }: RoadMapProps) {
  const mapRef = useRef<HTMLDivElement>(null)
  const [isMapLoaded, setIsMapLoaded] = useState(false)

  useEffect(() => {
    let leaflet: LeafletModule
    let map: LeafletMap

    const initializeMap = async () => {
      if (typeof window === 'undefined' || !mapRef.current) return

      try {
        // Dynamic import of Leaflet
        leaflet = await import('leaflet')
        
        // Fix for default markers
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        delete (leaflet.Icon.Default.prototype as any)._getIconUrl
        leaflet.Icon.Default.mergeOptions({
          iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
          iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
          shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
        })

        // Clear any existing map
        if (mapRef.current) {
          mapRef.current.innerHTML = ''
        }

        // Create map centered on Kuala Lumpur mining area
        map = leaflet.map(mapRef.current).setView([3.0738, 101.5183], 12)
        
        // Add tile layer
        leaflet.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
          attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }).addTo(map)

        // Add markers
        locations.forEach((location) => {
          const marker = leaflet.marker([location.lat, location.lng]).addTo(map)
          let severityClass = 'low'
          if (location.severity === 'high') {
            severityClass = 'high'
          } else if (location.severity === 'medium') {
            severityClass = 'medium'
          }
          
          marker.bindPopup(`
            <div class="leafletPopup">
              <div class="popupContent">
                <h3 class="popupTitle">${location.name}</h3>
                <div class="popupStats">
                  <span class="popupSeverity ${severityClass}">
                    ${location.severity.toUpperCase()}
                  </span>
                  <span class="popupDefectCount">
                    ${location.defectCount} defects
                  </span>
                </div>
              </div>
            </div>
          `)
        })

        setIsMapLoaded(true)
      } catch (error) {
        console.error('Error initializing map:', error)
      }
    }

    initializeMap()

    return () => {
      if (map) {
        map.remove()
      }
    }
  }, [locations])

  return (
    <Card className={cn(styles.roadMapContainer, className)}>
      <CardHeader className="pb-4">
        <CardTitle className="flex items-center gap-2 text-slate-800">
          <Navigation className="w-5 h-5 text-slate-600" />
          Mining Road Locations
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 flex flex-col">
        <div className={styles.mapWrapper}>
          <div 
            ref={mapRef} 
            className={styles.leafletMap}
          />
          {!isMapLoaded && (
            <div className={styles.loadingOverlay}>
              <Loader2 className={styles.loadingSpinner} />
            </div>
          )}
        </div>

        {/* Location List */}
        <div className={styles.locationsList}>
          <h4 className={styles.locationsTitle}>Recent Analysis Locations</h4>
          {locations.map((location) => (
            <div
              key={location.id}
              className={cn(styles.locationItem, styles[location.severity])}
            >
              <div className={styles.locationHeader}>
                <div className={styles.locationInfo}>
                  <MapPin className={styles.locationIcon} />
                  <span className={styles.locationName}>{location.name}</span>
                </div>
                <div className={styles.locationStats}>
                  <div className={cn(styles.defectCount, styles[location.severity])}>
                    {location.defectCount} defects
                  </div>
                  <div className={styles.severityLabel}>
                    {location.severity} severity
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}