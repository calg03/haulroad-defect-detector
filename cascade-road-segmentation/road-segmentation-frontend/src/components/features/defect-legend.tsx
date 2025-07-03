'use client'

import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'

interface DefectType {
  id: string
  name: string
  bgColor: string
}

const defectTypes: DefectType[] = [
  {
    id: 'background',
    name: 'Background',
    bgColor: 'rgb(0, 0, 0)'
  },
  {
    id: 'pothole',
    name: 'Pothole',
    bgColor: 'rgb(0, 0, 255)'
  },
  {
    id: 'crack',
    name: 'Crack',
    bgColor: 'rgb(0, 255, 0)'
  },
  {
    id: 'puddle',
    name: 'Puddle',
    bgColor: 'rgb(140, 160, 222)'
  },
  {
    id: 'distressed_patch',
    name: 'Distressed Patch',
    bgColor: 'rgb(119, 61, 128)'
  },
  {
    id: 'mud',
    name: 'Mud',
    bgColor: 'rgb(112, 84, 62)'
  }
]

export function DefectLegend() {
  return (
    <Card className="h-fit">
      <CardHeader>
        <CardTitle className="text-slate-900 flex items-center gap-3">
          <div className="w-3 h-3 bg-gradient-to-r from-red-500 to-purple-500 rounded-full"></div>
          Defect Legend
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {defectTypes.map((defect) => (
            <div 
              key={defect.id}
              className="flex items-center gap-3"
            >
              <div 
                className="w-4 h-4 rounded-full shadow-sm border border-gray-300"
                style={{ backgroundColor: defect.bgColor }}
              ></div>
              <span className="text-sm font-medium text-slate-900">{defect.name}</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}