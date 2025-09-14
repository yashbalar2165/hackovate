"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Radar, Play, Pause, RotateCcw } from "lucide-react"
import { useEffect, useState, useRef } from "react"

interface WeatherRadarProps {
  selectedAirport: {
    lat: number
    lng: number
    name: string
    code: string
  } | null
}

export function WeatherRadar({ selectedAirport }: WeatherRadarProps) {
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentFrame, setCurrentFrame] = useState(0)
  const [radarData, setRadarData] = useState<any[]>([])
  const intervalRef = useRef<NodeJS.Timeout | null>(null)

  const generateRadarFrames = () => {
    const frames = []
    for (let i = 0; i < 10; i++) {
      frames.push({
        timestamp: new Date(Date.now() - (9 - i) * 10 * 60 * 1000).toLocaleTimeString(),
        intensity: Math.random() * 100,
        coverage: Math.random() * 80 + 20,
        movement: Math.random() * 360,
      })
    }
    setRadarData(frames)
  }

  useEffect(() => {
    if (selectedAirport) {
      generateRadarFrames()
    }
  }, [selectedAirport])

  useEffect(() => {
    if (isPlaying && radarData.length > 0) {
      intervalRef.current = setInterval(() => {
        setCurrentFrame((prev) => (prev + 1) % radarData.length)
      }, 500)
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
        intervalRef.current = null
      }
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
    }
  }, [isPlaying, radarData.length])

  const handlePlayPause = () => {
    setIsPlaying(!isPlaying)
  }

  const handleReset = () => {
    setIsPlaying(false)
    setCurrentFrame(0)
  }

  if (!selectedAirport) {
    return (
      <Card className="animate-fade-in-scale">
        <CardContent className="flex flex-col items-center justify-center h-64 text-center space-y-4">
          <div className="w-16 h-16 bg-muted rounded-full flex items-center justify-center">
            <Radar className="h-8 w-8 text-muted-foreground" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-foreground">Weather Radar</h3>
            <p className="text-muted-foreground">Select an airport to view radar animation</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  const currentData = radarData[currentFrame]

  return (
    <Card className="animate-slide-in-up">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Radar className="h-5 w-5 text-green-500" />
            <span>Weather Radar - {selectedAirport.name}</span>
          </div>
          <Badge variant="secondary">{selectedAirport.code}</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Radar Controls */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handlePlayPause}
              className="flex items-center space-x-1 bg-transparent"
            >
              {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
              <span>{isPlaying ? "Pause" : "Play"}</span>
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={handleReset}
              className="flex items-center space-x-1 bg-transparent"
            >
              <RotateCcw className="h-4 w-4" />
              <span>Reset</span>
            </Button>
          </div>
          {currentData && (
            <Badge variant="outline">
              Frame {currentFrame + 1} of {radarData.length}
            </Badge>
          )}
        </div>

        {/* Radar Display */}
        <div className="relative bg-black rounded-lg overflow-hidden" style={{ aspectRatio: "1" }}>
          {/* Radar Grid */}
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="relative w-full h-full max-w-80 max-h-80">
              {/* Concentric circles */}
              {[1, 2, 3, 4].map((ring) => (
                <div
                  key={ring}
                  className="absolute border border-green-500/30 rounded-full"
                  style={{
                    width: `${ring * 25}%`,
                    height: `${ring * 25}%`,
                    top: `${50 - ring * 12.5}%`,
                    left: `${50 - ring * 12.5}%`,
                  }}
                />
              ))}

              {/* Cross lines */}
              <div className="absolute w-full h-0.5 bg-green-500/30 top-1/2 transform -translate-y-1/2" />
              <div className="absolute h-full w-0.5 bg-green-500/30 left-1/2 transform -translate-x-1/2" />

              {/* Radar sweep */}
              <div
                className="absolute top-1/2 left-1/2 origin-bottom w-0.5 bg-gradient-to-t from-green-400 to-transparent transform -translate-x-1/2 -translate-y-full transition-transform duration-500"
                style={{
                  height: "50%",
                  transform: `translate(-50%, -100%) rotate(${currentFrame * 36}deg)`,
                }}
              />

              {/* Weather echoes */}
              {currentData && (
                <>
                  {Array.from({ length: 8 }, (_, i) => (
                    <div
                      key={i}
                      className="absolute rounded-full animate-pulse"
                      style={{
                        width: `${Math.random() * 20 + 10}px`,
                        height: `${Math.random() * 20 + 10}px`,
                        top: `${Math.random() * 60 + 20}%`,
                        left: `${Math.random() * 60 + 20}%`,
                        backgroundColor: `rgba(${Math.random() > 0.5 ? "255, 255, 0" : "255, 0, 0"}, ${Math.random() * 0.8 + 0.2})`,
                      }}
                    />
                  ))}
                </>
              )}

              {/* Center dot */}
              <div className="absolute top-1/2 left-1/2 w-2 h-2 bg-green-400 rounded-full transform -translate-x-1/2 -translate-y-1/2" />
            </div>
          </div>
        </div>

        {/* Radar Info */}
        {currentData && (
          <div className="grid grid-cols-3 gap-4">
            <div className="text-center p-3 bg-muted/50 rounded-lg">
              <div className="text-lg font-bold text-green-500">{currentData.intensity.toFixed(1)}</div>
              <div className="text-xs text-muted-foreground">Intensity (dBZ)</div>
            </div>
            <div className="text-center p-3 bg-muted/50 rounded-lg">
              <div className="text-lg font-bold text-blue-500">{currentData.coverage.toFixed(0)}%</div>
              <div className="text-xs text-muted-foreground">Coverage</div>
            </div>
            <div className="text-center p-3 bg-muted/50 rounded-lg">
              <div className="text-lg font-bold text-yellow-500">{currentData.movement.toFixed(0)}°</div>
              <div className="text-xs text-muted-foreground">Movement</div>
            </div>
          </div>
        )}

        {/* Timestamp */}
        {currentData && <div className="text-center text-sm text-muted-foreground">{currentData.timestamp}</div>}
      </CardContent>
    </Card>
  )
}
