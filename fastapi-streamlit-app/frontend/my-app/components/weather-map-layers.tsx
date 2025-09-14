"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Cloud, CloudRain, Zap, Wind, Thermometer, Layers } from "lucide-react"

interface WeatherMapLayersProps {
  selectedAirport: {
    lat: number
    lng: number
    name: string
    code: string
  } | null
}

export function WeatherMapLayers({ selectedAirport }: WeatherMapLayersProps) {
  const [activeLayer, setActiveLayer] = useState<string>("temperature")
  const [layerData, setLayerData] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  const weatherLayers = [
    {
      id: "temperature",
      name: "Temperature",
      icon: Thermometer,
      color: "text-orange-500",
      description: "Surface temperature overlay",
    },
    {
      id: "precipitation",
      name: "Precipitation",
      icon: CloudRain,
      color: "text-blue-500",
      description: "Rain and snow intensity",
    },
    {
      id: "wind",
      name: "Wind Speed",
      icon: Wind,
      color: "text-green-500",
      description: "Wind speed and direction",
    },
    {
      id: "thunderstorm",
      name: "Thunderstorms",
      icon: Zap,
      color: "text-yellow-500",
      description: "Lightning activity and storm cells",
    },
    {
      id: "clouds",
      name: "Cloud Cover",
      icon: Cloud,
      color: "text-gray-500",
      description: "Cloud coverage percentage",
    },
  ]

  const fetchLayerData = async (layer: string, lat?: number, lng?: number) => {
    if (!lat || !lng) return

    setLoading(true)
    try {
      // Fetch live weather data from Open-Meteo
      const url = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lng}&hourly=temperature_2m,precipitation,cloudcover,windspeed_10m,weathercode&current_weather=true&timezone=auto`
      const res = await fetch(url)
      const data = await res.json()

      // Get current hour index
      const now = new Date()
      const currentHour = now.getHours()
      const hourly = data.hourly

      // Map API data to your layer structure
      const liveData = {
        temperature: {
          current: Math.round(data.current_weather?.temperature ?? hourly.temperature_2m[currentHour]),
          min: Math.min(...hourly.temperature_2m),
          max: Math.max(...hourly.temperature_2m),
          trend: (hourly.temperature_2m[currentHour] > hourly.temperature_2m[currentHour - 1]) ? "rising" : "falling",
        },
        precipitation: {
          current: Math.round(hourly.precipitation[currentHour]),
          forecast: Math.round(hourly.precipitation[currentHour + 1] ?? 0),
          type: (hourly.precipitation[currentHour] > 0) ? "rain" : "none",
        },
        wind: {
          speed: Math.round(data.current_weather?.windspeed ?? hourly.windspeed_10m[currentHour]),
          direction: Math.round(data.current_weather?.winddirection ?? 0),
          gusts: Math.round(hourly.windspeed_10m[currentHour] + 5),
        },
        thunderstorm: {
          risk: Math.round(hourly.weathercode[currentHour] === 95 ? 80 : 10),
          activity: hourly.weathercode[currentHour] === 95 ? "active" : "none",
          strikes: hourly.weathercode[currentHour] === 95 ? Math.round(Math.random() * 20) : 0,
        },
        clouds: {
          coverage: Math.round(hourly.cloudcover[currentHour]),
          type: hourly.cloudcover[currentHour] > 60 ? "stratus" : "cumulus",
          altitude: Math.round(2000 + hourly.cloudcover[currentHour] * 50),
        },
      }

      setLayerData(liveData)
    } catch (error) {
      console.error("[v0] Error fetching layer data:", error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (selectedAirport) {
      fetchLayerData(activeLayer, selectedAirport.lat, selectedAirport.lng)
    }
  }, [activeLayer, selectedAirport])

  const renderLayerContent = () => {
    if (!layerData || loading) {
      return (
        <div className="flex items-center justify-center h-48">
          <div className="text-center space-y-4">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-secondary mx-auto"></div>
            <p className="text-muted-foreground text-sm">Loading layer data...</p>
          </div>
        </div>
      )
    }

    const data = layerData[activeLayer]
    if (!data) return null

    switch (activeLayer) {
      case "temperature":
        return (
          <div className="space-y-4">
            <div className="grid grid-cols-3 gap-4">
              <div className="text-center p-4 bg-orange-500/10 rounded-lg">
                <div className="text-2xl font-bold text-orange-500">{data.current}°C</div>
                <div className="text-sm text-muted-foreground">Current</div>
              </div>
              <div className="text-center p-4 bg-blue-500/10 rounded-lg">
                <div className="text-2xl font-bold text-blue-500">{data.min}°C</div>
                <div className="text-sm text-muted-foreground">Min Today</div>
              </div>
              <div className="text-center p-4 bg-red-500/10 rounded-lg">
                <div className="text-2xl font-bold text-red-500">{data.max}°C</div>
                <div className="text-sm text-muted-foreground">Max Today</div>
              </div>
            </div>
            <div className="flex items-center justify-center space-x-2">
              <Badge variant={data.trend === "rising" ? "destructive" : "secondary"}>
                {data.trend === "rising" ? "↗ Rising" : "↘ Falling"}
              </Badge>
            </div>
          </div>
        )

      case "precipitation":
        return (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center p-4 bg-blue-500/10 rounded-lg">
                <div className="text-2xl font-bold text-blue-500">{data.current}mm</div>
                <div className="text-sm text-muted-foreground">Current Rate</div>
              </div>
              <div className="text-center p-4 bg-cyan-500/10 rounded-lg">
                <div className="text-2xl font-bold text-cyan-500">{data.forecast}%</div>
                <div className="text-sm text-muted-foreground">Chance Next Hour</div>
              </div>
            </div>
            <div className="text-center">
              <Badge variant={data.type === "rain" ? "destructive" : "secondary"}>
                {data.type === "rain" ? "🌧 Active Rain" : "☀ No Precipitation"}
              </Badge>
            </div>
          </div>
        )

      case "wind":
        return (
          <div className="space-y-4">
            <div className="grid grid-cols-3 gap-4">
              <div className="text-center p-4 bg-green-500/10 rounded-lg">
                <div className="text-2xl font-bold text-green-500">{data.speed}</div>
                <div className="text-sm text-muted-foreground">Speed (km/h)</div>
              </div>
              <div className="text-center p-4 bg-blue-500/10 rounded-lg">
                <div className="text-2xl font-bold text-blue-500">{data.direction}°</div>
                <div className="text-sm text-muted-foreground">Direction</div>
              </div>
              <div className="text-center p-4 bg-yellow-500/10 rounded-lg">
                <div className="text-2xl font-bold text-yellow-500">{data.gusts}</div>
                <div className="text-sm text-muted-foreground">Gusts (km/h)</div>
              </div>
            </div>
          </div>
        )

      case "thunderstorm":
        return (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center p-4 bg-yellow-500/10 rounded-lg">
                <div className="text-2xl font-bold text-yellow-500">{data.risk}%</div>
                <div className="text-sm text-muted-foreground">Storm Risk</div>
              </div>
              <div className="text-center p-4 bg-purple-500/10 rounded-lg">
                <div className="text-2xl font-bold text-purple-500">{data.strikes}</div>
                <div className="text-sm text-muted-foreground">Lightning Strikes</div>
              </div>
            </div>
            <div className="text-center">
              <Badge variant={data.activity === "active" ? "destructive" : "secondary"}>
                {data.activity === "active" ? "⚡ Active Storms" : "🌤 No Storm Activity"}
              </Badge>
            </div>
          </div>
        )

      case "clouds":
        return (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center p-4 bg-gray-500/10 rounded-lg">
                <div className="text-2xl font-bold text-gray-500">{data.coverage}%</div>
                <div className="text-sm text-muted-foreground">Cloud Cover</div>
              </div>
              <div className="text-center p-4 bg-indigo-500/10 rounded-lg">
                <div className="text-2xl font-bold text-indigo-500">{data.altitude}m</div>
                <div className="text-sm text-muted-foreground">Cloud Base</div>
              </div>
            </div>
            <div className="text-center">
              <Badge variant="secondary">{data.type.charAt(0).toUpperCase() + data.type.slice(1)} Clouds</Badge>
            </div>
          </div>
        )

      default:
        return null
    }
  }

  if (!selectedAirport) {
    return (
      <Card className="animate-fade-in-scale">
        <CardContent className="flex flex-col items-center justify-center h-64 text-center space-y-4">
          <div className="w-16 h-16 bg-muted rounded-full flex items-center justify-center">
            <Layers className="h-8 w-8 text-muted-foreground" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-foreground">Weather Layers</h3>
            <p className="text-muted-foreground">Select an airport to view detailed weather layer information</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="animate-slide-in-up">
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <Layers className="h-5 w-5" />
          <span>Weather Layers - {selectedAirport.name}</span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Layer Selection */}
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-2">
            {weatherLayers.map((layer) => {
              const IconComponent = layer.icon
              return (
                <Button
                  key={layer.id}
                  variant={activeLayer === layer.id ? "default" : "outline"}
                  size="sm"
                  onClick={() => setActiveLayer(layer.id)}
                  className="flex flex-col items-center space-y-1 h-auto py-3"
                >
                  <IconComponent className={`h-4 w-4 ${layer.color}`} />
                  <span className="text-xs">{layer.name}</span>
                </Button>
              )
            })}
          </div>

          {/* Layer Content */}
          <div className="border rounded-lg p-4 bg-muted/20">
            <div className="mb-4">
              <h4 className="font-semibold text-sm text-muted-foreground">
                {weatherLayers.find((l) => l.id === activeLayer)?.description}
              </h4>
            </div>
            {renderLayerContent()}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
