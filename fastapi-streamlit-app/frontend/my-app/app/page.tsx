"use client"

import { useState } from "react"
import { WeatherMap } from "@/components/weather-map"
import { WeatherSidebar } from "@/components/weather-sidebar"
import { AlertBanner } from "@/components/alert-banner"
import { Header } from "@/components/header"
import { WeatherCharts } from "@/components/weather-charts"
import { WeatherMapLayers } from "@/components/weather-map-layers"
import { WeatherRadar } from "@/components/weather-radar"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

export interface Airport {
  id: string
  name: string
  code: string
  country: string
  lat: number
  lng: number
}

export interface WeatherData {
  forecast: {
    temperature_2m: number
    relative_humidity_2m: number
    wind_speed_10m: number
    pressure_msl: number
  }
  predicted_next3h_max_weathercode: number
  model_confidence: number
  last_updated: string
  forecast_24h: Array<{
    time: string
    condition: string
    probability: number
  }>
  active_alerts: Array<{
    time: string
    text: string
  }>
}

export default function WeatherDashboard() {
  const [selectedAirport, setSelectedAirport] = useState<Airport | null>(null)
  const [weatherData, setWeatherData] = useState<WeatherData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchWeatherData = async (airport: Airport) => {
    setLoading(true)
    setError(null)

    try {
      const apiUrl = `http://127.0.0.1:8000/thunderstrome?lat=${airport.lat}&lng=${airport.lng}&airport=${airport.code}&country=${airport.country}`

      const response = await fetch(apiUrl)
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      setWeatherData(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch weather data")
      console.error("Error fetching weather data:", err)
    } finally {
      setLoading(false)
    }
  }

  const handleAirportSelect = (airport: Airport) => {
    setSelectedAirport(airport)
    fetchWeatherData(airport)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-muted/30 to-background">
      <Header />

      <AlertBanner alerts={weatherData?.active_alerts || []} airportName={selectedAirport?.name || ""} />

      <main className="container mx-auto p-4 space-y-6">
        <Tabs defaultValue="overview" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="analytics">Analytics</TabsTrigger>
            <TabsTrigger value="layers">Weather Layers</TabsTrigger>
            <TabsTrigger value="radar">Radar</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="mt-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-[calc(100vh-250px)]">
              <div className="lg:col-span-2">
                <WeatherMap onAirportSelect={handleAirportSelect} selectedAirport={selectedAirport} />
              </div>
              <div className="lg:col-span-1">
                <WeatherSidebar
                  selectedAirport={selectedAirport}
                  weatherData={weatherData}
                  loading={loading}
                  error={error}
                />
              </div>
            </div>
          </TabsContent>

          <TabsContent value="analytics" className="mt-6">
            <WeatherCharts selectedAirport={selectedAirport} />
          </TabsContent>

          <TabsContent value="layers" className="mt-6">
            <WeatherMapLayers selectedAirport={selectedAirport} />
          </TabsContent>

          <TabsContent value="radar" className="mt-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <WeatherRadar selectedAirport={selectedAirport} />
              {selectedAirport && (
                <WeatherSidebar
                  selectedAirport={selectedAirport}
                  weatherData={weatherData}
                  loading={loading}
                  error={error}
                />
              )}
            </div>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  )
}
