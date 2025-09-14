"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts"
import { Thermometer, Droplets, Wind, Gauge, Zap } from "lucide-react"
import { useEffect, useState } from "react"

interface WeatherChartData {
  time: string
  temperature: number
  humidity: number
  windSpeed: number
  pressure: number
  precipitation: number
  thunderstormRisk: number
  visibility: number
}

interface WeatherChartsProps {
  selectedAirport: {
    lat: number
    lng: number
    name: string
    code: string
  } | null
}

export function WeatherCharts({ selectedAirport }: WeatherChartsProps) {
  const [chartData, setChartData] = useState<WeatherChartData[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Fetch weather data from OpenWeatherMap API
  const fetchWeatherChartData = async (lat: number, lng: number) => {
    setLoading(true)
    setError(null)

    try {
      // Fetch live weather data from Open-Meteo
      const url = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lng}&hourly=temperature_2m,relative_humidity_2m,precipitation,pressure_msl,windspeed_10m,weathercode,visibility&current_weather=true&timezone=auto`
      const response = await fetch(url)

      if (!response.ok) {
        console.log("[v0] Open-Meteo API failed, using mock data")
        generateMockData()
        return
      }

      const data = await response.json()
      const hourly = data.hourly
      const times = hourly.time.map((t: string) => {
        const d = new Date(t)
        return d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" })
      })

      // Process hourly data for charts (first 24 hours)
      const processedData: WeatherChartData[] = times.slice(0, 24).map((time: string, i: number) => ({
        time,
        temperature: Math.round(hourly.temperature_2m[i]),
        humidity: Math.round(hourly.relative_humidity_2m[i]),
        windSpeed: Math.round(hourly.windspeed_10m[i]),
        pressure: Math.round(hourly.pressure_msl[i]),
        precipitation: Math.round(hourly.precipitation[i] * 100), // Convert to percentage
        thunderstormRisk: hourly.weathercode[i] === 95 ? Math.random() * 40 + 60 : Math.random() * 30 + 10, // Mock risk based on weathercode
        visibility: Math.round((hourly.visibility?.[i] ?? 10000) / 1000), // Convert to km
      }))

      setChartData(processedData)
    } catch (err) {
      console.error("[v0] Error fetching weather data:", err)
      generateMockData()
    } finally {
      setLoading(false)
    }
  }

  // Generate realistic mock data
  const generateMockData = () => {
    const mockData: WeatherChartData[] = Array.from({ length: 24 }, (_, i) => {
      const hour = new Date()
      hour.setHours(hour.getHours() + i)

      return {
        time: hour.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" }),
        temperature: Math.round(20 + Math.sin(i * 0.3) * 8 + Math.random() * 4),
        humidity: Math.round(60 + Math.sin(i * 0.2) * 20 + Math.random() * 10),
        windSpeed: Math.round(15 + Math.sin(i * 0.4) * 10 + Math.random() * 5),
        pressure: Math.round(1013 + Math.sin(i * 0.1) * 15 + Math.random() * 5),
        precipitation: Math.round(Math.max(0, Math.sin(i * 0.5) * 30 + Math.random() * 20)),
        thunderstormRisk: Math.round(Math.max(0, Math.sin(i * 0.3) * 40 + Math.random() * 30)),
        visibility: Math.round(8 + Math.sin(i * 0.2) * 3 + Math.random() * 2),
      }
    })

    setChartData(mockData)
  }

  useEffect(() => {
    if (selectedAirport) {
      fetchWeatherChartData(selectedAirport.lat, selectedAirport.lng)
    }
  }, [selectedAirport])

  if (!selectedAirport) {
    return (
      <Card className="animate-fade-in-scale">
        <CardContent className="flex flex-col items-center justify-center h-64 text-center space-y-4">
          <div className="w-16 h-16 bg-muted rounded-full flex items-center justify-center">
            <Thermometer className="h-8 w-8 text-muted-foreground" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-foreground">Weather Analytics</h3>
            <p className="text-muted-foreground">Select an airport to view detailed weather charts and forecasts</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (loading) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center h-64">
          <div className="text-center space-y-4">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-secondary mx-auto"></div>
            <p className="text-muted-foreground">Loading weather charts...</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-6 animate-slide-in-up">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Zap className="h-5 w-5 text-yellow-500" />
              <span>Weather Analytics - {selectedAirport.name}</span>
            </div>
            <Badge variant="secondary">{selectedAirport.code}</Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="temperature" className="w-full">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="temperature" className="flex items-center space-x-1">
                <Thermometer className="h-4 w-4" />
                <span className="hidden sm:inline">Temperature</span>
              </TabsTrigger>
              <TabsTrigger value="wind" className="flex items-center space-x-1">
                <Wind className="h-4 w-4" />
                <span className="hidden sm:inline">Wind</span>
              </TabsTrigger>
              <TabsTrigger value="humidity" className="flex items-center space-x-1">
                <Droplets className="h-4 w-4" />
                <span className="hidden sm:inline">Humidity</span>
              </TabsTrigger>
              <TabsTrigger value="thunderstorm" className="flex items-center space-x-1">
                <Zap className="h-4 w-4" />
                <span className="hidden sm:inline">Storms</span>
              </TabsTrigger>
            </TabsList>

            <TabsContent value="temperature" className="mt-6">
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis dataKey="time" stroke="hsl(var(--muted-foreground))" fontSize={12} />
                    <YAxis
                      stroke="hsl(var(--muted-foreground))"
                      fontSize={12}
                      label={{ value: "°C", angle: -90, position: "insideLeft" }}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "hsl(var(--card))",
                        border: "1px solid hsl(var(--border))",
                        borderRadius: "8px",
                        color: "hsl(var(--card-foreground))",
                      }}
                      formatter={(value: number) => [`${value}°C`, "Temperature"]}
                    />
                    <Area
                      type="monotone"
                      dataKey="temperature"
                      stroke="#f97316"
                      fill="#f97316"
                      fillOpacity={0.3}
                      strokeWidth={2}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </TabsContent>

            <TabsContent value="wind" className="mt-6">
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis dataKey="time" stroke="hsl(var(--muted-foreground))" fontSize={12} />
                    <YAxis
                      stroke="hsl(var(--muted-foreground))"
                      fontSize={12}
                      label={{ value: "km/h", angle: -90, position: "insideLeft" }}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "hsl(var(--card))",
                        border: "1px solid hsl(var(--border))",
                        borderRadius: "8px",
                        color: "hsl(var(--card-foreground))",
                      }}
                      formatter={(value: number) => [`${value} km/h`, "Wind Speed"]}
                    />
                    <Line
                      type="monotone"
                      dataKey="windSpeed"
                      stroke="#10b981"
                      strokeWidth={3}
                      dot={{ fill: "#10b981", strokeWidth: 2, r: 4 }}
                      activeDot={{ r: 6, stroke: "#10b981", strokeWidth: 2 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </TabsContent>

            <TabsContent value="humidity" className="mt-6">
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis dataKey="time" stroke="hsl(var(--muted-foreground))" fontSize={12} />
                    <YAxis
                      stroke="hsl(var(--muted-foreground))"
                      fontSize={12}
                      domain={[0, 100]}
                      label={{ value: "%", angle: -90, position: "insideLeft" }}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "hsl(var(--card))",
                        border: "1px solid hsl(var(--border))",
                        borderRadius: "8px",
                        color: "hsl(var(--card-foreground))",
                      }}
                      formatter={(value: number) => [`${value}%`, "Humidity"]}
                    />
                    <Area
                      type="monotone"
                      dataKey="humidity"
                      stroke="#3b82f6"
                      fill="#3b82f6"
                      fillOpacity={0.3}
                      strokeWidth={2}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </TabsContent>

            <TabsContent value="thunderstorm" className="mt-6">
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis dataKey="time" stroke="hsl(var(--muted-foreground))" fontSize={12} />
                    <YAxis
                      stroke="hsl(var(--muted-foreground))"
                      fontSize={12}
                      domain={[0, 100]}
                      label={{ value: "Risk %", angle: -90, position: "insideLeft" }}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "hsl(var(--card))",
                        border: "1px solid hsl(var(--border))",
                        borderRadius: "8px",
                        color: "hsl(var(--card-foreground))",
                      }}
                      formatter={(value: number) => [`${value}%`, "Thunderstorm Risk"]}
                    />
                    <Bar dataKey="thunderstormRisk" fill="#eab308" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      {/* Multi-parameter Overview Chart */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Gauge className="h-5 w-5" />
            <span>Multi-Parameter Overview</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis dataKey="time" stroke="hsl(var(--muted-foreground))" fontSize={12} />
                <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "hsl(var(--card))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "8px",
                    color: "hsl(var(--card-foreground))",
                  }}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="temperature"
                  stroke="#f97316"
                  strokeWidth={2}
                  name="Temperature (°C)"
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="humidity"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  name="Humidity (%)"
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="windSpeed"
                  stroke="#10b981"
                  strokeWidth={2}
                  name="Wind Speed (km/h)"
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="thunderstormRisk"
                  stroke="#eab308"
                  strokeWidth={3}
                  name="Thunderstorm Risk (%)"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
