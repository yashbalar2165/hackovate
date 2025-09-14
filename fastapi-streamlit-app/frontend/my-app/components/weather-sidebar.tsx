"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Thermometer, Droplets, Wind, Gauge, AlertTriangle, Clock, TrendingUp } from "lucide-react"

interface Airport {
  id: string
  name: string
  code: string
  country: string
  lat: number
  lng: number
}

interface WeatherData {
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

interface WeatherSidebarProps {
  selectedAirport: Airport | null
  weatherData: WeatherData | null
  loading: boolean
  error: string | null
}

export function WeatherSidebar({ selectedAirport, weatherData, loading, error }: WeatherSidebarProps) {
  const getRiskLevel = (weatherCode: number) => {
    if (weatherCode > 50) {
      return { level: "HIGH RISK", color: "destructive", icon: "🚨", bgColor: "bg-red-500/10" }
    } else if (weatherCode > 10) {
      return { level: "MODERATE RISK", color: "secondary", icon: "⚠️", bgColor: "bg-yellow-500/10" }
    }
    return { level: "LOW RISK", color: "default", icon: "✅", bgColor: "bg-green-500/10" }
  }

  if (!selectedAirport) {
    return (
      <div className="space-y-6">
        <Card className="animate-fade-in-scale">
          <CardContent className="flex flex-col items-center justify-center h-64 text-center space-y-4">
            <div className="w-16 h-16 bg-muted rounded-full flex items-center justify-center">
              <Thermometer className="h-8 w-8 text-muted-foreground" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-foreground">Select an Airport</h3>
              <p className="text-muted-foreground">
                Click on any airport marker above to view detailed weather conditions
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="space-y-6 animate-slide-in-up">
      {/* Airport Info & Current Weather */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div>
              <div className="text-lg font-bold">{selectedAirport.name}</div>
              <div className="text-sm text-muted-foreground">
                {selectedAirport.code} - {selectedAirport.country}
              </div>
            </div>
            {loading && <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-secondary"></div>}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {error ? (
            <div className="text-center py-4">
              <AlertTriangle className="h-8 w-8 text-destructive mx-auto mb-2" />
              <p className="text-destructive text-sm">{error}</p>
            </div>
          ) : weatherData ? (
            <>
              {/* Weather Metrics Grid */}
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-muted/50 rounded-lg p-3 text-center">
                  <Thermometer className="h-5 w-5 text-orange-500 mx-auto mb-1" />
                  <div className="text-2xl font-bold text-orange-500">
                    {weatherData.forecast.temperature_2m.toFixed(1)}°C
                  </div>
                  <div className="text-xs text-muted-foreground">Temperature</div>
                </div>

                <div className="bg-muted/50 rounded-lg p-3 text-center">
                  <Droplets className="h-5 w-5 text-blue-500 mx-auto mb-1" />
                  <div className="text-2xl font-bold text-blue-500">{weatherData.forecast.relative_humidity_2m}%</div>
                  <div className="text-xs text-muted-foreground">Humidity</div>
                </div>

                <div className="bg-muted/50 rounded-lg p-3 text-center">
                  <Wind className="h-5 w-5 text-green-500 mx-auto mb-1" />
                  <div className="text-2xl font-bold text-green-500">
                    {weatherData.forecast.wind_speed_10m.toFixed(0)}
                  </div>
                  <div className="text-xs text-muted-foreground">Wind (km/h)</div>
                </div>

                <div className="bg-muted/50 rounded-lg p-3 text-center">
                  <Gauge className="h-5 w-5 text-purple-500 mx-auto mb-1" />
                  <div className="text-2xl font-bold text-purple-500">
                    {weatherData.forecast.pressure_msl.toFixed(1)}
                  </div>
                  <div className="text-xs text-muted-foreground">Pressure (hPa)</div>
                </div>
              </div>

              {/* Risk Assessment */}
              {(() => {
                const risk = getRiskLevel(weatherData.predicted_next3h_max_weathercode)
                return (
                  <div className={`${risk.bgColor} rounded-lg p-4 text-center`}>
                    <div className="text-2xl mb-2">{risk.icon}</div>
                    <Badge variant={risk.color as any} className="mb-2">
                      {risk.level}
                    </Badge>
                    <div className="text-sm text-muted-foreground">
                      Thunderstorm Score: {weatherData.predicted_next3h_max_weathercode.toFixed(2)}
                    </div>
                  </div>
                )
              })()}

              {/* Model Confidence */}
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium">Model Confidence</span>
                  <span className="text-sm text-muted-foreground">{weatherData.model_confidence || 75}%</span>
                </div>
                <Progress value={weatherData.model_confidence || 75} className="h-2" />
              </div>

              {/* Last Updated */}
              <div className="flex items-center justify-center space-x-2 text-xs text-muted-foreground pt-2 border-t">
                <Clock className="h-3 w-3" />
                <span>Last updated: {new Date(weatherData.last_updated || Date.now()).toLocaleString()}</span>
              </div>
            </>
          ) : (
            <div className="text-center py-8 space-y-2">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-secondary mx-auto"></div>
              <p className="text-muted-foreground text-sm">Loading weather data...</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* 24-Hour Forecast */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <TrendingUp className="h-5 w-5" />
            <span>Forecast (Next 24h)</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {weatherData?.forecast_24h && weatherData.forecast_24h.length > 0 ? (
            <div className="space-y-3">
              {weatherData.forecast_24h.map((item, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-3 bg-muted/50 rounded-lg hover:bg-muted/70 transition-colors"
                >
                  <div className="font-medium text-secondary text-sm min-w-[60px]">{item.time}</div>
                  <div className="flex-1 text-center text-sm">{item.condition}</div>
                  <div className="text-green-500 text-sm font-semibold">{item.probability}%</div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-4 text-muted-foreground text-sm">
              {loading ? "Loading forecast data..." : "No forecast data available"}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Active Alerts */}
      {weatherData?.active_alerts && weatherData.active_alerts.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2 text-destructive">
              <AlertTriangle className="h-5 w-5" />
              <span>Active Alerts</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {weatherData.active_alerts.map((alert, index) => (
                <div
                  key={index}
                  className="bg-destructive/10 border-l-4 border-destructive p-3 rounded-lg hover:bg-destructive/20 transition-colors"
                >
                  <div className="text-xs font-semibold text-muted-foreground mb-1">{alert.time}</div>
                  <div className="text-sm text-destructive">{alert.text}</div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
