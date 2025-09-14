"use client"

import { useEffect, useRef, useState } from "react"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { MapPin } from "lucide-react"

interface Airport {
  id: string
  name: string
  code: string
  country: string
  lat: number
  lng: number
}

interface WeatherMapProps {
  onAirportSelect: (airport: Airport) => void
  selectedAirport: Airport | null
}

export function WeatherMap({ onAirportSelect, selectedAirport }: WeatherMapProps) {
  const mapRef = useRef<HTMLDivElement>(null)
  const [airports, setAirports] = useState<Airport[]>([])
  const [loading, setLoading] = useState(true)
  const [mapInstance, setMapInstance] = useState<any>(null)
  const markersRef = useRef<{ [key: string]: any }>({})

  // Airport data - same as your original data
  const airportData = [
    // Normal conditions airports
    { name: "Hartsfield-Jackson Atlanta", code: "ATL", country: "USA", lat: 33.6407, lng: -84.4277 },
    { name: "Los Angeles International", code: "LAX", country: "USA", lat: 33.9425, lng: -118.4081 },
    { name: "Dubai International", code: "DXB", country: "UAE", lat: 25.2532, lng: 55.3657 },
    { name: "Tokyo Haneda", code: "HND", country: "Japan", lat: 35.5494, lng: 139.7798 },
    { name: "London Heathrow", code: "LHR", country: "UK", lat: 51.47, lng: -0.4543 },
    { name: "Paris Charles de Gaulle", code: "CDG", country: "France", lat: 49.0097, lng: 2.5479 },
    { name: "Amsterdam Schiphol", code: "AMS", country: "Netherlands", lat: 52.3086, lng: 4.7639 },
    { name: "Frankfurt am Main", code: "FRA", country: "Germany", lat: 50.0379, lng: 8.5622 },
    { name: "Istanbul Airport", code: "IST", country: "Turkey", lat: 41.2753, lng: 28.7519 },
    { name: "Beijing Capital", code: "PEK", country: "China", lat: 40.0799, lng: 116.6031 },
    { name: "Shanghai Pudong", code: "PVG", country: "China", lat: 31.1443, lng: 121.8083 },
    { name: "Singapore Changi", code: "SIN", country: "Singapore", lat: 1.3644, lng: 103.9915 },
    { name: "Sydney Kingsford Smith", code: "SYD", country: "Australia", lat: -33.9399, lng: 151.1753 },
    { name: "Delhi Indira Gandhi", code: "DEL", country: "India", lat: 28.5562, lng: 77.1 },
    { name: "Mumbai Chhatrapati Shivaji", code: "BOM", country: "India", lat: 19.0896, lng: 72.8656 },
    { name: "Bangalore Kempegowda", code: "BLR", country: "India", lat: 13.1986, lng: 77.7066 },
    { name: "Hyderabad Rajiv Gandhi", code: "HYD", country: "India", lat: 17.2313, lng: 78.4298 },
    { name: "Chennai International", code: "MAA", country: "India", lat: 12.9941, lng: 80.1709 },
    { name: "Kolkata Netaji Subhas Chandra Bose", code: "CCU", country: "India", lat: 22.6548, lng: 88.4467 },
    { name: "Ahmedabad Sardar Vallabhbhai Patel", code: "AMD", country: "India", lat: 23.0726, lng: 72.6344 },
    { name: "Pune International", code: "PNQ", country: "India", lat: 18.5821, lng: 73.9197 },
    { name: "Sao Paulo Guarulhos", code: "GRU", country: "Brazil", lat: -23.4356, lng: -46.4731 },
    { name: "Mexico City International", code: "MEX", country: "Mexico", lat: 19.4363, lng: -99.0721 },
    { name: "Toronto Pearson", code: "YYZ", country: "Canada", lat: 43.6777, lng: -79.6248 },
    { name: "Moscow Domodedovo", code: "DME", country: "Russia", lat: 55.4086, lng: 37.9061 },
    { name: "Cairo International", code: "CAI", country: "Egypt", lat: 30.1219, lng: 31.4056 },
    { name: "Johannesburg OR Tambo", code: "JNB", country: "South Africa", lat: -26.1367, lng: 28.2411 },
    { name: "Lagos Murtala Muhammed", code: "LOS", country: "Nigeria", lat: 6.5774, lng: 3.321 },

    // EXTREME WEATHER CONDITIONS
    { name: "Chicago O'Hare Storm Zone", code: "ORD", country: "USA", lat: 41.9786, lng: -87.9048 },
    { name: "Wellington Windy Airport", code: "WLG", country: "New Zealand", lat: -41.3272, lng: 174.8055 },
    { name: "Patagonia High Wind Base", code: "EZE", country: "Argentina", lat: -34.8222, lng: -58.5358 },
    { name: "Seattle Storm Center", code: "SEA", country: "USA", lat: 47.4502, lng: -122.3088 },
    { name: "Monsoon Mumbai Override", code: "BOM2", country: "India", lat: 19.0896, lng: 72.8656 },
    { name: "Tropical Storm Base", code: "MIA", country: "USA", lat: 25.7959, lng: -80.287 },
    { name: "Hurricane Tracking Station", code: "KEY", country: "USA", lat: 24.5557, lng: -81.7594 },
    { name: "Cyclone Watch Darwin", code: "DRW", country: "Australia", lat: -12.4074, lng: 130.8756 },
    { name: "Typhoon Alert Okinawa", code: "OKA", country: "Japan", lat: 26.1958, lng: 127.6458 },
    { name: "Desert Heat Phoenix", code: "PHX", country: "USA", lat: 33.4342, lng: -112.0116 },
    { name: "Arctic Cold Fairbanks", code: "FAI", country: "USA", lat: 64.8151, lng: -147.8761 },
    { name: "Sahara Heat Station", code: "ASW", country: "Egypt", lat: 24.0934, lng: 32.8998 },
    { name: "Siberian Freeze Base", code: "YKS", country: "Russia", lat: 62.0939, lng: 129.7711 },
    { name: "Perfect Storm Atlantic", code: "BGI", country: "Barbados", lat: 13.0765, lng: -59.4927 },
    { name: "Tornado Alley Center", code: "OKC", country: "USA", lat: 35.3931, lng: -97.6007 },
    { name: "Himalayan Weather Station", code: "KTM", country: "Nepal", lat: 27.6966, lng: 85.3591 },
  ]

  useEffect(() => {
    // Simulate loading airports
    setTimeout(() => {
      const processedAirports = airportData.map((airport, index) => ({
        ...airport,
        id: `airport_${index}_${airport.code}`,
      }))
      setAirports(processedAirports)
      setLoading(false)
    }, 1000)
  }, [])

  useEffect(() => {
    if (typeof window !== "undefined" && !loading && airports.length > 0) {
      // Load Leaflet dynamically
      import("leaflet").then((L) => {
        if (mapRef.current && !mapInstance) {
          // Initialize map
          const map = L.map(mapRef.current, {
            center: [20, 0],
            zoom: 2,
            zoomControl: true,
            scrollWheelZoom: true,
            doubleClickZoom: false,
            boxZoom: false,
            touchZoom: true,
            dragging: true,
            minZoom: 2,
            maxZoom: 18,
          })

          // Add dark tile layer
          L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
            attribution:
              '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
            maxZoom: 19,
          }).addTo(map)

          setMapInstance(map)

          // Add airport markers
          airports.forEach((airport) => {
            const isSelected = selectedAirport?.id === airport.id

            const marker = L.marker([airport.lat, airport.lng], {
              icon: L.divIcon({
                className: `airport-marker ${isSelected ? "selected" : ""}`,
                html: "✈️",
                iconSize: [40, 40],
                iconAnchor: [20, 20],
                popupAnchor: [0, -20],
              }),
            }).addTo(map)

            marker.bindPopup(`
              <div style="text-align: center; color: white;">
                <strong>${airport.name}</strong><br>
                <span style="color: #6366f1;">${airport.code} - ${airport.country}</span><br>
                <small>Lat: ${airport.lat.toFixed(4)}, Lng: ${airport.lng.toFixed(4)}</small>
              </div>
            `)

            marker.on("click", () => {
              onAirportSelect(airport)
              setTimeout(() => marker.closePopup(), 1000)
            })

            markersRef.current[airport.id] = marker
          })
        }
      })
    }
  }, [loading, airports, mapInstance, onAirportSelect, selectedAirport])

  // Update marker styles when selection changes
  useEffect(() => {
    if (mapInstance && airports.length > 0) {
      airports.forEach((airport) => {
        const marker = markersRef.current[airport.id]
        if (marker) {
          const isSelected = selectedAirport?.id === airport.id
          marker.setIcon(
            (window as any).L.divIcon({
              className: `airport-marker ${isSelected ? "selected" : ""}`,
              html: "✈️",
              iconSize: [40, 40],
              iconAnchor: [20, 20],
              popupAnchor: [0, -20],
            }),
          )
        }
      })
    }
  }, [selectedAirport, mapInstance, airports])

  return (
    <Card className="h-full relative overflow-hidden animate-fade-in-scale">
      {/* Airport Count Badge */}
      <div className="absolute top-4 left-4 z-10">
        <Badge variant="secondary" className="bg-card/90 backdrop-blur-sm">
          <MapPin className="h-4 w-4 mr-1" />
          {loading ? "Loading airports..." : `${airports.length} airports loaded`}
        </Badge>
      </div>

      {/* Legend */}
      <div className="absolute bottom-4 left-4 z-10 bg-card/90 backdrop-blur-sm rounded-lg p-3 space-y-2">
        <div className="flex items-center space-x-2">
          <div className="w-5 h-5 bg-red-500 rounded-full flex items-center justify-center text-xs">✈️</div>
          <span className="text-sm text-muted-foreground">Airports</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-5 h-5 bg-green-500 rounded-full flex items-center justify-center text-xs animate-pulse-glow">
            ✈️
          </div>
          <span className="text-sm text-muted-foreground">Selected</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-5 h-5 bg-secondary rounded-full flex items-center justify-center text-xs">✈️</div>
          <span className="text-sm text-muted-foreground">Hovered</span>
        </div>
      </div>

      {/* Loading Overlay */}
      {loading && (
        <div className="absolute inset-0 bg-card/80 backdrop-blur-sm flex items-center justify-center z-20">
          <div className="text-center space-y-4">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-secondary mx-auto"></div>
            <div className="text-muted-foreground">Loading airports worldwide...</div>
          </div>
        </div>
      )}

      {/* Map Container */}
      <div ref={mapRef} className="w-full h-full rounded-lg" style={{ minHeight: "400px" }} />

      {/* Custom Styles for Leaflet */}
      <style jsx global>{`
        .airport-marker {
          background: #ef4444;
          border: 3px solid #fff;
          border-radius: 50%;
          width: 40px;
          height: 40px;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 16px;
          color: white;
          font-weight: bold;
          cursor: pointer;
          transition: all 0.3s ease;
          box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        }

        .airport-marker:hover {
          background: #6366f1 !important;
          box-shadow: 0 0 20px rgba(99, 102, 241, 0.5);
          transform: scale(1.1);
        }

        .airport-marker.selected {
          background: #10b981 !important;
          animation: pulse-glow 2s infinite;
        }

        .leaflet-popup-content-wrapper {
          background: rgba(0, 0, 0, 0.9);
          color: white;
          border-radius: 8px;
          box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        }

        .leaflet-popup-content {
          margin: 10px;
          font-size: 14px;
          text-align: center;
        }

        .leaflet-popup-tip {
          background: rgba(0, 0, 0, 0.9);
        }

        .leaflet-control-zoom a {
          background: rgba(0, 0, 0, 0.7) !important;
          color: white !important;
          border: 1px solid rgba(255, 255, 255, 0.2) !important;
        }

        .leaflet-control-zoom a:hover {
          background: rgba(0, 0, 0, 0.9) !important;
        }
      `}</style>
    </Card>
  )
}
