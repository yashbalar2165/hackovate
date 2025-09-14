"use client"

import { AlertTriangle, X } from "lucide-react"
import { useState } from "react"
import { Button } from "@/components/ui/button"

interface AlertBannerProps {
  alerts: Array<{
    time: string
    text: string
  }>
  airportName: string
}

export function AlertBanner({ alerts, airportName }: AlertBannerProps) {
  const [dismissed, setDismissed] = useState(false)

  if (!alerts || alerts.length === 0 || dismissed) {
    return null
  }

  // Find highest priority alert
  const highPriorityAlert =
    alerts.find((alert) => alert.time.includes("HIGH PRIORITY")) ||
    alerts.find((alert) => alert.time.includes("MODERATE")) ||
    alerts[0]

  if (!highPriorityAlert) return null

  const isHighPriority = highPriorityAlert.time.includes("HIGH PRIORITY")
  const isModerate = highPriorityAlert.time.includes("MODERATE")

  return (
    <div
      className={`
      ${isHighPriority ? "bg-destructive" : isModerate ? "bg-yellow-500" : "bg-secondary"}
      text-white px-4 py-3 animate-slide-in-up
    `}
    >
      <div className="container mx-auto flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <AlertTriangle className="h-5 w-5 animate-pulse" />
          <span className="font-semibold">
            {isHighPriority ? "⚡ SEVERE WEATHER ALERT" : isModerate ? "⚠️ WEATHER ADVISORY" : "📢 WEATHER NOTICE"}
          </span>
          <span>
            {airportName && `${airportName} - `}
            {highPriorityAlert.text}
          </span>
        </div>

        <Button variant="ghost" size="sm" onClick={() => setDismissed(true)} className="text-white hover:bg-white/20">
          <X className="h-4 w-4" />
        </Button>
      </div>
    </div>
  )
}
