"use client"

import { useState, useEffect } from "react"
import { Cloud, Zap } from "lucide-react"

export function Header() {
  const [currentTime, setCurrentTime] = useState<string>("")

  useEffect(() => {
    const updateTime = () => {
      const now = new Date()
      setCurrentTime(now.toLocaleDateString() + " " + now.toLocaleTimeString())
    }

    updateTime()
    const interval = setInterval(updateTime, 1000)

    return () => clearInterval(interval)
  }, [])

  return (
    <header className="bg-card/80 backdrop-blur-lg border-b border-border sticky top-0 z-50">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="relative">
              <Zap className="h-8 w-8 text-secondary animate-pulse-glow" />
              <Cloud className="h-4 w-4 text-muted-foreground absolute -top-1 -right-1 animate-float" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-foreground">AirGuard Weather System</h1>
              <p className="text-sm text-muted-foreground">Professional Aviation Weather Monitoring</p>
            </div>
          </div>

          <div className="text-right">
            <div className="text-lg font-mono text-foreground">{currentTime || "Loading..."}</div>
            <div className="text-sm text-muted-foreground">Real-time Updates</div>
          </div>
        </div>
      </div>
    </header>
  )
}
