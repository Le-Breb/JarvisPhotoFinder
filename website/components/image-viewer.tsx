"use client"

import { SearchResult } from "@/types"
import { Dialog, DialogContent } from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import {
  X,
  ChevronLeft,
  ChevronRight,
  Download,
  ZoomIn,
  ZoomOut,
} from "lucide-react"
import { useState } from "react"
import Image from "next/image"
import { format } from "date-fns"

interface ImageViewerProps {
  result: SearchResult
  onClose: () => void
  onNavigate: (direction: "prev" | "next") => void
}

export function ImageViewer({ result, onClose, onNavigate }: ImageViewerProps) {
  const [zoom, setZoom] = useState(1)

  const handleDownload = () => {
    const link = document.createElement("a")
    link.href = result.filepath
    link.download = result.filename
    link.click()
  }

  return (
    <Dialog open onOpenChange={onClose}>
      <DialogContent className="max-w-[95vw] max-h-[95vh] p-0 overflow-hidden">
        <div className="relative w-full h-[90vh] bg-black">
          {/* Header */}
          <div className="absolute top-0 left-0 right-0 z-10 bg-gradient-to-b from-black/80 to-transparent p-4">
            <div className="flex items-center justify-between text-white">
              <div>
                <h3 className="font-semibold">{result.filename}</h3>
                <p className="text-sm text-gray-300">
                  {format(new Date(result.date), "MMMM d, yyyy")}
                </p>
              </div>
              <div className="flex items-center gap-2">
                <Button
                  variant="ghost"
                  size="icon"
                  className="text-white hover:bg-white/20"
                  onClick={() => setZoom(Math.max(0.5, zoom - 0.25))}
                >
                  <ZoomOut className="h-5 w-5" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  className="text-white hover:bg-white/20"
                  onClick={() => setZoom(Math.min(3, zoom + 0.25))}
                >
                  <ZoomIn className="h-5 w-5" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  className="text-white hover:bg-white/20"
                  onClick={handleDownload}
                >
                  <Download className="h-5 w-5" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  className="text-white hover:bg-white/20"
                  onClick={onClose}
                >
                  <X className="h-5 w-5" />
                </Button>
              </div>
            </div>
          </div>

          {/* Image */}
          <div className="w-full h-full flex items-center justify-center overflow-auto p-4">
            <div
              style={{
                transform: `scale(${zoom})`,
                transition: "transform 0.2s",
              }}
            >
              <Image
                src={result.filepath}
                alt={result.filename}
                width={1200}
                height={800}
                className="max-w-full h-auto"
                unoptimized
              />
            </div>
          </div>

          {/* Navigation */}
          <Button
            variant="ghost"
            size="icon"
            className="absolute left-4 top-1/2 -translate-y-1/2 text-white hover:bg-white/20 h-12 w-12"
            onClick={() => onNavigate("prev")}
          >
            <ChevronLeft className="h-8 w-8" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="absolute right-4 top-1/2 -translate-y-1/2 text-white hover:bg-white/20 h-12 w-12"
            onClick={() => onNavigate("next")}
          >
            <ChevronRight className="h-8 w-8" />
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}
