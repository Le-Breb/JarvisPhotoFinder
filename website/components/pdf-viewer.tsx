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
import { useState, useEffect } from "react"
import { format } from "date-fns"
import { Document, Page, pdfjs } from "react-pdf"
import "react-pdf/dist/Page/AnnotationLayer.css"
import "react-pdf/dist/Page/TextLayer.css"

// Configure PDF.js worker
pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.js`

interface PDFViewerProps {
  result: SearchResult
  onClose: () => void
  onNavigate: (direction: "prev" | "next") => void
}

export function PDFViewer({ result, onClose, onNavigate }: PDFViewerProps) {
  const [numPages, setNumPages] = useState<number>(0)
  const [pageNumber, setPageNumber] = useState<number>(1)
  const [scale, setScale] = useState<number>(1.0)

  function onDocumentLoadSuccess({ numPages }: { numPages: number }) {
    setNumPages(numPages)
    setPageNumber(1)
  }

  const handleDownload = () => {
    const link = document.createElement("a")
    link.href = result.filepath
    link.download = result.filename
    link.click()
  }

  const changePage = (offset: number) => {
    setPageNumber((prevPageNumber) => {
      const newPage = prevPageNumber + offset
      return Math.min(Math.max(1, newPage), numPages)
    })
  }

  return (
    <Dialog open onOpenChange={onClose}>
      <DialogContent className="max-w-[95vw] max-h-[95vh] p-0 overflow-hidden">
        <div className="relative w-full h-[90vh] bg-gray-900">
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
                  onClick={() => setScale(Math.max(0.5, scale - 0.25))}
                >
                  <ZoomOut className="h-5 w-5" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  className="text-white hover:bg-white/20"
                  onClick={() => setScale(Math.min(2, scale + 0.25))}
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

          {/* PDF Document */}
          <div className="w-full h-full flex items-center justify-center overflow-auto pt-16 pb-16">
            <Document
              file={result.filepath}
              onLoadSuccess={onDocumentLoadSuccess}
              loading={
                <div className="text-white">Loading PDF...</div>
              }
              error={
                <div className="text-white">Failed to load PDF</div>
              }
            >
              <Page
                pageNumber={pageNumber}
                scale={scale}
                renderTextLayer={true}
                renderAnnotationLayer={true}
              />
            </Document>
          </div>

          {/* Page Navigation */}
          <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-4">
            <div className="flex items-center justify-center gap-4 text-white">
              <Button
                variant="ghost"
                size="icon"
                className="text-white hover:bg-white/20"
                onClick={() => changePage(-1)}
                disabled={pageNumber <= 1}
              >
                <ChevronLeft className="h-5 w-5" />
              </Button>
              <span className="text-sm">
                Page {pageNumber} of {numPages}
              </span>
              <Button
                variant="ghost"
                size="icon"
                className="text-white hover:bg-white/20"
                onClick={() => changePage(1)}
                disabled={pageNumber >= numPages}
              >
                <ChevronRight className="h-5 w-5" />
              </Button>
            </div>
          </div>

          {/* Document Navigation (previous/next document) */}
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
