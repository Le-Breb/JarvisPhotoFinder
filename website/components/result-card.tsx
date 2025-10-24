"use client"

import { SearchResult } from "@/types"
import { Card } from "@/components/ui/card"
import { FileText, Image as ImageIcon } from "lucide-react"
import { format } from "date-fns"

interface ResultCardProps {
  result: SearchResult
  onClick: () => void
}

export function ResultCard({ result, onClick }: ResultCardProps) {
  const isImage = result.type === "image"

  return (
    <Card
      className="group cursor-pointer overflow-hidden transition-all hover:shadow-lg hover:scale-[1.02] animate-fade-in"
      onClick={onClick}
    >
      <div className="aspect-square relative bg-muted">
        {isImage ? (
          <img
            src={result.thumbnail}
            alt={result.filename}
            className="w-full h-full object-cover transition-transform group-hover:scale-105"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-orange-100 to-red-100 dark:from-orange-900 dark:to-red-900">
            <FileText className="w-16 h-16 text-orange-600 dark:text-orange-400" />
          </div>
        )}
        <div className="absolute top-2 right-2 bg-background/80 backdrop-blur-sm rounded-full p-1.5">
          {isImage ? (
            <ImageIcon className="w-4 h-4" />
          ) : (
            <FileText className="w-4 h-4" />
          )}
        </div>
      </div>
      <div className="p-3">
        <p className="font-medium text-sm truncate" title={result.filename}>
          {result.filename}
        </p>
        <p className="text-xs text-muted-foreground mt-1">
          {format(new Date(result.date), "MMM d, yyyy")}
        </p>
      </div>
    </Card>
  )
}