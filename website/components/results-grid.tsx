"use client"

import { SearchResult } from "@/types"
import { ResultCard } from "@/components/result-card"
import { Skeleton } from "@/components/ui/skeleton"

interface ResultsGridProps {
  results: SearchResult[]
  onResultClick: (result: SearchResult) => void
  loading?: boolean
}

export function ResultsGrid({ results, onResultClick, loading }: ResultsGridProps) {
  if (loading) {
    return (
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {Array.from({ length: 8 }).map((_, i) => (
          <div key={i} className="space-y-2">
            <Skeleton className="aspect-square w-full" />
            <Skeleton className="h-4 w-3/4" />
            <Skeleton className="h-3 w-1/2" />
          </div>
        ))}
      </div>
    )
  }

  if (results.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-center">
        <div className="text-6xl mb-4">🔍</div>
        <h3 className="text-lg font-semibold mb-2">No results found</h3>
        <p className="text-muted-foreground max-w-md">
          Try adjusting your search query or date range to find what you&apos;re looking for.
        </p>
      </div>
    )
  }

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
      {results.map((result) => (
        <ResultCard
          key={result.id}
          result={result}
          onClick={() => onResultClick(result)}
        />
      ))}
    </div>
  )
}
