"use client"

import { useState, useEffect, useCallback } from "react"
import { useRouter, useSearchParams } from "next/navigation"
import { DateRange } from "react-day-picker"
import { Header } from "@/components/header"
import { SearchBar } from "@/components/search-bar"
import { DateRangePicker } from "@/components/date-range-picker"
import { ResultsGrid } from "@/components/results-grid"
import { ImageViewer } from "@/components/image-viewer"
import { PDFViewer } from "@/components/pdf-viewer"
import { SearchResult } from "@/types"
import axios from "axios"

export default function SearchPage() {
  const router = useRouter()
  const searchParams = useSearchParams()

  const [query, setQuery] = useState(searchParams.get("q") || "")
  const [dateRange, setDateRange] = useState<DateRange | undefined>(() => {
    const from = searchParams.get("from")
    const to = searchParams.get("to")
    return {
      from: from ? new Date(from) : undefined,
      to: to ? new Date(to) : undefined,
    }
  })

  const [results, setResults] = useState<SearchResult[]>([])
  const [loading, setLoading] = useState(false)
  const [selectedResult, setSelectedResult] = useState<SearchResult | null>(null)
  const [selectedIndex, setSelectedIndex] = useState(-1)
  const [matchedPerson, setMatchedPerson] = useState<string | null>(null)
  const [searchType, setSearchType] = useState<string | null>(null)

  useEffect(() => {
    if (query) {
      performSearch()
    }
  }, [])

  const performSearch = async () => {
    if (!query.trim()) return

    setLoading(true)
    setMatchedPerson(null)
    setSearchType(null)
    
    try {
      const response = await axios.post("/api/search", {
        query,
        dateFrom: dateRange?.from?.toISOString(),
        dateTo: dateRange?.to?.toISOString(),
      })
      setResults(response.data.results)
      
      // Check if search matched a person name
      if (response.data.matched_person) {
        setMatchedPerson(response.data.matched_person)
        setSearchType(response.data.search_type)
      }
    } catch (error) {
      console.error("Search error:", error)
      setResults([])
    } finally {
      setLoading(false)
    }
  }

  const handleSearch = () => {
    const params = new URLSearchParams()
    params.set("q", query)

    if (dateRange?.from) {
      params.set("from", dateRange.from.toISOString())
    }
    if (dateRange?.to) {
      params.set("to", dateRange.to.toISOString())
    }

    router.push(`/search?${params.toString()}`)
    performSearch()
  }

  const handleResultClick = (result: SearchResult) => {
    const index = results.findIndex((r) => r.id === result.id)
    setSelectedResult(result)
    setSelectedIndex(index)
  }

  const handleClose = () => {
    setSelectedResult(null)
    setSelectedIndex(-1)
  }

  const handleNavigate = (direction: "prev" | "next") => {
    if (selectedIndex === -1) return

    const newIndex =
      direction === "prev"
        ? (selectedIndex - 1 + results.length) % results.length
        : (selectedIndex + 1) % results.length

    setSelectedResult(results[newIndex])
    setSelectedIndex(newIndex)
  }

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!selectedResult) return

      if (e.key === "ArrowLeft") {
        handleNavigate("prev")
      } else if (e.key === "ArrowRight") {
        handleNavigate("next")
      } else if (e.key === "Escape") {
        handleClose()
      }
    }

    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [selectedResult, selectedIndex])

  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      <div className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 sticky top-0 z-10 animate-slide-up">
        <div className="container mx-auto px-4 py-4 space-y-4">
          <div className="flex gap-4">
            <SearchBar
              value={query}
              onChange={setQuery}
              onSubmit={handleSearch}
              compact
              className="flex-1"
            />
          </div>
          <DateRangePicker
            dateRange={dateRange}
            setDateRange={setDateRange}
          />
        </div>
      </div>

      <main className="flex-1 container mx-auto px-4 py-6">
        {matchedPerson && searchType === 'face_by_name' && (
          <div className="mb-4 p-4 bg-primary/10 border border-primary/20 rounded-lg">
            <p className="text-sm font-medium">
              🎯 Found person: <span className="font-bold">{matchedPerson}</span>
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              Showing photos using face recognition similarity
            </p>
          </div>
        )}
        
        {!loading && results.length > 0 && (
          <p className="text-sm text-muted-foreground mb-4">
            Found {results.length} results
          </p>
        )}

        <ResultsGrid
          results={results}
          onResultClick={handleResultClick}
          loading={loading}
        />
      </main>

      {selectedResult && selectedResult.type === "image" && (
        <ImageViewer
          result={selectedResult}
          onClose={handleClose}
          onNavigate={handleNavigate}
        />
      )}

      {selectedResult && selectedResult.type === "pdf" && (
        <PDFViewer
          result={selectedResult}
          onClose={handleClose}
          onNavigate={handleNavigate}
        />
      )}
    </div>
  )
}
