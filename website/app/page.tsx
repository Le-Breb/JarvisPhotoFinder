"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { DateRange } from "react-day-picker"
import { Header } from "@/components/header"
import { SearchBar } from "@/components/search-bar"
import { DateRangePicker } from "@/components/date-range-picker"
import { Button } from "@/components/ui/button"

export default function HomePage() {
  const router = useRouter()
  const [query, setQuery] = useState("")
  const [dateRange, setDateRange] = useState<DateRange | undefined>()

  const handleSearch = () => {
    if (!query.trim()) return

    const params = new URLSearchParams()
    params.set("q", query)

    if (dateRange?.from) {
      params.set("from", dateRange.from.toISOString())
    }
    if (dateRange?.to) {
      params.set("to", dateRange.to.toISOString())
    }

    router.push(`/search?${params.toString()}`)
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <main className="flex-1 flex items-center justify-center p-4">
        <div className="w-full max-w-3xl space-y-6 animate-fade-in">
          <div className="text-center space-y-2 mb-12">
            <h1 className="text-4xl md:text-5xl font-bold tracking-tight">
              Find your files instantly
            </h1>
            <p className="text-lg text-muted-foreground">
              Search through images and PDFs using natural language
            </p>
          </div>

          <div className="space-y-4">
            <SearchBar
              value={query}
              onChange={setQuery}
              onSubmit={handleSearch}
            />

            <div className="flex flex-col sm:flex-row gap-4 items-stretch sm:items-center">
              <DateRangePicker
                dateRange={dateRange}
                setDateRange={setDateRange}
                className="flex-1"
              />

              <Button
                onClick={handleSearch}
                disabled={!query.trim()}
                size="lg"
                className="sm:w-32"
              >
                Search
              </Button>
            </div>
          </div>

          <div className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
            <div className="p-6 rounded-lg border bg-card">
              <h3 className="font-semibold mb-2">Natural Language</h3>
              <p className="text-sm text-muted-foreground">
                Search using everyday language, no complex queries needed
              </p>
            </div>
            <div className="p-6 rounded-lg border bg-card">
              <h3 className="font-semibold mb-2">Smart Filtering</h3>
              <p className="text-sm text-muted-foreground">
                Filter results by date range to find exactly what you need
              </p>
            </div>
            <div className="p-6 rounded-lg border bg-card">
              <h3 className="font-semibold mb-2">Multiple Formats</h3>
              <p className="text-sm text-muted-foreground">
                Works with both images and PDF documents seamlessly
              </p>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
