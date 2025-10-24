import { NextRequest, NextResponse } from "next/server"

const PYTHON_API_URL = process.env.PYTHON_API_URL || "http://localhost:5000"

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { query, dateFrom, dateTo, type } = body

    // Call Python backend
    const response = await fetch(`${PYTHON_API_URL}/api/search`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query,
        dateFrom,
        dateTo,
        type: type || "text", // 'text' for CLIP search, 'face' for face search
      }),
    })

    if (!response.ok) {
      throw new Error(`Python API error: ${response.statusText}`)
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error("Search failed:", error)
    return NextResponse.json(
      { error: "Search failed", results: [] },
      { status: 500 }
    )
  }
}
