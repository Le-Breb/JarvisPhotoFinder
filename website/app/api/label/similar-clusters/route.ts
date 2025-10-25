import { NextRequest, NextResponse } from "next/server"
import { getServerSession } from "next-auth"
import { authOptions } from "@/lib/auth"

const PYTHON_API_URL = process.env.PYTHON_API_URL || "http://localhost:5000"

export async function GET(request: NextRequest) {
  const session = await getServerSession(authOptions)
  if (!session) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
  }

  try {
    const { searchParams } = new URL(request.url)
    const minSimilarity = searchParams.get("min_similarity") || "0.7"
    const maxPairs = searchParams.get("max_pairs") || "50"

    const response = await fetch(
      `${PYTHON_API_URL}/api/people/similar-clusters?min_similarity=${minSimilarity}&max_pairs=${maxPairs}`
    )

    if (!response.ok) {
      throw new Error(`Python API error: ${response.statusText}`)
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error("Error fetching similar clusters:", error)
    return NextResponse.json(
      { error: "Failed to fetch similar clusters" },
      { status: 500 }
    )
  }
}
