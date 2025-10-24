import { NextRequest, NextResponse } from "next/server"
import { getServerSession } from "next-auth"
import { authOptions } from "@/lib/auth"
import path from "path"
import fs from "fs/promises"

export async function GET(request: NextRequest) {
  const session = await getServerSession(authOptions)
  if (!session) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
  }

  try {
    const clustersPath = path.join(process.cwd(), "python", "faces", "clusters.json")
    
    // Check if clusters file exists
    try {
      await fs.access(clustersPath)
    } catch {
      return NextResponse.json({ 
        people: [],
        message: "No face clusters found. Run face indexing first." 
      })
    }

    const data = await fs.readFile(clustersPath, "utf-8")
    const clustersData = JSON.parse(data)

    const people = Object.entries(clustersData.clusters).map(([id, cluster]: [string, any]) => {
      const faces = cluster.faces || []
      
      // Get up to 4 representative faces, evenly distributed
      const representatives: any[] = []
      if (faces.length > 0) {
        if (faces.length <= 4) {
          representatives.push(...faces)
        } else {
          // Evenly distribute 4 samples across the face array
          const step = Math.floor(faces.length / 4)
          for (let i = 0; i < 4; i++) {
            representatives.push(faces[i * step])
          }
        }
      }
      
      return {
        id,
        name: cluster.name,
        faceCount: faces.length,
        representative: faces[0] || null,
        representatives,
      }
    })

    // Sort by faceCount in descending order (most photos first)
    people.sort((a, b) => b.faceCount - a.faceCount)

    return NextResponse.json({ people })
  } catch (error) {
    console.error("Error loading people:", error)
    return NextResponse.json({ error: "Failed to load people" }, { status: 500 })
  }
}
