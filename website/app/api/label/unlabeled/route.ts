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

    try {
      await fs.access(clustersPath)
    } catch {
      return NextResponse.json({
        clusters: [],
        message: "No face clusters found. Run face indexing first."
      })
    }

    const data = await fs.readFile(clustersPath, "utf-8")
    const clustersData = JSON.parse(data)

    // Find unlabeled clusters with > 5 faces
    const unlabeledClusters = Object.entries(clustersData.clusters)
      .map(([id, cluster]: [string, any]) => {
        const name = cluster.name || ''
        const faces = cluster.faces || []

        // Check if unlabeled (default "Person X" pattern or empty)
        const isUnlabeled = (
          !name ||
          name.trim() === '' ||
          name.startsWith('Person ')
        )

        if (isUnlabeled && faces.length > 5) {
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
        }
        return null
      })
      .filter(Boolean)
      .sort((a, b) => (b?.faceCount || 0) - (a?.faceCount || 0)) // Sort by face count desc

    return NextResponse.json({ clusters: unlabeledClusters })
  } catch (error) {
    console.error("Error loading unlabeled clusters:", error)
    return NextResponse.json({ error: "Failed to load unlabeled clusters" }, { status: 500 })
  }
}
