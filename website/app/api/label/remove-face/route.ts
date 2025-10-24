import { NextRequest, NextResponse } from "next/server"
import { getServerSession } from "next-auth"
import { authOptions } from "@/lib/auth"
import path from "path"
import fs from "fs/promises"

export async function POST(request: NextRequest) {
  const session = await getServerSession(authOptions)
  if (!session) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
  }

  try {
    const { faceId, clusterId } = await request.json()

    if (!faceId || !clusterId) {
      return NextResponse.json({ error: "Missing required parameters" }, { status: 400 })
    }

    const clustersPath = path.join(process.cwd(), "python", "faces", "clusters.json")
    const data = await fs.readFile(clustersPath, "utf-8")
    const clustersData = JSON.parse(data)

    // Find the cluster
    const cluster = clustersData.clusters[clusterId]
    if (!cluster) {
      return NextResponse.json({ error: "Cluster not found" }, { status: 404 })
    }

    // Find and remove the face
    const faceIndex = cluster.faces.findIndex((f: any) => f.face_id === faceId)
    if (faceIndex === -1) {
      return NextResponse.json({ error: "Face not found in cluster" }, { status: 404 })
    }

    // Remove the face
    cluster.faces.splice(faceIndex, 1)

    // If cluster is now empty or has too few faces, delete it
    if (cluster.faces.length < 2) {
      delete clustersData.clusters[clusterId]
      clustersData.n_clusters = Object.keys(clustersData.clusters).length
    }

    // Save changes
    await fs.writeFile(clustersPath, JSON.stringify(clustersData, null, 2))

    console.log(`✅ Removed false positive face ${faceId} from cluster ${clusterId}`)

    return NextResponse.json({
      success: true,
      removed: true,
      clusterDeleted: cluster.faces.length < 2
    })
  } catch (error) {
    console.error("Error removing face:", error)
    return NextResponse.json({ error: "Failed to remove face" }, { status: 500 })
  }
}
