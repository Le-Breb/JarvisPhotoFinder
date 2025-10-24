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
    const { faceId, sourceClusterId, targetClusterId, newPersonName } = await request.json()

    if (!faceId || !sourceClusterId) {
      return NextResponse.json({ error: "Missing required parameters" }, { status: 400 })
    }

    const clustersPath = path.join(process.cwd(), "python", "faces", "clusters.json")
    const data = await fs.readFile(clustersPath, "utf-8")
    const clustersData = JSON.parse(data)

    // Find the face in the source cluster
    const sourceCluster = clustersData.clusters[sourceClusterId]
    if (!sourceCluster) {
      return NextResponse.json({ error: "Source cluster not found" }, { status: 404 })
    }

    const faceIndex = sourceCluster.faces.findIndex((f: any) => f.face_id === faceId)
    if (faceIndex === -1) {
      return NextResponse.json({ error: "Face not found in source cluster" }, { status: 404 })
    }

    const face = sourceCluster.faces[faceIndex]

    // Determine target cluster
    let finalTargetId = targetClusterId

    if (targetClusterId === "NEW" || newPersonName) {
      // Create new cluster
      const maxId = Math.max(...Object.keys(clustersData.clusters).map(id => parseInt(id)))
      finalTargetId = String(maxId + 1)

      clustersData.clusters[finalTargetId] = {
        cluster_id: finalTargetId,
        name: newPersonName || `Person ${finalTargetId}`,
        faces: []
      }
    }

    // Check target cluster exists
    if (!clustersData.clusters[finalTargetId]) {
      return NextResponse.json({ error: "Target cluster not found" }, { status: 404 })
    }

    // Move face to target cluster
    clustersData.clusters[finalTargetId].faces.push(face)

    // Remove face from source cluster
    sourceCluster.faces.splice(faceIndex, 1)

    // If source cluster is now empty, delete it
    if (sourceCluster.faces.length === 0) {
      delete clustersData.clusters[sourceClusterId]
      clustersData.n_clusters = Object.keys(clustersData.clusters).length
    }

    // Save changes
    await fs.writeFile(clustersPath, JSON.stringify(clustersData, null, 2))

    return NextResponse.json({
      success: true,
      targetClusterId: finalTargetId,
      targetClusterName: clustersData.clusters[finalTargetId].name
    })
  } catch (error) {
    console.error("Error reassigning face:", error)
    return NextResponse.json({ error: "Failed to reassign face" }, { status: 500 })
  }
}
