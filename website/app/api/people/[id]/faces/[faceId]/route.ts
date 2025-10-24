import { NextRequest, NextResponse } from "next/server"
import { getServerSession } from "next-auth"
import { authOptions } from "@/lib/auth"
import path from "path"
import fs from "fs/promises"

export async function DELETE(
  request: NextRequest,
  { params }: { params: { id: string; faceId: string } }
) {
  const session = await getServerSession(authOptions)
  if (!session) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
  }

  try {
    const clustersPath = path.join(process.cwd(), "python", "faces", "clusters.json")
    const data = await fs.readFile(clustersPath, "utf-8")
    const clustersData = JSON.parse(data)

    const cluster = clustersData.clusters[params.id]
    if (!cluster) {
      return NextResponse.json({ error: "Person not found" }, { status: 404 })
    }

    // Remove the face from the cluster
    const originalLength = cluster.faces.length
    cluster.faces = cluster.faces.filter((face: any) => face.face_id !== params.faceId)

    if (cluster.faces.length === originalLength) {
      return NextResponse.json({ error: "Face not found" }, { status: 404 })
    }

    // Move to noise if cluster becomes too small
    if (cluster.faces.length < 2) {
      clustersData.noise.push(...cluster.faces)
      delete clustersData.clusters[params.id]
      clustersData.n_clusters = Object.keys(clustersData.clusters).length
      clustersData.n_noise = clustersData.noise.length
    }

    await fs.writeFile(clustersPath, JSON.stringify(clustersData, null, 2))

    return NextResponse.json({ 
      success: true, 
      remainingFaces: cluster.faces.length,
      movedToNoise: cluster.faces.length < 2 
    })
  } catch (error) {
    console.error("Error removing face:", error)
    return NextResponse.json({ error: "Failed to remove face" }, { status: 500 })
  }
}
