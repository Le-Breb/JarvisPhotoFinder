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
    const { clusterIds, name } = await request.json()
    
    if (!clusterIds || !Array.isArray(clusterIds) || clusterIds.length < 2) {
      return NextResponse.json({ 
        error: "Must provide at least 2 cluster IDs to merge" 
      }, { status: 400 })
    }

    if (!name || typeof name !== "string") {
      return NextResponse.json({ error: "Invalid name" }, { status: 400 })
    }

    const clustersPath = path.join(process.cwd(), "python", "faces", "clusters.json")
    const data = await fs.readFile(clustersPath, "utf-8")
    const clustersData = JSON.parse(data)

    // Collect all faces from clusters to merge
    const mergedFaces: any[] = []
    const missingClusters: string[] = []

    for (const cid of clusterIds) {
      if (clustersData.clusters[String(cid)]) {
        mergedFaces.push(...clustersData.clusters[String(cid)].faces)
        delete clustersData.clusters[String(cid)]
      } else {
        missingClusters.push(String(cid))
      }
    }

    if (mergedFaces.length === 0) {
      return NextResponse.json({ 
        error: "No valid clusters found to merge" 
      }, { status: 404 })
    }

    // Create new merged cluster with first cluster ID
    const newId = String(clusterIds[0])
    clustersData.clusters[newId] = {
      cluster_id: newId,
      name: name,
      faces: mergedFaces
    }

    // Update cluster count
    clustersData.n_clusters = Object.keys(clustersData.clusters).length

    await fs.writeFile(clustersPath, JSON.stringify(clustersData, null, 2))

    return NextResponse.json({ 
      success: true, 
      newClusterId: newId,
      totalFaces: mergedFaces.length,
      mergedCount: clusterIds.length - missingClusters.length,
      missingClusters: missingClusters.length > 0 ? missingClusters : undefined
    })
  } catch (error) {
    console.error("Error merging clusters:", error)
    return NextResponse.json({ error: "Failed to merge clusters" }, { status: 500 })
  }
}
