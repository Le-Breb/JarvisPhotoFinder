import { NextRequest, NextResponse } from "next/server"
import { getServerSession } from "next-auth"
import { authOptions } from "@/lib/auth"
import path from "path"
import fs from "fs/promises"

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
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

    return NextResponse.json({ person: cluster })
  } catch (error) {
    console.error("Error loading person:", error)
    return NextResponse.json({ error: "Failed to load person" }, { status: 500 })
  }
}

export async function PUT(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  const session = await getServerSession(authOptions)
  if (!session) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
  }

  try {
    const { name } = await request.json()
    
    if (!name || typeof name !== "string") {
      return NextResponse.json({ error: "Invalid name" }, { status: 400 })
    }

    const clustersPath = path.join(process.cwd(), "python", "faces", "clusters.json")
    const data = await fs.readFile(clustersPath, "utf-8")
    const clustersData = JSON.parse(data)

    if (!clustersData.clusters[params.id]) {
      return NextResponse.json({ error: "Person not found" }, { status: 404 })
    }

    clustersData.clusters[params.id].name = name
    await fs.writeFile(clustersPath, JSON.stringify(clustersData, null, 2))

    return NextResponse.json({ success: true, name })
  } catch (error) {
    console.error("Error updating person name:", error)
    return NextResponse.json({ error: "Failed to update name" }, { status: 500 })
  }
}

export async function DELETE(
  request: NextRequest,
  { params }: { params: { id: string } }
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

    // Move all faces to noise
    clustersData.noise.push(...cluster.faces)
    
    // Delete the cluster
    delete clustersData.clusters[params.id]
    
    // Update counts
    clustersData.n_clusters = Object.keys(clustersData.clusters).length
    clustersData.n_noise = clustersData.noise.length

    await fs.writeFile(clustersPath, JSON.stringify(clustersData, null, 2))

    return NextResponse.json({ 
      success: true, 
      facesMovedToNoise: cluster.faces.length 
    })
  } catch (error) {
    console.error("Error deleting person:", error)
    return NextResponse.json({ error: "Failed to delete person" }, { status: 500 })
  }
}
