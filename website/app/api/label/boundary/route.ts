import { NextRequest, NextResponse } from "next/server"
import { getServerSession } from "next-auth"
import { authOptions } from "@/lib/auth"
import path from "path"
import fs from "fs/promises"
import { exec } from 'child_process'
import { promisify } from 'util'

const execAsync = promisify(exec)

export async function GET(request: NextRequest) {
  const session = await getServerSession(authOptions)
  if (!session) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
  }

  try {
    const clustersPath = path.join(process.cwd(), "python", "faces", "clusters.json")
    const embeddingsPath = path.join(process.cwd(), "python", "faces", "embeddings.npy")

    // Check if files exist
    try {
      await fs.access(clustersPath)
      await fs.access(embeddingsPath)
    } catch {
      return NextResponse.json({
        faces: [],
        message: "Face data not found. Run face indexing first."
      })
    }

    // Get query parameters for customization
    const { searchParams } = new URL(request.url)
    const distanceThreshold = searchParams.get('threshold') || '0.4'
    const maxResults = searchParams.get('limit') || '50'

    // Execute Python script to find boundary faces
    const pythonDir = path.join(process.cwd(), "python")
    const pythonPath = path.join(pythonDir, ".venv", "bin", "python")
    const scriptPath = path.join(pythonDir, "find_boundary_faces.py")

    console.log('🔍 Finding boundary faces with threshold:', distanceThreshold)
    
    const { stdout, stderr } = await execAsync(
      `cd "${pythonDir}" && "${pythonPath}" "${scriptPath}" ${distanceThreshold} ${maxResults}`,
      { maxBuffer: 10 * 1024 * 1024 } // 10MB buffer
    )

    if (stderr && stderr.trim().length > 0) {
      console.error('Python stderr:', stderr)
    }

    const result = JSON.parse(stdout)
    
    if (result.error) {
      console.error('Python script error:', result.error)
      return NextResponse.json({ 
        error: "Failed to find boundary faces", 
        details: result.error 
      }, { status: 500 })
    }

    console.log(`✅ Found ${result.count} boundary faces`)

    return NextResponse.json({ 
      faces: result.boundary_faces,
      count: result.count
    })

  } catch (error: any) {
    console.error("Error loading boundary faces:", error)
    return NextResponse.json({ 
      error: "Failed to load boundary faces",
      details: error.message 
    }, { status: 500 })
  }
}
