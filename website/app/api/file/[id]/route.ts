import { NextRequest, NextResponse } from "next/server"
import { readFile } from "fs/promises"
import { stat } from "fs/promises"
import path from "path"

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const { id } = params
    const searchParams = request.nextUrl.searchParams
    const filepath = searchParams.get("filepath")

    if (!filepath) {
      return NextResponse.json(
        { error: "File path not provided" },
        { status: 400 }
      )
    }

    // Security check: prevent directory traversal
    const normalizedPath = path.normalize(filepath)
    if (normalizedPath.includes("..")) {
      return NextResponse.json(
        { error: "Invalid file path" },
        { status: 400 }
      )
    }

    // Check if file exists
    try {
      await stat(filepath)
    } catch {
      return NextResponse.json(
        { error: "File not found" },
        { status: 404 }
      )
    }

    // Read and return the file
    const fileBuffer = await readFile(filepath)
    const ext = path.extname(filepath).toLowerCase()

    // Determine content type
    let contentType = "application/octet-stream"
    if ([".jpg", ".jpeg"].includes(ext)) contentType = "image/jpeg"
    else if (ext === ".png") contentType = "image/png"
    else if (ext === ".webp") contentType = "image/webp"
    else if (ext === ".pdf") contentType = "application/pdf"

    return new NextResponse(fileBuffer, {
      headers: {
        "Content-Type": contentType,
        "Cache-Control": "public, max-age=31536000",
      },
    })
  } catch (error) {
    console.error("Failed to retrieve file:", error)
    return NextResponse.json(
      { error: "Failed to retrieve file" },
      { status: 500 }
    )
  }
}
