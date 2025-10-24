"use client"

import { useState } from "react"
import Image from "next/image"

interface FaceImageProps {
  filename: string
  bbox: number[] // [x1, y1, x2, y2]
  alt?: string
  className?: string
  padding?: number
}

export function FaceImage({ filename, bbox, alt = "Face", className = "", padding = 30 }: FaceImageProps) {
  const [imageError, setImageError] = useState(false)

  if (imageError) {
    return (
      <div className={`flex items-center justify-center bg-muted ${className}`}>
        <p className="text-muted-foreground text-sm">Failed to load image</p>
      </div>
    )
  }

  // Remove "images/" prefix if present
  const cleanFilename = filename.startsWith('images/') ? filename.substring(7) : filename

  // Build the face API URL with bbox and padding parameters
  const bboxParam = bbox.join(',')
  const faceImageUrl = `/api/face/${cleanFilename}?bbox=${bboxParam}&padding=${padding}`

  return (
    <div className={`relative ${className}`}>
      <Image
        src={faceImageUrl}
        alt={alt}
        fill
        className="object-contain"
        onError={() => setImageError(true)}
        unoptimized
      />
    </div>
  )
}
