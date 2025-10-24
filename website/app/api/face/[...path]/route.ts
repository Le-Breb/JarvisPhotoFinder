import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth/next';
import { authOptions } from '@/lib/auth';
import fs from 'fs';
import path from 'path';
import sharp from 'sharp';

export async function GET(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  try {
    const session = await getServerSession(authOptions);
    
    if (!session) {
      return new NextResponse('Unauthorized', { status: 401 });
    }

    const { searchParams } = new URL(request.url);
    const bbox = searchParams.get('bbox');
    const padding = parseInt(searchParams.get('padding') || '30');

    if (!bbox) {
      return new NextResponse('Bbox parameter required', { status: 400 });
    }

    const [x1, y1, x2, y2] = bbox.split(',').map(parseFloat);
    
    // Construct the image path
    const imagePath = path.join(process.cwd(), 'private', 'images', ...params.path);
    
    if (!fs.existsSync(imagePath)) {
      console.log('❌ Image not found:', imagePath);
      return new NextResponse('Image not found', { status: 404 });
    }

    // Get image metadata to know dimensions
    const metadata = await sharp(imagePath).metadata();
    const imageWidth = metadata.width || 0;
    const imageHeight = metadata.height || 0;

    // Calculate crop dimensions with padding
    const faceWidth = x2 - x1;
    const faceHeight = y2 - y1;
    
    // Calculate crop area with padding, clamped to image boundaries
    let cropX = Math.round(x1 - padding);
    let cropY = Math.round(y1 - padding);
    let cropWidth = Math.round(faceWidth + (padding * 2));
    let cropHeight = Math.round(faceHeight + (padding * 2));

    // Clamp to image boundaries
    cropX = Math.max(0, cropX);
    cropY = Math.max(0, cropY);
    cropWidth = Math.min(cropWidth, imageWidth - cropX);
    cropHeight = Math.min(cropHeight, imageHeight - cropY);

    // Ensure dimensions are valid
    if (cropWidth <= 0 || cropHeight <= 0 || cropX >= imageWidth || cropY >= imageHeight) {
      console.log('❌ Invalid crop dimensions:', { cropX, cropY, cropWidth, cropHeight, imageWidth, imageHeight });
      return new NextResponse('Invalid crop area', { status: 400 });
    }

    // Read and crop the image using sharp
    const imageBuffer = await sharp(imagePath)
      .extract({
        left: cropX,
        top: cropY,
        width: cropWidth,
        height: cropHeight
      })
      .resize(400, 400, { fit: 'cover' })
      .jpeg({ quality: 90 })
      .toBuffer();

    return new NextResponse(imageBuffer as any, {
      headers: {
        'Content-Type': 'image/jpeg',
        'Cache-Control': 'private, max-age=3600',
      },
    });
  } catch (error) {
    console.error('💥 Error serving face image:', error);
    return new NextResponse('Internal Server Error', { status: 500 });
  }
}
