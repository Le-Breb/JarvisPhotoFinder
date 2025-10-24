import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth/next';
import { authOptions } from '@/lib/auth';
import fs from 'fs';
import path from 'path';

export async function GET(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  try {
    const session = await getServerSession(authOptions);
    
    if (!session) {
      console.log('❌ No session found');
      return new NextResponse('Unauthorized', { status: 401 });
    }

    const imagePath = path.join(process.cwd(), 'private', 'images', ...params.path);
    console.log('🔍 Looking for image at:', imagePath);
    console.log('📁 File exists:', fs.existsSync(imagePath));
    console.log('📂 Params path:', params.path);
    
    if (!fs.existsSync(imagePath)) {
      console.log('❌ Image not found:', imagePath);
      return new NextResponse('Image not found', { status: 404 });
    }

    const imageBuffer = fs.readFileSync(imagePath);
    console.log('✅ Image loaded, size:', imageBuffer.length, 'bytes');
    
    const ext = path.extname(imagePath).toLowerCase();
    const contentType = {
      '.jpg': 'image/jpeg',
      '.jpeg': 'image/jpeg',
      '.png': 'image/png',
      '.gif': 'image/gif',
      '.webp': 'image/webp',
      '.svg': 'image/svg+xml',
    }[ext] || 'application/octet-stream';

    return new NextResponse(imageBuffer, {
      headers: {
        'Content-Type': contentType,
        'Cache-Control': 'private, max-age=3600',
      },
    });
  } catch (error) {
    console.error('💥 Error serving image:', error);
    return new NextResponse('Internal Server Error', { status: 500 });
  }
}