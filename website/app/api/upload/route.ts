import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth/next';
import { authOptions } from '@/lib/auth';
import fs from 'fs';
import path from 'path';
import { writeFile } from 'fs/promises';

export async function POST(request: NextRequest) {
  try {
    const session = await getServerSession(authOptions);

    if (!session) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const formData = await request.formData();
    const files = formData.getAll('files') as File[];

    if (files.length === 0) {
      return NextResponse.json({ error: 'No files provided' }, { status: 400 });
    }

    // Define upload directory
    const uploadDir = path.join(process.cwd(), 'private', 'images');

    // Ensure directory exists
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }

    const uploadedFiles: string[] = [];
    const errors: string[] = [];

    for (const file of files) {
      try {
        // Validate file type
        if (!file.type.startsWith('image/')) {
          errors.push(`${file.name}: Not an image file`);
          continue;
        }

        // Validate file size (max 50MB)
        const maxSize = 50 * 1024 * 1024;
        if (file.size > maxSize) {
          errors.push(`${file.name}: File too large (max 50MB)`);
          continue;
        }

        // Generate unique filename if file already exists
        let filename = file.name;
        let filePath = path.join(uploadDir, filename);
        let counter = 1;

        while (fs.existsSync(filePath)) {
          const ext = path.extname(filename);
          const base = path.basename(filename, ext);
          filename = `${base}_${counter}${ext}`;
          filePath = path.join(uploadDir, filename);
          counter++;
        }

        // Convert file to buffer and save
        const bytes = await file.arrayBuffer();
        const buffer = Buffer.from(bytes);
        await writeFile(filePath, buffer);

        uploadedFiles.push(filename);
        console.log(`✅ Uploaded: ${filename}`);
      } catch (error) {
        console.error(`❌ Error uploading ${file.name}:`, error);
        errors.push(`${file.name}: Upload failed`);
      }
    }

    // Trigger indexing via Flask API if files were uploaded
    if (uploadedFiles.length > 0) {
      try {
        console.log('🔄 Triggering indexing for:', uploadedFiles);

        // Call Flask API to trigger background indexing
        const response = await fetch('http://localhost:5000/api/index/trigger', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
        });

        if (response.ok) {
          console.log('✅ Indexing triggered successfully');
        } else {
          console.error('❌ Failed to trigger indexing:', response.statusText);
        }
      } catch (e) {
        console.error('❌ Error calling indexing API:', e);
      }
    }

    return NextResponse.json({
      uploaded: uploadedFiles.length,
      files: uploadedFiles,
      errors: errors.length > 0 ? errors : undefined,
    });
  } catch (error) {
    console.error('💥 Upload error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

// Increase payload size limit for file uploads
export const config = {
  api: {
    bodyParser: false,
  },
};

