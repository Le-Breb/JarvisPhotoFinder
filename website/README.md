# Semantic Search - Images & PDFs

A modern, full-featured semantic search application for images and PDFs built with Next.js 14, TypeScript, and Tailwind CSS.

## Features

- **Natural Language Search**: Search for images and PDFs using everyday language
- **Date Range Filtering**: Filter results by custom date ranges
- **Multi-User Authentication**: Login system with role-based access (admin and read-only users)
- **Image Lightbox**: View images in a full-screen lightbox with zoom and navigation
- **PDF Viewer**: Browse PDF documents with page navigation and zoom controls
- **Dark Mode**: Full dark mode support with system preference detection
- **Responsive Design**: Works seamlessly on mobile, tablet, and desktop
- **Keyboard Shortcuts**: Navigate with arrow keys, ESC to close viewers, Enter to search
- **Loading States**: Beautiful skeleton loaders and animations

## Tech Stack

### Frontend
- **Framework**: Next.js 14+ with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **UI Components**: shadcn/ui (Radix UI)
- **Authentication**: NextAuth.js
- **PDF Rendering**: react-pdf
- **Date Handling**: date-fns
- **HTTP Client**: Axios

### Backend
- **Framework**: Flask
- **AI Models**: OpenAI CLIP, InsightFace
- **Vector Search**: FAISS
- **Image Processing**: PyTorch, Pillow, OpenCV

## Getting Started

### Prerequisites

- Node.js 18+ installed
- Python 3.8+ installed
- npm or yarn package manager
- pip (Python package manager)
- CUDA-compatible GPU (optional, for faster processing)

### Installation

#### 1. Clone the repository:
```bash
git clone <repository-url>
cd website
```

#### 2. Setup Frontend

Install Node.js dependencies:
```bash
npm install
```

Create environment file:
```bash
cp .env.example .env.local
```

Update the environment variables in `.env.local` (see Configuration section below)

#### 3. Setup Python Backend

Install Python dependencies:
```bash
cd python
pip install -r requirements.txt
```

#### 4. Index your images

Before you can search, you need to index your images:

```bash
# Index images using CLIP (for text-based search)
python image_index.py

# Index faces (for face recognition search)
python index_faces.py index
```

By default, the scripts look for images in an `images/` folder. You can modify the folder path in the scripts.

#### 5. Run the Application

Start the Python backend (from the `python/` directory):
```bash
python main.py
```

In a new terminal, start the Next.js frontend (from the root directory):
```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser

## Configuration

Create a `.env.local` file with the following variables:

```env
# NextAuth Configuration
NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=your-secret-key-here-generate-with-openssl

# Python Backend API URL
PYTHON_API_URL=http://localhost:5000
```

To generate a secure `NEXTAUTH_SECRET`:
```bash
openssl rand -base64 32
```

## Demo Credentials

The application comes with demo credentials for testing:

- **Admin User**:
  - Email: `admin@example.com`
  - Password: `admin123`
  - Full access to all features

- **Read-Only User**:
  - Email: `user@example.com`
  - Password: `user123`
  - Read-only access

## Project Structure

```
website/
├── app/                    # Next.js app directory
│   ├── api/               # API routes
│   │   ├── auth/         # NextAuth authentication
│   │   ├── search/       # Search endpoint
│   │   └── file/         # File retrieval endpoint
│   ├── login/            # Login page
│   ├── search/           # Search results page
│   ├── page.tsx          # Home page
│   ├── layout.tsx        # Root layout
│   └── globals.css       # Global styles
├── components/            # React components
│   ├── ui/               # shadcn/ui components
│   ├── header.tsx        # App header with user menu
│   ├── search-bar.tsx    # Search input component
│   ├── date-range-picker.tsx  # Date range selector
│   ├── results-grid.tsx  # Search results grid
│   ├── result-card.tsx   # Individual result card
│   ├── image-viewer.tsx  # Image lightbox viewer
│   ├── pdf-viewer.tsx    # PDF document viewer
│   ├── auth-provider.tsx # NextAuth provider
│   └── theme-provider.tsx # Dark mode provider
├── lib/                   # Utility functions
│   └── utils.ts          # Common utilities
├── python/                # Python backend
│   ├── main.py           # Flask API server
│   ├── search.py         # CLIP-based image search
│   ├── index_faces.py    # Face recognition indexing & search
│   ├── image_index.py    # CLIP image indexing
│   └── requirements.txt  # Python dependencies
├── types/                 # TypeScript type definitions
│   └── index.ts
├── public/                # Static assets
└── package.json          # Dependencies and scripts
```

## How It Works

### Search Types

The application supports two types of semantic search:

1. **Text-based Search (CLIP)**:
   - Search for images using natural language descriptions
   - Powered by OpenAI's CLIP model
   - Example queries: "a sunset over the ocean", "a person holding coffee"

2. **Face Recognition Search**:
   - Find all images containing a specific person
   - Powered by InsightFace
   - Provide a reference image of a person to find similar faces

### Architecture

1. **Frontend (Next.js)**: Handles UI, authentication, and routing
2. **Next.js API Routes**: Acts as a proxy to the Python backend
3. **Python Flask Backend**: Performs the actual AI-powered search using CLIP and InsightFace
4. **FAISS Index**: Stores and searches through image embeddings efficiently

## Available Scripts

```bash
# Frontend Development
npm run dev          # Start Next.js development server
npm run build        # Build for production
npm start           # Start production server
npm run lint        # Run ESLint

# Python Backend
cd python
python main.py      # Start Flask API server
python image_index.py  # Index images with CLIP
python index_faces.py index  # Index faces
python index_faces.py search path/to/person.jpg  # Search for a person
```

## Keyboard Shortcuts

- **Enter**: Submit search from search bar
- **Escape**: Close image/PDF viewer
- **Arrow Left**: Navigate to previous result in viewer
- **Arrow Right**: Navigate to next result in viewer

## Features in Detail

### Search Interface

- Centered search bar on home page that animates to the top on search
- Natural language query support
- Date range picker with calendar interface
- Real-time validation

### Results Display

- Responsive grid layout (2-4 columns based on screen size)
- Thumbnail previews for both images and PDFs
- File type indicators
- Date display
- Skeleton loaders during search

### Image Viewer

- Full-screen lightbox display
- Zoom in/out controls
- Download functionality
- Keyboard navigation between results
- Swipe support on mobile

### PDF Viewer

- Page-by-page navigation
- Zoom controls
- Download functionality
- Page counter
- Text selection support

### Authentication

- Secure JWT-based authentication with NextAuth.js
- Role-based access control (admin vs read-only)
- Protected routes
- Automatic session management

### Dark Mode

- System preference detection
- Manual toggle in header
- Persisted user preference
- Smooth transitions

## Customization

### Changing Theme Colors

Edit `tailwind.config.ts` to customize the color scheme:

```typescript
colors: {
  primary: {
    DEFAULT: "hsl(var(--primary))",
    foreground: "hsl(var(--primary-foreground))",
  },
  // ... other colors
}
```

### Adding New UI Components

Use shadcn/ui CLI to add more components:

```bash
npx shadcn-ui@latest add [component-name]
```

## Production Deployment

### Environment Variables

Ensure all production environment variables are set:
- `NEXTAUTH_URL`: Your production domain
- `NEXTAUTH_SECRET`: Secure random string
- `PYTHON_API_URL`: Your Python backend API URL

### Build and Deploy

```bash
npm run build
npm start
```

Or deploy to Vercel with one click:

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new)

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)
- Mobile browsers (iOS Safari, Chrome)

## Performance

- Server-side rendering for optimal performance
- Image optimization with Next.js Image component
- Code splitting and lazy loading
- Responsive images with automatic sizing

## Troubleshooting

### PDF viewer not working

Make sure the PDF.js worker is loaded correctly. Check browser console for errors.

### Authentication issues

Verify that `NEXTAUTH_SECRET` and `NEXTAUTH_URL` are properly set in `.env.local`

### Dark mode not persisting

Clear browser cache and local storage, then try again.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

MIT

## Support

For issues and questions, please open an issue on GitHub.
