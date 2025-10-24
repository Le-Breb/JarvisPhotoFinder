export interface SearchResult {
  id: string
  filename: string
  filepath: string
  type: 'image' | 'pdf'
  thumbnail: string
  date: string
  score?: number
}

export interface SearchParams {
  query: string
  dateFrom?: Date | null
  dateTo?: Date | null
}

export interface User {
  id: string
  email: string
  name: string
  role: 'admin' | 'read-only'
}

export interface Face {
  face_id: string
  filename: string
  bbox: number[]
  embedding_idx: number
}

export interface Person {
  id: string
  cluster_id: string
  name: string
  faces: Face[]
}

export interface PersonSummary {
  id: string
  name: string
  faceCount: number
  representative: Face | null
  representatives: Face[] // Up to 4 sample faces
}

