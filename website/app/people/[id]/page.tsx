"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { Header } from "@/components/header"
import { Person, Face, SearchResult } from "@/types"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Skeleton } from "@/components/ui/skeleton"
import { ImageViewer } from "@/components/image-viewer"
import { ArrowLeft, Pencil, Save, X, Trash2 } from "lucide-react"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import axios from "axios"

export default function PersonDetailPage({ params }: { params: { id: string } }) {
  const router = useRouter()
  const [person, setPerson] = useState<Person | null>(null)
  const [loading, setLoading] = useState(true)
  const [editing, setEditing] = useState(false)
  const [newName, setNewName] = useState("")
  const [faceToRemove, setFaceToRemove] = useState<Face | null>(null)
  const [removingFaceId, setRemovingFaceId] = useState<string | null>(null)
  const [selectedResult, setSelectedResult] = useState<SearchResult | null>(null)
  const [selectedIndex, setSelectedIndex] = useState(-1)
  const [searchResults, setSearchResults] = useState<SearchResult[]>([])
  const [showDeleteDialog, setShowDeleteDialog] = useState(false)
  const [deleting, setDeleting] = useState(false)

  useEffect(() => {
    loadPerson()
  }, [params.id])

  useEffect(() => {
    // Convert faces to SearchResult format for the image viewer
    if (person) {
      const results: SearchResult[] = person.faces.map((face, index) => {
        const cleanPath = face.filename.replace(/^images\//, "")
        const apiPath = `/api/images/${cleanPath}`
        return {
          id: face.face_id,
          filename: cleanPath,
          filepath: apiPath,
          type: 'image' as const,
          thumbnail: apiPath,
          date: new Date().toISOString(), // You might want to get actual date from metadata
        }
      })
      setSearchResults(results)
    }
  }, [person])

  const loadPerson = async () => {
    setLoading(true)
    try {
      const response = await axios.get(`/api/people/${params.id}`)
      setPerson(response.data.person)
      setNewName(response.data.person.name)
    } catch (error) {
      console.error("Error loading person:", error)
    } finally {
      setLoading(false)
    }
  }

  const handleSaveName = async () => {
    if (!newName.trim() || !person) return

    try {
      await axios.put(`/api/people/${params.id}`, { name: newName })
      setPerson({ ...person, name: newName })
      setEditing(false)
    } catch (error) {
      console.error("Error updating name:", error)
      alert("Failed to update name")
    }
  }

  const handleRemoveFaceClick = (face: Face) => {
    setFaceToRemove(face)
  }

  const confirmRemoveFace = async () => {
    if (!faceToRemove) return

    setRemovingFaceId(faceToRemove.face_id)
    try {
      const response = await axios.delete(`/api/people/${params.id}/faces/${faceToRemove.face_id}`)
      
      if (response.data.movedToNoise) {
        // Cluster was removed, go back to people page
        alert("This person had too few photos remaining and was removed.")
        router.push("/people")
      } else {
        // Reload person data
        await loadPerson()
      }
    } catch (error) {
      console.error("Error removing face:", error)
      alert("Failed to remove face")
    } finally {
      setRemovingFaceId(null)
      setFaceToRemove(null)
    }
  }

  const handleDeletePerson = () => {
    setShowDeleteDialog(true)
  }

  const confirmDeletePerson = async () => {
    if (!person) return

    setDeleting(true)
    try {
      await axios.delete(`/api/people/${params.id}`)
      router.push("/people")
    } catch (error) {
      console.error("Error deleting person:", error)
      alert("Failed to delete person")
      setDeleting(false)
    }
  }

  const handleImageClick = (face: Face) => {
    const index = person?.faces.findIndex((f) => f.face_id === face.face_id) ?? -1
    const result = searchResults[index]
    if (result) {
      setSelectedResult(result)
      setSelectedIndex(index)
    }
  }

  const handleCloseViewer = () => {
    setSelectedResult(null)
    setSelectedIndex(-1)
  }

  const handleNavigate = (direction: "prev" | "next") => {
    if (selectedIndex === -1) return

    const newIndex =
      direction === "prev"
        ? (selectedIndex - 1 + searchResults.length) % searchResults.length
        : (selectedIndex + 1) % searchResults.length

    setSelectedResult(searchResults[newIndex])
    setSelectedIndex(newIndex)
  }

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!selectedResult) return

      if (e.key === "ArrowLeft") {
        handleNavigate("prev")
      } else if (e.key === "ArrowRight") {
        handleNavigate("next")
      } else if (e.key === "Escape") {
        handleCloseViewer()
      }
    }

    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [selectedResult, selectedIndex])

  const getImageUrl = (filename: string) => {
    const cleanPath = filename.replace(/^images\//, "")
    return `/api/images/${cleanPath}`
  }

  if (loading) {
    return (
      <div className="min-h-screen flex flex-col">
        <Header />
        <main className="flex-1 container mx-auto px-4 py-6">
          <Skeleton className="h-10 w-64 mb-6" />
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
            {Array.from({ length: 10 }).map((_, i) => (
              <Skeleton key={i} className="aspect-square w-full" />
            ))}
          </div>
        </main>
      </div>
    )
  }

  if (!person) {
    return (
      <div className="min-h-screen flex flex-col">
        <Header />
        <main className="flex-1 container mx-auto px-4 py-6">
          <div className="text-center py-12">
            <p className="text-lg text-muted-foreground">Person not found</p>
            <Button onClick={() => router.push("/people")} className="mt-4">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to People
            </Button>
          </div>
        </main>
      </div>
    )
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      <main className="flex-1 container mx-auto px-4 py-6">
        <Button
          variant="ghost"
          onClick={() => router.push("/people")}
          className="mb-4"
        >
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to People
        </Button>

        <div className="mb-6 flex items-center justify-between">
          {editing ? (
            <div className="flex-1 flex items-center gap-2">
              <Input
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="Enter person name"
                className="max-w-md"
                onKeyDown={(e) => e.key === "Enter" && handleSaveName()}
              />
              <Button onClick={handleSaveName} size="sm">
                <Save className="h-4 w-4 mr-2" />
                Save
              </Button>
              <Button
                onClick={() => {
                  setEditing(false)
                  setNewName(person.name)
                }}
                size="sm"
                variant="ghost"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          ) : (
            <>
              <div className="flex items-center gap-4">
                <h1 className="text-3xl font-bold">{person.name}</h1>
                <Button onClick={() => setEditing(true)} size="sm" variant="outline">
                  <Pencil className="h-4 w-4 mr-2" />
                  Rename
                </Button>
              </div>
              <Button 
                onClick={handleDeletePerson} 
                size="sm" 
                variant="destructive"
                disabled={deleting}
              >
                <Trash2 className="h-4 w-4 mr-2" />
                Delete Person
              </Button>
            </>
          )}
        </div>

        <p className="text-sm text-muted-foreground mb-6">
          {person.faces.length} {person.faces.length === 1 ? "photo" : "photos"}
        </p>

        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
          {person.faces.map((face) => (
            <Card key={face.face_id} className="overflow-hidden group relative">
              <div 
                className="aspect-square relative overflow-hidden bg-muted cursor-pointer"
                onClick={() => handleImageClick(face)}
              >
                <img
                  src={getImageUrl(face.filename)}
                  alt={`${person.name} - ${face.face_id}`}
                  className="w-full h-full object-cover transition-transform group-hover:scale-105"
                />
                <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
                  <Button
                    variant="destructive"
                    size="sm"
                    onClick={(e) => {
                      e.stopPropagation()
                      handleRemoveFaceClick(face)
                    }}
                    disabled={removingFaceId === face.face_id}
                  >
                    {removingFaceId === face.face_id ? (
                      <span className="animate-spin">⏳</span>
                    ) : (
                      <Trash2 className="h-4 w-4" />
                    )}
                  </Button>
                </div>
              </div>
            </Card>
          ))}
        </div>
      </main>

      {/* Remove Face Confirmation Dialog */}
      <Dialog open={!!faceToRemove} onOpenChange={(open) => !open && setFaceToRemove(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Remove this photo?</DialogTitle>
            <DialogDescription>
              Are you sure this person is incorrectly identified in this photo? This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setFaceToRemove(null)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={confirmRemoveFace}>
              Remove
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Person Confirmation Dialog */}
      <Dialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete {person?.name}?</DialogTitle>
            <DialogDescription>
              This will remove this person cluster and move all {person?.faces.length} photos to unclustered faces. 
              This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowDeleteDialog(false)} disabled={deleting}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={confirmDeletePerson} disabled={deleting}>
              {deleting ? "Deleting..." : "Delete Person"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Image Viewer */}
      {selectedResult && (
        <ImageViewer
          result={selectedResult}
          onClose={handleCloseViewer}
          onNavigate={handleNavigate}
        />
      )}
    </div>
  )
}
