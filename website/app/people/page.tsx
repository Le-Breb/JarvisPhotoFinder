"use client"

import { useState, useEffect } from "react"
import { Header } from "@/components/header"
import { PersonSummary } from "@/types"
import { Card, CardContent } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Pencil, Check, X, Users, Trash2 } from "lucide-react"
import { FaceImage } from "@/components/face-image"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Label } from "@/components/ui/label"
import Link from "next/link"
import axios from "axios"

export default function PeoplePage() {
  const [people, setPeople] = useState<PersonSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editingName, setEditingName] = useState("")
  const [selectionMode, setSelectionMode] = useState(false)
  const [selectedPeople, setSelectedPeople] = useState<Set<string>>(new Set())
  const [showMergeDialog, setShowMergeDialog] = useState(false)
  const [mergeName, setMergeName] = useState("")
  const [merging, setMerging] = useState(false)

  useEffect(() => {
    loadPeople()
  }, [])

  const loadPeople = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await axios.get("/api/people")
      setPeople(response.data.people)
    } catch (error) {
      console.error("Error loading people:", error)
      setError("Failed to load people. Make sure face indexing has been run.")
    } finally {
      setLoading(false)
    }
  }

  const handleStartEdit = (person: PersonSummary, e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setEditingId(person.id)
    setEditingName(person.name)
  }

  const handleSaveName = async (personId: string, e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    
    if (!editingName.trim()) {
      setEditingId(null)
      return
    }

    try {
      await axios.put(`/api/people/${personId}`, { name: editingName })
      setPeople(people.map(p => 
        p.id === personId ? { ...p, name: editingName } : p
      ))
      setEditingId(null)
    } catch (error) {
      console.error("Error updating name:", error)
      alert("Failed to update name")
    }
  }

  const handleCancelEdit = (e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setEditingId(null)
    setEditingName("")
  }

  const toggleSelectionMode = () => {
    setSelectionMode(!selectionMode)
    setSelectedPeople(new Set())
  }

  const togglePersonSelection = (personId: string) => {
    const newSelected = new Set(selectedPeople)
    if (newSelected.has(personId)) {
      newSelected.delete(personId)
    } else {
      newSelected.add(personId)
    }
    setSelectedPeople(newSelected)
  }

  const handleMergeClick = () => {
    if (selectedPeople.size < 2) {
      alert("Please select at least 2 people to merge")
      return
    }
    
    // Set default name from first selected person
    const firstId = Array.from(selectedPeople)[0]
    const firstPerson = people.find(p => p.id === firstId)
    setMergeName(firstPerson?.name || "Merged Person")
    setShowMergeDialog(true)
  }

  const confirmMerge = async () => {
    if (!mergeName.trim() || selectedPeople.size < 2) return

    setMerging(true)
    try {
      const response = await axios.post("/api/people/merge", {
        clusterIds: Array.from(selectedPeople),
        name: mergeName
      })

      // Reload people list
      await loadPeople()
      
      // Reset state
      setShowMergeDialog(false)
      setSelectionMode(false)
      setSelectedPeople(new Set())
      setMergeName("")
    } catch (error) {
      console.error("Error merging people:", error)
      alert("Failed to merge people")
    } finally {
      setMerging(false)
    }
  }

  const getImageUrl = (filename: string) => {
    // Remove 'images/' prefix if present and construct URL
    const cleanPath = filename.replace(/^images\//, "")
    return `/api/images/${cleanPath}`
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      <main className="flex-1 container mx-auto px-4 py-6">
        <div className="mb-6 flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold mb-2">People</h1>
            <p className="text-muted-foreground">
              Browse and manage identified people from your photos
            </p>
          </div>
          <div className="flex gap-2">
            {selectionMode ? (
              <>
                <Button
                  variant="outline"
                  onClick={toggleSelectionMode}
                >
                  <X className="h-4 w-4 mr-2" />
                  Cancel
                </Button>
                <Button
                  onClick={handleMergeClick}
                  disabled={selectedPeople.size < 2}
                >
                  <Users className="h-4 w-4 mr-2" />
                  Merge ({selectedPeople.size})
                </Button>
              </>
            ) : (
              <Button
                variant="outline"
                onClick={toggleSelectionMode}
              >
                <Users className="h-4 w-4 mr-2" />
                Merge People
              </Button>
            )}
          </div>
        </div>

        {error && (
          <div className="bg-destructive/10 text-destructive px-4 py-3 rounded-md mb-6">
            {error}
          </div>
        )}

        {loading ? (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4">
            {Array.from({ length: 12 }).map((_, i) => (
              <Card key={i} className="overflow-hidden">
                <Skeleton className="aspect-square w-full" />
                <CardContent className="p-4">
                  <Skeleton className="h-5 w-full mb-2" />
                  <Skeleton className="h-4 w-20" />
                </CardContent>
              </Card>
            ))}
          </div>
        ) : people.length === 0 ? (
          <div className="text-center py-12">
            <p className="text-muted-foreground text-lg mb-4">
              No people found yet.
            </p>
            <p className="text-sm text-muted-foreground">
              Run face indexing and clustering to identify people in your photos.
            </p>
          </div>
        ) : (
          <>
            <p className="text-sm text-muted-foreground mb-4">
              Found {people.length} {people.length === 1 ? "person" : "people"}
            </p>

            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4">
              {people.map((person) => (
                <div key={person.id} className="relative">
                  {selectionMode && (
                    <div 
                      className="absolute top-2 left-2 z-10"
                      onClick={(e) => {
                        e.preventDefault()
                        e.stopPropagation()
                        togglePersonSelection(person.id)
                      }}
                    >
                      <div className={`w-6 h-6 rounded border-2 flex items-center justify-center cursor-pointer transition-colors ${
                        selectedPeople.has(person.id) 
                          ? 'bg-primary border-primary' 
                          : 'bg-white border-gray-400 hover:border-primary'
                      }`}>
                        {selectedPeople.has(person.id) && (
                          <Check className="h-4 w-4 text-white" />
                        )}
                      </div>
                    </div>
                  )}
                  <Link 
                    href={selectionMode ? '#' : `/people/${person.id}`}
                    onClick={(e) => {
                      if (selectionMode) {
                        e.preventDefault()
                        togglePersonSelection(person.id)
                      }
                    }}
                  >
                    <Card className={`overflow-hidden hover:shadow-lg transition-all cursor-pointer group ${
                      selectedPeople.has(person.id) ? 'ring-2 ring-primary' : ''
                    }`}>
                      <div className="aspect-square relative overflow-hidden bg-muted">
                        {person.representatives && person.representatives.length > 0 ? (
                          <div className="w-full h-full grid grid-cols-2 grid-rows-2 gap-0.5">
                            {person.representatives.slice(0, 4).map((face, idx) => (
                              <div key={idx} className="relative w-full h-full bg-muted">
                                <FaceImage
                                  filename={face.filename}
                                  bbox={face.bbox}
                                  alt={`${person.name} ${idx + 1}`}
                                  className="w-full h-full"
                                  padding={20}
                                />
                              </div>
                            ))}
                            {/* Fill empty slots if less than 4 */}
                            {Array.from({ length: Math.max(0, 4 - person.representatives.length) }).map((_, idx) => (
                              <div key={`empty-${idx}`} className="relative w-full h-full bg-muted/50" />
                            ))}
                          </div>
                        ) : (
                          <div className="w-full h-full flex items-center justify-center">
                            <span className="text-4xl">👤</span>
                          </div>
                        )}
                      </div>
                      <CardContent className="p-4">
                      {editingId === person.id ? (
                        <div className="space-y-2" onClick={(e) => e.preventDefault()}>
                          <Input
                            value={editingName}
                            onChange={(e) => setEditingName(e.target.value)}
                            onClick={(e) => e.stopPropagation()}
                            onKeyDown={(e) => {
                              if (e.key === "Enter") handleSaveName(person.id, e as any)
                              if (e.key === "Escape") handleCancelEdit(e as any)
                            }}
                            className="h-8"
                            autoFocus
                          />
                          <div className="flex gap-1">
                            <Button
                              size="sm"
                              variant="default"
                              className="flex-1 h-7"
                              onClick={(e) => handleSaveName(person.id, e)}
                            >
                              <Check className="h-3 w-3" />
                            </Button>
                            <Button
                              size="sm"
                              variant="ghost"
                              className="flex-1 h-7"
                              onClick={handleCancelEdit}
                            >
                              <X className="h-3 w-3" />
                            </Button>
                          </div>
                        </div>
                      ) : (
                        <>
                          <div className="flex items-center justify-between mb-1">
                            <h3 className="font-semibold truncate flex-1">
                              {person.name}
                            </h3>
                            <Button
                              size="sm"
                              variant="ghost"
                              className="h-6 w-6 p-0 opacity-0 group-hover:opacity-100 transition-opacity"
                              onClick={(e) => handleStartEdit(person, e)}
                            >
                              <Pencil className="h-3 w-3" />
                            </Button>
                          </div>
                          <p className="text-sm text-muted-foreground">
                            {person.faceCount} {person.faceCount === 1 ? "photo" : "photos"}
                          </p>
                        </>
                      )}
                    </CardContent>
                  </Card>
                </Link>
                </div>
              ))}
            </div>
          </>
        )}
      </main>

      {/* Merge Dialog */}
      <Dialog open={showMergeDialog} onOpenChange={setShowMergeDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Merge People</DialogTitle>
            <DialogDescription>
              Merging {selectedPeople.size} people into one. All photos will be combined under a single name.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="merge-name">Person Name</Label>
              <Input
                id="merge-name"
                value={mergeName}
                onChange={(e) => setMergeName(e.target.value)}
                placeholder="Enter name for merged person"
                onKeyDown={(e) => e.key === "Enter" && confirmMerge()}
              />
            </div>
            <div className="text-sm text-muted-foreground">
              <p className="font-semibold mb-2">Selected people:</p>
              <ul className="list-disc list-inside">
                {Array.from(selectedPeople).map(id => {
                  const person = people.find(p => p.id === id)
                  return person ? (
                    <li key={id}>{person.name} ({person.faceCount} photos)</li>
                  ) : null
                })}
              </ul>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowMergeDialog(false)} disabled={merging}>
              Cancel
            </Button>
            <Button onClick={confirmMerge} disabled={merging || !mergeName.trim()}>
              {merging ? "Merging..." : "Merge People"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
