"use client"

import { useState, useEffect } from "react"
import { Header } from "@/components/header"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Tag, CheckCircle, XCircle, Users, AlertTriangle, SkipForward, GitMerge } from "lucide-react"
import axios from "axios"
import { FaceImage } from "@/components/face-image"

interface ClusterData {
  id: string
  name: string
  faceCount: number
  representative: any
  representatives?: any[]
}

interface BoundaryFace {
  face: any
  cluster_id: string
  cluster_name: string
  distance_to_centroid: number
  closest_other_cluster: string | null
  closest_other_name?: string
  distance_to_other: number
  uncertainty_score?: number
  context_faces?: any[]
}

interface SimilarPair {
  cluster1_id: string
  cluster1_name: string
  cluster1_face_count: number
  cluster1_representative: {
    filename: string
    bbox: number[]
  } | null
  cluster2_id: string
  cluster2_name: string
  cluster2_face_count: number
  cluster2_representative: {
    filename: string
    bbox: number[]
  } | null
  similarity: number
}

export default function LabelPage() {
  const [activeTab, setActiveTab] = useState<"label" | "validate" | "merge">("label")

  // Label mode state
  const [unlabeledClusters, setUnlabeledClusters] = useState<ClusterData[]>([])
  const [currentLabelIndex, setCurrentLabelIndex] = useState(0)
  const [labelName, setLabelName] = useState("")
  const [loadingLabel, setLoadingLabel] = useState(true)
  const [labelingInProgress, setLabelingInProgress] = useState(false)

  // Validation mode state
  const [boundaryFaces, setBoundaryFaces] = useState<BoundaryFace[]>([])
  const [currentValidationIndex, setCurrentValidationIndex] = useState(0)
  const [loadingValidation, setLoadingValidation] = useState(true)
  const [validating, setValidating] = useState(false)
  const [showReassignDialog, setShowReassignDialog] = useState(false)
  const [allClusters, setAllClusters] = useState<ClusterData[]>([])
  const [newPersonName, setNewPersonName] = useState("")
  const [searchQuery, setSearchQuery] = useState("")

  // Merge mode state
  const [similarPairs, setSimilarPairs] = useState<SimilarPair[]>([])
  const [currentMergeIndex, setCurrentMergeIndex] = useState(0)
  const [loadingMerge, setLoadingMerge] = useState(true)
  const [merging, setMerging] = useState(false)
  const [mergeName, setMergeName] = useState("")

  // Load data on mount
  useEffect(() => {
    if (activeTab === "label") {
      loadUnlabeledClusters()
    } else if (activeTab === "validate") {
      loadBoundaryFaces()
    } else if (activeTab === "merge") {
      loadSimilarPairs()
    }
  }, [activeTab])

  const loadUnlabeledClusters = async () => {
    setLoadingLabel(true)
    try {
      const response = await axios.get("/api/label/unlabeled")
      setUnlabeledClusters(response.data.clusters)
      setCurrentLabelIndex(0)
    } catch (error) {
      console.error("Error loading unlabeled clusters:", error)
    } finally {
      setLoadingLabel(false)
    }
  }

  const loadBoundaryFaces = async () => {
    setLoadingValidation(true)
    try {
      const [boundaryResponse, clustersResponse] = await Promise.all([
        axios.get("/api/label/boundary"),
        axios.get("/api/people")
      ])
      setBoundaryFaces(boundaryResponse.data.faces)
      setAllClusters(clustersResponse.data.people)
      setCurrentValidationIndex(0)
    } catch (error) {
      console.error("Error loading boundary faces:", error)
    } finally {
      setLoadingValidation(false)
    }
  }

  const loadSimilarPairs = async () => {
    setLoadingMerge(true)
    try {
      const response = await axios.get("/api/label/similar-clusters?min_similarity=0.7&max_pairs=50")
      setSimilarPairs(response.data.pairs)
      setCurrentMergeIndex(0)
    } catch (error) {
      console.error("Error loading similar pairs:", error)
    } finally {
      setLoadingMerge(false)
    }
  }

  const handleSubmitName = async () => {
    if (!labelName.trim()) return

    setLabelingInProgress(true)
    try {
      const currentCluster = unlabeledClusters[currentLabelIndex]
      await axios.put(`/api/people/${currentCluster.id}`, { name: labelName })

      // Move to next cluster
      setLabelName("")
      setCurrentLabelIndex(currentLabelIndex + 1)
    } catch (error) {
      console.error("Error submitting name:", error)
      alert("Failed to update name")
    } finally {
      setLabelingInProgress(false)
    }
  }

  const handleSkipLabel = () => {
    setLabelName("")
    setCurrentLabelIndex(currentLabelIndex + 1)
  }

  const handleValidateYes = () => {
    setCurrentValidationIndex(currentValidationIndex + 1)
  }

  const handleValidateNo = () => {
    setShowReassignDialog(true)
  }

  const handleNotAFace = async () => {
    if (!confirm("Are you sure this is not a face? This will permanently remove it from the cluster.")) {
      return
    }

    setValidating(true)
    try {
      const currentFace = boundaryFaces[currentValidationIndex]

      await axios.post("/api/label/remove-face", {
        faceId: currentFace.face.face_id,
        clusterId: currentFace.cluster_id
      })

      // Move to next face
      setCurrentValidationIndex(currentValidationIndex + 1)
    } catch (error) {
      console.error("Error removing face:", error)
      alert("Failed to remove face")
    } finally {
      setValidating(false)
    }
  }

  const handleSkipValidation = () => {
    // Simply move to the next face without making any changes
    setCurrentValidationIndex(currentValidationIndex + 1)
  }

  const handleReassign = async (targetClusterId: string) => {
    setValidating(true)
    try {
      const currentFace = boundaryFaces[currentValidationIndex]

      await axios.post("/api/label/reassign", {
        faceId: currentFace.face.face_id,
        sourceClusterId: currentFace.cluster_id,
        targetClusterId,
        newPersonName: targetClusterId === "NEW" ? newPersonName : undefined
      })

      setShowReassignDialog(false)
      setNewPersonName("")
      setSearchQuery("")
      setCurrentValidationIndex(currentValidationIndex + 1)
    } catch (error) {
      console.error("Error reassigning face:", error)
      alert("Failed to reassign face")
    } finally {
      setValidating(false)
    }
  }

  const handleMergeClusters = async () => {
    if (!mergeName.trim()) return

    setMerging(true)
    try {
      const currentPair = similarPairs[currentMergeIndex]
      
      await axios.post("/api/people/merge", {
        clusterIds: [currentPair.cluster1_id, currentPair.cluster2_id],
        name: mergeName
      })

      setMergeName("")
      setCurrentMergeIndex(currentMergeIndex + 1)
    } catch (error) {
      console.error("Error merging clusters:", error)
      alert("Failed to merge clusters")
    } finally {
      setMerging(false)
    }
  }

  const handleSkipMerge = () => {
    setMergeName("")
    setCurrentMergeIndex(currentMergeIndex + 1)
  }

  const handleKeepSeparate = () => {
    setCurrentMergeIndex(currentMergeIndex + 1)
  }

  // Filter clusters based on search query
  const filteredClusters = allClusters.filter(cluster =>
    cluster.name.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const renderLabelMode = () => {
    if (loadingLabel) {
      return (
        <div className="space-y-4">
          <Skeleton className="h-64 w-full" />
          <Skeleton className="h-10 w-full" />
        </div>
      )
    }

    if (unlabeledClusters.length === 0) {
      return (
        <Card>
          <CardContent className="pt-6">
            <div className="text-center text-muted-foreground">
              <CheckCircle className="h-12 w-12 mx-auto mb-4 text-green-500" />
              <p className="text-lg font-medium mb-2">All done!</p>
              <p>No unlabeled clusters with more than 5 photos found.</p>
            </div>
          </CardContent>
        </Card>
      )
    }

    if (currentLabelIndex >= unlabeledClusters.length) {
      return (
        <Card>
          <CardContent className="pt-6">
            <div className="text-center text-muted-foreground">
              <CheckCircle className="h-12 w-12 mx-auto mb-4 text-green-500" />
              <p className="text-lg font-medium mb-2">Labeling complete!</p>
              <p className="mb-4">You&apos;ve processed all unlabeled clusters.</p>
              <Button onClick={() => setCurrentLabelIndex(0)}>
                Start Over
              </Button>
            </div>
          </CardContent>
        </Card>
      )
    }

    const currentCluster = unlabeledClusters[currentLabelIndex]
    const progress = ((currentLabelIndex) / unlabeledClusters.length) * 100

    return (
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>Label Cluster {currentLabelIndex + 1} of {unlabeledClusters.length}</CardTitle>
                <CardDescription>
                  This cluster contains {currentCluster.faceCount} photos
                </CardDescription>
              </div>
              <Badge variant="outline">{Math.round(progress)}% Complete</Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-6">
            <Progress value={progress} />

            {/* Face Images - 2x2 Grid */}
            <div className="flex justify-center">
              {currentCluster.representatives && currentCluster.representatives.length > 0 ? (
                <div className="w-96 h-96 grid grid-cols-2 grid-rows-2 gap-2 border-2 border-border rounded-lg overflow-hidden bg-muted">
                  {currentCluster.representatives.slice(0, 4).map((face, idx) => (
                    <div key={idx} className="relative w-full h-full bg-muted">
                      <FaceImage
                        filename={face.filename}
                        bbox={face.bbox}
                        alt={`Representative face ${idx + 1}`}
                        className="w-full h-full"
                        padding={20}
                      />
                    </div>
                  ))}
                  {/* Fill empty slots if less than 4 */}
                  {Array.from({ length: Math.max(0, 4 - currentCluster.representatives.length) }).map((_, idx) => (
                    <div key={`empty-${idx}`} className="relative w-full h-full bg-muted/50" />
                  ))}
                </div>
              ) : currentCluster.representative ? (
                <FaceImage
                  filename={currentCluster.representative.filename}
                  bbox={currentCluster.representative.bbox}
                  alt="Representative face"
                  className="w-96 h-96 border-2 border-border rounded-lg bg-muted"
                  padding={50}
                />
              ) : null}
            </div>

            {/* Input */}
            <div className="space-y-2">
              <Label htmlFor="name">Person Name</Label>
              <Input
                id="name"
                placeholder="Enter person's name..."
                value={labelName}
                onChange={(e) => setLabelName(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSubmitName()}
                disabled={labelingInProgress}
                autoFocus
              />
            </div>

            {/* Buttons */}
            <div className="flex gap-3 justify-center">
              <Button
                onClick={handleSubmitName}
                disabled={!labelName.trim() || labelingInProgress}
                className="min-w-32"
              >
                <Tag className="mr-2 h-4 w-4" />
                Submit Name
              </Button>
              <Button
                variant="outline"
                onClick={handleSkipLabel}
                disabled={labelingInProgress}
              >
                Skip
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  const renderValidationMode = () => {
    if (loadingValidation) {
      return (
        <div className="space-y-4">
          <Skeleton className="h-64 w-full" />
          <Skeleton className="h-10 w-full" />
        </div>
      )
    }

    if (boundaryFaces.length === 0) {
      return (
        <Card>
          <CardContent className="pt-6">
            <div className="text-center text-muted-foreground">
              <CheckCircle className="h-12 w-12 mx-auto mb-4 text-green-500" />
              <p className="text-lg font-medium mb-2">All validated!</p>
              <p>No boundary faces found for validation.</p>
            </div>
          </CardContent>
        </Card>
      )
    }

    if (currentValidationIndex >= boundaryFaces.length) {
      return (
        <Card>
          <CardContent className="pt-6">
            <div className="text-center text-muted-foreground">
              <CheckCircle className="h-12 w-12 mx-auto mb-4 text-green-500" />
              <p className="text-lg font-medium mb-2">Validation complete!</p>
              <p className="mb-4">You&apos;ve processed all boundary faces.</p>
              <Button onClick={() => setCurrentValidationIndex(0)}>
                Start Over
              </Button>
            </div>
          </CardContent>
        </Card>
      )
    }

    const currentFace = boundaryFaces[currentValidationIndex]
    const progress = ((currentValidationIndex) / boundaryFaces.length) * 100

    return (
      <>
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Validate Face {currentValidationIndex + 1} of {boundaryFaces.length}</CardTitle>
                  <CardDescription className="space-y-1 mt-2">
                    <div>Distance to &quot;{currentFace.cluster_name}&quot;: <span className="font-mono">{currentFace.distance_to_centroid.toFixed(3)}</span></div>
                    {currentFace.closest_other_name && (
                      <div>Distance to &quot;{currentFace.closest_other_name}&quot;: <span className="font-mono">{currentFace.distance_to_other.toFixed(3)}</span></div>
                    )}
                    <div className="text-amber-600 dark:text-amber-400 font-medium">
                      Uncertainty score: {currentFace.uncertainty_score?.toFixed(3) || 'N/A'}
                    </div>
                  </CardDescription>
                </div>
                <Badge variant="outline">{Math.round(progress)}% Complete</Badge>
              </div>
            </CardHeader>
            <CardContent className="space-y-6">
              <Progress value={progress} />

              {/* Face Images - 2x2 Grid with main face and context */}
              <div className="flex justify-center">
                {currentFace.face && (
                  <div className="w-96 h-96 grid grid-cols-2 grid-rows-2 gap-2 border-2 border-border rounded-lg overflow-hidden bg-muted">
                    {/* Main uncertain face - larger, top-left */}
                    <div className="relative w-full h-full bg-muted col-span-2 row-span-2">
                      <FaceImage
                        filename={currentFace.face.filename}
                        bbox={currentFace.face.bbox}
                        alt="Face to validate"
                        className="w-full h-full"
                        padding={50}
                      />
                      <div className="absolute top-2 left-2 bg-amber-500 text-white px-2 py-1 rounded text-xs font-semibold">
                        Uncertain
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Context faces from the same cluster */}
              {currentFace.context_faces && currentFace.context_faces.length > 0 && (
                <div>
                  <p className="text-sm text-muted-foreground text-center mb-2">
                    Other faces in &quot;{currentFace.cluster_name}&quot; cluster:
                  </p>
                  <div className="flex justify-center gap-2">
                    {currentFace.context_faces.slice(0, 3).map((face, idx) => (
                      <div key={idx} className="relative w-24 h-24 rounded border border-border overflow-hidden bg-muted">
                        <FaceImage
                          filename={face.filename}
                          bbox={face.bbox}
                          alt={`Context face ${idx + 1}`}
                          className="w-full h-full"
                          padding={20}
                        />
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Question */}
              <div className="text-center space-y-2">
                <p className="text-lg font-medium flex items-center justify-center gap-2">
                  <AlertTriangle className="h-5 w-5 text-amber-500" />
                  Is this person &quot;{currentFace.cluster_name}&quot;?
                </p>
              </div>

              {/* Buttons */}
              <div className="flex gap-3 justify-center flex-wrap">
                <Button
                  onClick={handleValidateYes}
                  disabled={validating}
                  className="min-w-32"
                >
                  <CheckCircle className="mr-2 h-4 w-4" />
                  Yes, Correct
                </Button>
                <Button
                  variant="outline"
                  onClick={handleValidateNo}
                  disabled={validating}
                  className="min-w-32"
                >
                  <XCircle className="mr-2 h-4 w-4" />
                  No, Incorrect
                </Button>
                <Button
                  variant="destructive"
                  onClick={handleNotAFace}
                  disabled={validating}
                  className="min-w-32"
                >
                  <XCircle className="mr-2 h-4 w-4" />
                  Not a Face
                </Button>
                <Button
                  variant="secondary"
                  onClick={handleSkipValidation}
                  disabled={validating}
                  className="min-w-32"
                >
                  <SkipForward className="mr-2 h-4 w-4" />
                  Skip
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Reassign Dialog */}
        <Dialog 
          open={showReassignDialog} 
          onOpenChange={(open) => {
            setShowReassignDialog(open)
            if (!open) setSearchQuery("") // Reset search when closing
          }}
        >
          <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle>Select Correct Person</DialogTitle>
              <DialogDescription>
                Choose the correct person for this face or create a new person
              </DialogDescription>
            </DialogHeader>

            <div className="space-y-4">
              {/* Search Input */}
              <div className="space-y-2">
                <Label htmlFor="search">Search People</Label>
                <Input
                  id="search"
                  type="text"
                  placeholder="Type to search by name..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && filteredClusters.length > 0) {
                      handleReassign(filteredClusters[0].id)
                    }
                  }}
                  autoFocus
                  className="w-full"
                />
                {searchQuery && (
                  <p className="text-sm text-muted-foreground">
                    Found {filteredClusters.length} {filteredClusters.length === 1 ? 'person' : 'people'}
                    {filteredClusters.length > 0 && ' (Press Enter to select first)'}
                  </p>
                )}
              </div>

              {/* New Person Option */}
              <Card
                className="cursor-pointer hover:bg-accent transition-colors"
                onClick={() => {
                  const name = prompt("Enter new person's name:")
                  if (name) {
                    setNewPersonName(name)
                    handleReassign("NEW")
                  }
                }}
              >
                <CardContent className="py-4">
                  <div className="flex items-center gap-3">
                    <div className="h-12 w-12 rounded-full bg-primary flex items-center justify-center">
                      <Users className="h-6 w-6 text-primary-foreground" />
                    </div>
                    <div>
                      <p className="font-medium">Create New Person</p>
                      <p className="text-sm text-muted-foreground">Add as a new person</p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Existing People */}
              <div className="space-y-2">
                {filteredClusters.length > 0 ? (
                  filteredClusters.map((cluster) => (
                    <Card
                      key={cluster.id}
                      className="cursor-pointer hover:bg-accent transition-colors"
                      onClick={() => handleReassign(cluster.id)}
                    >
                      <CardContent className="py-4">
                        <div className="flex items-center gap-3">
                          <div className="relative h-12 w-12 rounded-full overflow-hidden bg-muted">
                            {cluster.representative && (
                              <FaceImage
                                filename={cluster.representative.filename}
                                bbox={cluster.representative.bbox}
                                alt={cluster.name}
                                className="h-12 w-12"
                                padding={20}
                              />
                            )}
                          </div>
                          <div>
                            <p className="font-medium">{cluster.name}</p>
                            <p className="text-sm text-muted-foreground">
                              {cluster.faceCount} faces
                            </p>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    <p>No people found matching &quot;{searchQuery}&quot;</p>
                    <p className="text-sm mt-2">Try a different search term</p>
                  </div>
                )}
              </div>
            </div>
          </DialogContent>
        </Dialog>
      </>
    )
  }

  const renderMergeMode = () => {
    if (loadingMerge) {
      return (
        <div className="space-y-4">
          <Skeleton className="h-64 w-full" />
          <Skeleton className="h-10 w-full" />
        </div>
      )
    }

    if (similarPairs.length === 0) {
      return (
        <Card>
          <CardContent className="pt-6">
            <div className="text-center text-muted-foreground">
              <CheckCircle className="h-12 w-12 mx-auto mb-4 text-green-500" />
              <p className="text-lg font-medium mb-2">All done!</p>
              <p>No similar clusters found that need merging.</p>
            </div>
          </CardContent>
        </Card>
      )
    }

    if (currentMergeIndex >= similarPairs.length) {
      return (
        <Card>
          <CardContent className="pt-6">
            <div className="text-center text-muted-foreground">
              <CheckCircle className="h-12 w-12 mx-auto mb-4 text-green-500" />
              <p className="text-lg font-medium mb-2">Merging complete!</p>
              <p className="mb-4">You&apos;ve processed all similar cluster pairs.</p>
              <Button onClick={() => setCurrentMergeIndex(0)}>
                Start Over
              </Button>
            </div>
          </CardContent>
        </Card>
      )
    }

    const currentPair = similarPairs[currentMergeIndex]
    const progress = ((currentMergeIndex) / similarPairs.length) * 100

    return (
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>Review Pair {currentMergeIndex + 1} of {similarPairs.length}</CardTitle>
                <CardDescription>
                  Similarity: <span className="font-mono font-semibold text-primary">{(currentPair.similarity * 100).toFixed(1)}%</span>
                </CardDescription>
              </div>
              <Badge variant="outline">{Math.round(progress)}% Complete</Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-6">
            <Progress value={progress} />

            {/* Two clusters side by side */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Cluster 1 */}
              <div className="space-y-3">
                <div className="text-center">
                  <h3 className="font-semibold text-lg">{currentPair.cluster1_name}</h3>
                  <p className="text-sm text-muted-foreground">{currentPair.cluster1_face_count} faces</p>
                </div>
                {currentPair.cluster1_representative && (
                  <div className="flex justify-center">
                    <FaceImage
                      filename={currentPair.cluster1_representative.filename}
                      bbox={currentPair.cluster1_representative.bbox}
                      alt={currentPair.cluster1_name}
                      className="w-64 h-64 border-2 border-border rounded-lg bg-muted"
                      padding={30}
                    />
                  </div>
                )}
              </div>

              {/* Cluster 2 */}
              <div className="space-y-3">
                <div className="text-center">
                  <h3 className="font-semibold text-lg">{currentPair.cluster2_name}</h3>
                  <p className="text-sm text-muted-foreground">{currentPair.cluster2_face_count} faces</p>
                </div>
                {currentPair.cluster2_representative && (
                  <div className="flex justify-center">
                    <FaceImage
                      filename={currentPair.cluster2_representative.filename}
                      bbox={currentPair.cluster2_representative.bbox}
                      alt={currentPair.cluster2_name}
                      className="w-64 h-64 border-2 border-border rounded-lg bg-muted"
                      padding={30}
                    />
                  </div>
                )}
              </div>
            </div>

            {/* Question */}
            <div className="text-center space-y-2">
              <p className="text-lg font-medium flex items-center justify-center gap-2">
                <GitMerge className="h-5 w-5 text-primary" />
                Are these the same person?
              </p>
              <p className="text-sm text-muted-foreground">
                If yes, they will be merged into one cluster
              </p>
            </div>

            {/* Merge name input (shown when user wants to merge) */}
            {mergeName !== null && (
              <div className="space-y-2">
                <Label htmlFor="merge-name">Merged Person Name</Label>
                <Input
                  id="merge-name"
                  placeholder={`Enter name (default: ${currentPair.cluster1_name})`}
                  value={mergeName}
                  onChange={(e) => setMergeName(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      if (!mergeName.trim()) {
                        setMergeName(currentPair.cluster1_name)
                      }
                      handleMergeClusters()
                    }
                  }}
                  disabled={merging}
                  autoFocus
                />
              </div>
            )}

            {/* Buttons */}
            <div className="flex gap-3 justify-center flex-wrap">
              <Button
                onClick={() => {
                  if (!mergeName.trim()) {
                    setMergeName(currentPair.cluster1_name)
                    setTimeout(handleMergeClusters, 0)
                  } else {
                    handleMergeClusters()
                  }
                }}
                disabled={merging}
                className="min-w-32"
              >
                <GitMerge className="mr-2 h-4 w-4" />
                Yes, Merge
              </Button>
              <Button
                variant="outline"
                onClick={handleKeepSeparate}
                disabled={merging}
                className="min-w-32"
              >
                <XCircle className="mr-2 h-4 w-4" />
                No, Keep Separate
              </Button>
              <Button
                variant="secondary"
                onClick={handleSkipMerge}
                disabled={merging}
                className="min-w-32"
              >
                <SkipForward className="mr-2 h-4 w-4" />
                Skip
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto space-y-6">
          <div>
            <h1 className="text-3xl font-bold mb-2">Face Labeling</h1>
            <p className="text-muted-foreground">
              Label unlabeled clusters, validate uncertain faces, and merge similar clusters
            </p>
          </div>

          <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as "label" | "validate" | "merge")}>
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="label">
                <Tag className="mr-2 h-4 w-4" />
                Label Clusters ({unlabeledClusters.length})
              </TabsTrigger>
              <TabsTrigger value="validate">
                <AlertTriangle className="mr-2 h-4 w-4" />
                Validate Faces ({boundaryFaces.length})
              </TabsTrigger>
              <TabsTrigger value="merge">
                <GitMerge className="mr-2 h-4 w-4" />
                Merge Similar ({similarPairs.length})
              </TabsTrigger>
            </TabsList>

            <TabsContent value="label" className="mt-6">
              {renderLabelMode()}
            </TabsContent>

            <TabsContent value="validate" className="mt-6">
              {renderValidationMode()}
            </TabsContent>

            <TabsContent value="merge" className="mt-6">
              {renderMergeMode()}
            </TabsContent>
          </Tabs>
        </div>
      </main>
    </div>
  )
}
