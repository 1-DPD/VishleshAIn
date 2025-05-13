"use client"

import type React from "react"

import { useState } from "react"
import PageLayout from "../page-layout"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { useToast } from "@/hooks/use-toast"
import { Upload, LinkIcon, FileType, FileSpreadsheet, FileJson, Database } from "lucide-react"
import { useRouter } from "next/navigation"

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null)
  const [url, setUrl] = useState("")
  const [isUploading, setIsUploading] = useState(false)
  const { toast } = useToast()
  const router = useRouter()

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
    }
  }

  const handleUrlChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setUrl(e.target.value)
  }

  const handleUpload = async () => {
    setIsUploading(true)

    try {
      // Simulate API call
      await new Promise((resolve) => setTimeout(resolve, 2000))

      toast({
        title: "Upload successful",
        description: "Your data has been uploaded and is ready for preprocessing.",
      })

      // Navigate to preprocessing page
      router.push("/preprocessing")
    } catch (error) {
      toast({
        title: "Upload failed",
        description: "There was an error uploading your data. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsUploading(false)
    }
  }

  const handleUrlSubmit = async () => {
    if (!url) {
      toast({
        title: "URL required",
        description: "Please enter a valid URL to extract data from.",
        variant: "destructive",
      })
      return
    }

    setIsUploading(true)

    try {
      // Simulate API call
      await new Promise((resolve) => setTimeout(resolve, 2000))

      toast({
        title: "Data extraction successful",
        description: "Data has been extracted from the URL and is ready for preprocessing.",
      })

      // Navigate to preprocessing page
      router.push("/preprocessing")
    } catch (error) {
      toast({
        title: "Data extraction failed",
        description: "There was an error extracting data from the URL. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsUploading(false)
    }
  }

  return (
    <PageLayout>
      <div className="container py-10">
        <div className="mx-auto max-w-3xl">
          <div className="mb-8 text-center">
            <h1 className="text-3xl font-bold mb-2">Upload Your Data</h1>
            <p className="text-muted-foreground">Upload your data file or provide a URL to extract data from</p>
          </div>

          <Tabs defaultValue="file" className="w-full">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="file">File Upload</TabsTrigger>
              <TabsTrigger value="url">URL</TabsTrigger>
            </TabsList>

            <TabsContent value="file">
              <Card>
                <CardHeader>
                  <CardTitle>Upload Data File</CardTitle>
                  <CardDescription>Upload data in CSV, JSON, XML, PDF, or DOCX format</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid w-full gap-4">
                    <div className="flex flex-col items-center justify-center border-2 border-dashed rounded-lg p-12 text-center">
                      <div className="mb-4 rounded-full bg-primary/10 p-4">
                        <Upload className="h-6 w-6 text-primary" />
                      </div>
                      <div className="mb-4">
                        <h3 className="text-lg font-semibold">Drag and drop your file here</h3>
                        <p className="text-sm text-muted-foreground">or click to browse files</p>
                      </div>
                      <Input
                        id="file-upload"
                        type="file"
                        className="hidden"
                        accept=".csv,.json,.xml,.pdf,.docx,.xlsx,.xls"
                        onChange={handleFileChange}
                      />
                      <Label
                        htmlFor="file-upload"
                        className="cursor-pointer inline-flex h-9 items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground shadow transition-colors hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50"
                      >
                        Select File
                      </Label>
                    </div>

                    {file && (
                      <div className="flex items-center gap-2 p-2 border rounded-md">
                        <div className="rounded-full bg-primary/10 p-2">
                          {file.name.endsWith(".csv") ? (
                            <FileSpreadsheet className="h-4 w-4 text-primary" />
                          ) : file.name.endsWith(".json") ? (
                            <FileJson className="h-4 w-4 text-primary" />
                          ) : (
                            <FileType className="h-4 w-4 text-primary" />
                          )}
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium truncate">{file.name}</p>
                          <p className="text-xs text-muted-foreground">{(file.size / 1024).toFixed(2)} KB</p>
                        </div>
                        <Button variant="ghost" size="sm" onClick={() => setFile(null)}>
                          Remove
                        </Button>
                      </div>
                    )}
                  </div>
                </CardContent>
                <CardFooter>
                  <Button className="w-full" onClick={handleUpload} disabled={!file || isUploading}>
                    {isUploading ? "Uploading..." : "Upload and Continue"}
                  </Button>
                </CardFooter>
              </Card>
            </TabsContent>

            <TabsContent value="url">
              <Card>
                <CardHeader>
                  <CardTitle>Extract Data from URL</CardTitle>
                  <CardDescription>
                    Provide a URL to extract data from web pages, APIs, or online datasets
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid w-full gap-4">
                    <div className="flex flex-col space-y-2">
                      <Label htmlFor="url">URL</Label>
                      <div className="flex items-center gap-2">
                        <div className="rounded-l-md border border-r-0 bg-muted px-3 py-2">
                          <LinkIcon className="h-4 w-4 text-muted-foreground" />
                        </div>
                        <Input
                          id="url"
                          placeholder="https://example.com/data"
                          value={url}
                          onChange={handleUrlChange}
                          className="rounded-l-none"
                        />
                      </div>
                    </div>
                  </div>
                </CardContent>
                <CardFooter>
                  <Button className="w-full" onClick={handleUrlSubmit} disabled={!url || isUploading}>
                    {isUploading ? "Extracting Data..." : "Extract Data and Continue"}
                  </Button>
                </CardFooter>
              </Card>
            </TabsContent>
          </Tabs>

          <div className="mt-8">
            <h2 className="text-xl font-semibold mb-4">Supported Data Formats</h2>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <div className="flex items-center gap-2 p-4 border rounded-md">
                <FileSpreadsheet className="h-5 w-5 text-primary" />
                <span>CSV</span>
              </div>
              <div className="flex items-center gap-2 p-4 border rounded-md">
                <FileJson className="h-5 w-5 text-primary" />
                <span>JSON</span>
              </div>
              <div className="flex items-center gap-2 p-4 border rounded-md">
                <FileType className="h-5 w-5 text-primary" />
                <span>XML</span>
              </div>
              <div className="flex items-center gap-2 p-4 border rounded-md">
                <FileType className="h-5 w-5 text-primary" />
                <span>PDF</span>
              </div>
              <div className="flex items-center gap-2 p-4 border rounded-md">
                <FileType className="h-5 w-5 text-primary" />
                <span>DOCX</span>
              </div>
              <div className="flex items-center gap-2 p-4 border rounded-md">
                <Database className="h-5 w-5 text-primary" />
                <span>Web Data</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </PageLayout>
  )
}
