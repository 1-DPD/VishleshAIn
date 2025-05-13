"use client"

import { useState } from "react"
import PageLayout from "../page-layout"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { useToast } from "@/hooks/use-toast"
import { useRouter } from "next/navigation"
import { Download, FileText, BarChart2, Database, Home, CheckCircle2, RefreshCw } from "lucide-react"
import { Progress } from "@/components/ui/progress"

export default function DownloadPage() {
  const [isDownloading, setIsDownloading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [downloadComplete, setDownloadComplete] = useState(false)
  const { toast } = useToast()
  const router = useRouter()

  const handleDownload = async () => {
    setIsDownloading(true)
    setDownloadComplete(false)

    // Simulate download with progress updates
    for (let i = 0; i <= 100; i += 5) {
      await new Promise((resolve) => setTimeout(resolve, 100))
      setProgress(i)
    }

    try {
      toast({
        title: "Download complete",
        description: "All files have been successfully downloaded.",
      })

      setDownloadComplete(true)
    } catch (error) {
      toast({
        title: "Download failed",
        description: "There was an error downloading your files. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsDownloading(false)
      setProgress(0)
    }
  }

  return (
    <PageLayout>
      <div className="container py-10">
        <div className="mx-auto max-w-4xl">
          <div className="mb-8 text-center">
            <h1 className="text-3xl font-bold mb-2">Download Files</h1>
            <p className="text-muted-foreground">Download your report, visualizations, and processed data</p>
          </div>

          <div className="grid gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Available Files</CardTitle>
                <CardDescription>Files ready for download</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-4 border rounded-md">
                    <div className="flex items-center gap-4">
                      <div className="rounded-full bg-primary/10 p-2">
                        <FileText className="h-6 w-6 text-primary" />
                      </div>
                      <div>
                        <p className="font-medium">Data Analysis Report</p>
                        <p className="text-sm text-muted-foreground">PDF, 10 pages, 2.5 MB</p>
                      </div>
                    </div>
                    <Button size="sm" variant="outline" className="gap-1">
                      <Download className="h-4 w-4" />
                      Download
                    </Button>
                  </div>

                  <div className="flex items-center justify-between p-4 border rounded-md">
                    <div className="flex items-center gap-4">
                      <div className="rounded-full bg-primary/10 p-2">
                        <BarChart2 className="h-6 w-6 text-primary" />
                      </div>
                      <div>
                        <p className="font-medium">Visualizations</p>
                        <p className="text-sm text-muted-foreground">8 PNG files, 4.2 MB</p>
                      </div>
                    </div>
                    <Button size="sm" variant="outline" className="gap-1">
                      <Download className="h-4 w-4" />
                      Download
                    </Button>
                  </div>

                  <div className="flex items-center justify-between p-4 border rounded-md">
                    <div className="flex items-center gap-4">
                      <div className="rounded-full bg-primary/10 p-2">
                        <Database className="h-6 w-6 text-primary" />
                      </div>
                      <div>
                        <p className="font-medium">Processed Dataset</p>
                        <p className="text-sm text-muted-foreground">CSV, 1.8 MB</p>
                      </div>
                    </div>
                    <Button size="sm" variant="outline" className="gap-1">
                      <Download className="h-4 w-4" />
                      Download
                    </Button>
                  </div>

                  <div className="flex items-center justify-between p-4 border rounded-md">
                    <div className="flex items-center gap-4">
                      <div className="rounded-full bg-primary/10 p-2">
                        <Database className="h-6 w-6 text-primary" />
                      </div>
                      <div>
                        <p className="font-medium">Model Files</p>
                        <p className="text-sm text-muted-foreground">Pickle files, 15.3 MB</p>
                      </div>
                    </div>
                    <Button size="sm" variant="outline" className="gap-1">
                      <Download className="h-4 w-4" />
                      Download
                    </Button>
                  </div>
                </div>
              </CardContent>
              <CardFooter>
                <Button onClick={handleDownload} disabled={isDownloading} className="w-full gap-2">
                  {isDownloading ? (
                    <>
                      <RefreshCw className="h-4 w-4 animate-spin" />
                      Downloading...
                    </>
                  ) : (
                    <>
                      <Download className="h-4 w-4" />
                      Download All Files (ZIP)
                    </>
                  )}
                </Button>
              </CardFooter>
            </Card>

            {isDownloading && (
              <Card>
                <CardContent className="pt-6">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Downloading files...</span>
                      <span className="text-sm text-muted-foreground">{progress}%</span>
                    </div>
                    <Progress value={progress} />
                  </div>
                </CardContent>
              </Card>
            )}

            {downloadComplete && (
              <Card className="bg-green-50 dark:bg-green-950 border-green-200 dark:border-green-800">
                <CardContent className="pt-6 flex items-center gap-4">
                  <div className="rounded-full bg-green-100 p-2 dark:bg-green-900">
                    <CheckCircle2 className="h-6 w-6 text-green-600 dark:text-green-400" />
                  </div>
                  <div>
                    <p className="font-medium text-green-800 dark:text-green-300">Download Complete</p>
                    <p className="text-sm text-green-600 dark:text-green-400">
                      All files have been successfully downloaded to your device.
                    </p>
                  </div>
                </CardContent>
              </Card>
            )}

            <Card>
              <CardHeader>
                <CardTitle>Next Steps</CardTitle>
                <CardDescription>What you can do with your analysis results</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid gap-4">
                  <div className="p-4 border rounded-md">
                    <h3 className="text-lg font-medium mb-2">Review the Report</h3>
                    <p className="text-muted-foreground mb-4">
                      Carefully review the comprehensive report to understand the insights and findings from your data
                      analysis.
                    </p>
                    <ul className="space-y-2">
                      <li className="flex items-start gap-2">
                        <div className="mt-0.5 rounded-full bg-primary/10 p-1">
                          <CheckCircle2 className="h-4 w-4 text-primary" />
                        </div>
                        <span>Understand the methodology used for preprocessing and modeling</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <div className="mt-0.5 rounded-full bg-primary/10 p-1">
                          <CheckCircle2 className="h-4 w-4 text-primary" />
                        </div>
                        <span>Examine the visualizations to identify patterns and relationships</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <div className="mt-0.5 rounded-full bg-primary/10 p-1">
                          <CheckCircle2 className="h-4 w-4 text-primary" />
                        </div>
                        <span>Review the model performance metrics and insights</span>
                      </li>
                    </ul>
                  </div>

                  <div className="p-4 border rounded-md">
                    <h3 className="text-lg font-medium mb-2">Apply the Insights</h3>
                    <p className="text-muted-foreground mb-4">
                      Use the insights from your analysis to inform decision-making and take action.
                    </p>
                    <ul className="space-y-2">
                      <li className="flex items-start gap-2">
                        <div className="mt-0.5 rounded-full bg-primary/10 p-1">
                          <CheckCircle2 className="h-4 w-4 text-primary" />
                        </div>
                        <span>Implement recommendations based on the findings</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <div className="mt-0.5 rounded-full bg-primary/10 p-1">
                          <CheckCircle2 className="h-4 w-4 text-primary" />
                        </div>
                        <span>Share the report with stakeholders for collaborative decision-making</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <div className="mt-0.5 rounded-full bg-primary/10 p-1">
                          <CheckCircle2 className="h-4 w-4 text-primary" />
                        </div>
                        <span>Use the model for predictions on new data</span>
                      </li>
                    </ul>
                  </div>

                  <div className="p-4 border rounded-md">
                    <h3 className="text-lg font-medium mb-2">Start a New Analysis</h3>
                    <p className="text-muted-foreground mb-4">
                      Begin a new analysis with different data or explore additional aspects of your current dataset.
                    </p>
                    <Button onClick={() => router.push("/")} className="gap-2">
                      <Home className="h-4 w-4" />
                      Return to Home
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </PageLayout>
  )
}
