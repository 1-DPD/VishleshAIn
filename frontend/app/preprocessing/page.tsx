"use client"

import { useState } from "react"
import PageLayout from "../page-layout"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Label } from "@/components/ui/label"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Textarea } from "@/components/ui/textarea"
import { useToast } from "@/hooks/use-toast"
import { useRouter } from "next/navigation"
import { AlertCircle, ArrowRight, RefreshCw } from "lucide-react"
import { Progress } from "@/components/ui/progress"

export default function PreprocessingPage() {
  const [activeTab, setActiveTab] = useState("basic")
  const [isProcessing, setIsProcessing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [requirements, setRequirements] = useState("")
  const { toast } = useToast()
  const router = useRouter()

  const [basicOptions, setBasicOptions] = useState({
    handleMissingValues: true,
    removeDuplicates: true,
    encodeCategorial: true,
    fixInconsistentData: true,
    handleOutliers: true,
  })

  const [advancedOptions, setAdvancedOptions] = useState({
    normalization: false,
    scaling: false,
    featureEngineering: false,
    dimensionalityReduction: false,
    dataCompression: false,
  })

  const handleBasicOptionChange = (option: keyof typeof basicOptions) => {
    setBasicOptions({
      ...basicOptions,
      [option]: !basicOptions[option],
    })
  }

  const handleAdvancedOptionChange = (option: keyof typeof advancedOptions) => {
    setAdvancedOptions({
      ...advancedOptions,
      [option]: !advancedOptions[option],
    })
  }

  const handleProcessing = async () => {
    if (!Object.values(basicOptions).some((value) => value)) {
      toast({
        title: "No preprocessing options selected",
        description: "Please select at least one preprocessing option to continue.",
        variant: "destructive",
      })
      return
    }

    setIsProcessing(true)

    // Simulate processing with progress updates
    for (let i = 0; i <= 100; i += 10) {
      await new Promise((resolve) => setTimeout(resolve, 300))
      setProgress(i)
    }

    try {
      toast({
        title: "Preprocessing complete",
        description: "Your data has been successfully preprocessed.",
      })

      // Navigate to analysis page
      router.push("/analysis")
    } catch (error) {
      toast({
        title: "Preprocessing failed",
        description: "There was an error preprocessing your data. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsProcessing(false)
      setProgress(0)
    }
  }

  return (
    <PageLayout>
      <div className="container py-10">
        <div className="mx-auto max-w-4xl">
          <div className="mb-8 text-center">
            <h1 className="text-3xl font-bold mb-2">Data Preprocessing</h1>
            <p className="text-muted-foreground">Clean and prepare your data for analysis</p>
          </div>

          <div className="grid gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Data Overview</CardTitle>
                <CardDescription>Summary of the uploaded data</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid gap-4">
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="flex flex-col p-4 border rounded-md">
                      <span className="text-sm text-muted-foreground">Rows</span>
                      <span className="text-2xl font-bold">1,245</span>
                    </div>
                    <div className="flex flex-col p-4 border rounded-md">
                      <span className="text-sm text-muted-foreground">Columns</span>
                      <span className="text-2xl font-bold">15</span>
                    </div>
                    <div className="flex flex-col p-4 border rounded-md">
                      <span className="text-sm text-muted-foreground">Missing Values</span>
                      <span className="text-2xl font-bold">124</span>
                    </div>
                    <div className="flex flex-col p-4 border rounded-md">
                      <span className="text-sm text-muted-foreground">Duplicates</span>
                      <span className="text-2xl font-bold">32</span>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="flex items-center gap-2 p-4 border rounded-md">
                      <AlertCircle className="h-5 w-5 text-yellow-500" />
                      <div>
                        <p className="font-medium">Missing Values Detected</p>
                        <p className="text-sm text-muted-foreground">9.9% of data is missing</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 p-4 border rounded-md">
                      <AlertCircle className="h-5 w-5 text-yellow-500" />
                      <div>
                        <p className="font-medium">Outliers Detected</p>
                        <p className="text-sm text-muted-foreground">15 potential outliers found</p>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Tabs value={activeTab} onValueChange={setActiveTab}>
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="basic">Basic Preprocessing</TabsTrigger>
                <TabsTrigger value="advanced">Advanced Options</TabsTrigger>
              </TabsList>

              <TabsContent value="basic">
                <Card>
                  <CardHeader>
                    <CardTitle>Basic Preprocessing Options</CardTitle>
                    <CardDescription>Select the preprocessing steps to apply to your data</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid gap-4">
                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="missing-values"
                          checked={basicOptions.handleMissingValues}
                          onCheckedChange={() => handleBasicOptionChange("handleMissingValues")}
                        />
                        <div className="grid gap-1.5">
                          <Label htmlFor="missing-values" className="font-medium">
                            Handle Missing Values
                          </Label>
                          <p className="text-sm text-muted-foreground">Fill or remove null values and NA values</p>
                        </div>
                      </div>

                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="duplicates"
                          checked={basicOptions.removeDuplicates}
                          onCheckedChange={() => handleBasicOptionChange("removeDuplicates")}
                        />
                        <div className="grid gap-1.5">
                          <Label htmlFor="duplicates" className="font-medium">
                            Remove Duplicates
                          </Label>
                          <p className="text-sm text-muted-foreground">Identify and remove duplicate records</p>
                        </div>
                      </div>

                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="encoding"
                          checked={basicOptions.encodeCategorial}
                          onCheckedChange={() => handleBasicOptionChange("encodeCategorial")}
                        />
                        <div className="grid gap-1.5">
                          <Label htmlFor="encoding" className="font-medium">
                            Encode Categorical Variables
                          </Label>
                          <p className="text-sm text-muted-foreground">Convert categorical data to numerical format</p>
                        </div>
                      </div>

                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="inconsistent"
                          checked={basicOptions.fixInconsistentData}
                          onCheckedChange={() => handleBasicOptionChange("fixInconsistentData")}
                        />
                        <div className="grid gap-1.5">
                          <Label htmlFor="inconsistent" className="font-medium">
                            Fix Inconsistent Data
                          </Label>
                          <p className="text-sm text-muted-foreground">Correct type errors and inconsistent values</p>
                        </div>
                      </div>

                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="outliers"
                          checked={basicOptions.handleOutliers}
                          onCheckedChange={() => handleBasicOptionChange("handleOutliers")}
                        />
                        <div className="grid gap-1.5">
                          <Label htmlFor="outliers" className="font-medium">
                            Handle Outliers
                          </Label>
                          <p className="text-sm text-muted-foreground">Detect and handle extreme values</p>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="advanced">
                <Card>
                  <CardHeader>
                    <CardTitle>Advanced Preprocessing Options</CardTitle>
                    <CardDescription>Select additional preprocessing techniques</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid gap-4">
                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="normalization"
                          checked={advancedOptions.normalization}
                          onCheckedChange={() => handleAdvancedOptionChange("normalization")}
                        />
                        <div className="grid gap-1.5">
                          <Label htmlFor="normalization" className="font-medium">
                            Normalization
                          </Label>
                          <p className="text-sm text-muted-foreground">Scale features to a range between 0 and 1</p>
                        </div>
                      </div>

                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="scaling"
                          checked={advancedOptions.scaling}
                          onCheckedChange={() => handleAdvancedOptionChange("scaling")}
                        />
                        <div className="grid gap-1.5">
                          <Label htmlFor="scaling" className="font-medium">
                            Scaling
                          </Label>
                          <p className="text-sm text-muted-foreground">
                            Standardize features to have mean=0 and variance=1
                          </p>
                        </div>
                      </div>

                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="feature-engineering"
                          checked={advancedOptions.featureEngineering}
                          onCheckedChange={() => handleAdvancedOptionChange("featureEngineering")}
                        />
                        <div className="grid gap-1.5">
                          <Label htmlFor="feature-engineering" className="font-medium">
                            Feature Engineering
                          </Label>
                          <p className="text-sm text-muted-foreground">Create new features from existing ones</p>
                        </div>
                      </div>

                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="dimensionality-reduction"
                          checked={advancedOptions.dimensionalityReduction}
                          onCheckedChange={() => handleAdvancedOptionChange("dimensionalityReduction")}
                        />
                        <div className="grid gap-1.5">
                          <Label htmlFor="dimensionality-reduction" className="font-medium">
                            Dimensionality Reduction
                          </Label>
                          <p className="text-sm text-muted-foreground">
                            Reduce the number of features using PCA or other techniques
                          </p>
                        </div>
                      </div>

                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="data-compression"
                          checked={advancedOptions.dataCompression}
                          onCheckedChange={() => handleAdvancedOptionChange("dataCompression")}
                        />
                        <div className="grid gap-1.5">
                          <Label htmlFor="data-compression" className="font-medium">
                            Data Compression
                          </Label>
                          <p className="text-sm text-muted-foreground">Compress data to reduce storage requirements</p>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>

            <Card>
              <CardHeader>
                <CardTitle>Project Requirements</CardTitle>
                <CardDescription>Describe your project goals and requirements</CardDescription>
              </CardHeader>
              <CardContent>
                <Textarea
                  placeholder="Describe why you are preprocessing this data and what your analysis goals are..."
                  value={requirements}
                  onChange={(e) => setRequirements(e.target.value)}
                  className="min-h-[100px]"
                />
              </CardContent>
            </Card>

            {isProcessing && (
              <Card>
                <CardContent className="pt-6">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Processing...</span>
                      <span className="text-sm text-muted-foreground">{progress}%</span>
                    </div>
                    <Progress value={progress} />
                  </div>
                </CardContent>
              </Card>
            )}

            <div className="flex justify-end gap-4">
              <Button variant="outline" onClick={() => router.push("/upload")}>
                Back to Upload
              </Button>
              <Button onClick={handleProcessing} disabled={isProcessing} className="gap-2">
                {isProcessing ? (
                  <>
                    <RefreshCw className="h-4 w-4 animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    Process and Continue
                    <ArrowRight className="h-4 w-4" />
                  </>
                )}
              </Button>
            </div>
          </div>
        </div>
      </div>
    </PageLayout>
  )
}
