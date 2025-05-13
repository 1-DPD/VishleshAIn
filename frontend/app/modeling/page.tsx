"use client"

import { useState } from "react"
import PageLayout from "../page-layout"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { useToast } from "@/hooks/use-toast"
import { useRouter } from "next/navigation"
import { ArrowRight, RefreshCw, CheckCircle2, XCircle, AlertCircle } from "lucide-react"
import { Progress } from "@/components/ui/progress"
import {
  ResponsiveContainer,
  LineChart as RechartsLineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Legend,
  BarChart,
  Bar,
} from "recharts"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"

// Sample data for model performance
const performanceData = [
  { name: "Linear Regression", accuracy: 0.82, precision: 0.8, recall: 0.85, f1: 0.82 },
  { name: "Random Forest", accuracy: 0.89, precision: 0.88, recall: 0.87, f1: 0.87 },
  { name: "SVM", accuracy: 0.85, precision: 0.84, recall: 0.83, f1: 0.83 },
  { name: "Neural Network", accuracy: 0.91, precision: 0.9, recall: 0.89, f1: 0.89 },
]

// Sample data for prediction vs actual
const predictionData = [
  { name: "Sample 1", actual: 45, predicted: 42 },
  { name: "Sample 2", actual: 58, predicted: 55 },
  { name: "Sample 3", actual: 75, predicted: 78 },
  { name: "Sample 4", actual: 34, predicted: 30 },
  { name: "Sample 5", actual: 62, predicted: 65 },
  { name: "Sample 6", actual: 90, predicted: 88 },
  { name: "Sample 7", actual: 55, predicted: 57 },
]

export default function ModelingPage() {
  const [selectedModel, setSelectedModel] = useState("random-forest")
  const [trainTestSplit, setTrainTestSplit] = useState([80])
  const [isTraining, setIsTraining] = useState(false)
  const [progress, setProgress] = useState(0)
  const [modelTrained, setModelTrained] = useState(false)
  const { toast } = useToast()
  const router = useRouter()

  const handleTrainModel = async () => {
    setIsTraining(true)
    setModelTrained(false)

    // Simulate training with progress updates
    for (let i = 0; i <= 100; i += 5) {
      await new Promise((resolve) => setTimeout(resolve, 100))
      setProgress(i)
    }

    try {
      toast({
        title: "Model training complete",
        description: "Your model has been successfully trained and evaluated.",
      })

      setModelTrained(true)
    } catch (error) {
      toast({
        title: "Model training failed",
        description: "There was an error training your model. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsTraining(false)
      setProgress(0)
    }
  }

  const handleContinue = () => {
    if (!modelTrained) {
      toast({
        title: "Model not trained",
        description: "Please train your model before continuing.",
        variant: "destructive",
      })
      return
    }

    router.push("/reports")
  }

  return (
    <PageLayout>
      <div className="container py-10">
        <div className="mx-auto max-w-6xl">
          <div className="mb-8 text-center">
            <h1 className="text-3xl font-bold mb-2">Data Modeling</h1>
            <p className="text-muted-foreground">Build and evaluate machine learning models</p>
          </div>

          <div className="grid gap-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <Card className="md:col-span-1">
                <CardHeader>
                  <CardTitle>Model Selection</CardTitle>
                  <CardDescription>Choose a model for your data</CardDescription>
                </CardHeader>
                <CardContent>
                  <RadioGroup value={selectedModel} onValueChange={setSelectedModel} className="space-y-4">
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="linear-regression" id="linear-regression" />
                      <Label htmlFor="linear-regression" className="font-medium">
                        Linear Regression
                      </Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="random-forest" id="random-forest" />
                      <Label htmlFor="random-forest" className="font-medium">
                        Random Forest
                      </Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="svm" id="svm" />
                      <Label htmlFor="svm" className="font-medium">
                        Support Vector Machine
                      </Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="neural-network" id="neural-network" />
                      <Label htmlFor="neural-network" className="font-medium">
                        Neural Network
                      </Label>
                    </div>
                  </RadioGroup>
                </CardContent>
              </Card>

              <Card className="md:col-span-2">
                <CardHeader>
                  <CardTitle>Model Parameters</CardTitle>
                  <CardDescription>Configure model parameters</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-6">
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <Label>Train/Test Split</Label>
                        <span className="text-sm text-muted-foreground">
                          {trainTestSplit}% / {100 - trainTestSplit}%
                        </span>
                      </div>
                      <Slider value={trainTestSplit} onValueChange={setTrainTestSplit} min={50} max={90} step={5} />
                    </div>

                    {selectedModel === "random-forest" && (
                      <>
                        <div className="space-y-2">
                          <div className="flex justify-between">
                            <Label>Number of Trees</Label>
                            <span className="text-sm text-muted-foreground">100</span>
                          </div>
                          <Slider defaultValue={[100]} min={10} max={500} step={10} disabled={isTraining} />
                        </div>
                        <div className="space-y-2">
                          <div className="flex justify-between">
                            <Label>Max Depth</Label>
                            <span className="text-sm text-muted-foreground">10</span>
                          </div>
                          <Slider defaultValue={[10]} min={1} max={30} step={1} disabled={isTraining} />
                        </div>
                      </>
                    )}

                    {selectedModel === "neural-network" && (
                      <>
                        <div className="space-y-2">
                          <div className="flex justify-between">
                            <Label>Hidden Layers</Label>
                            <span className="text-sm text-muted-foreground">2</span>
                          </div>
                          <Slider defaultValue={[2]} min={1} max={5} step={1} disabled={isTraining} />
                        </div>
                        <div className="space-y-2">
                          <div className="flex justify-between">
                            <Label>Neurons per Layer</Label>
                            <span className="text-sm text-muted-foreground">64</span>
                          </div>
                          <Slider defaultValue={[64]} min={8} max={128} step={8} disabled={isTraining} />
                        </div>
                      </>
                    )}

                    {selectedModel === "svm" && (
                      <>
                        <div className="space-y-2">
                          <div className="flex justify-between">
                            <Label>C (Regularization)</Label>
                            <span className="text-sm text-muted-foreground">1.0</span>
                          </div>
                          <Slider defaultValue={[1]} min={0.1} max={10} step={0.1} disabled={isTraining} />
                        </div>
                        <div className="space-y-2">
                          <div className="flex justify-between">
                            <Label>Kernel</Label>
                          </div>
                          <RadioGroup defaultValue="rbf" className="flex space-x-4">
                            <div className="flex items-center space-x-2">
                              <RadioGroupItem value="linear" id="kernel-linear" />
                              <Label htmlFor="kernel-linear">Linear</Label>
                            </div>
                            <div className="flex items-center space-x-2">
                              <RadioGroupItem value="rbf" id="kernel-rbf" />
                              <Label htmlFor="kernel-rbf">RBF</Label>
                            </div>
                            <div className="flex items-center space-x-2">
                              <RadioGroupItem value="poly" id="kernel-poly" />
                              <Label htmlFor="kernel-poly">Polynomial</Label>
                            </div>
                          </RadioGroup>
                        </div>
                      </>
                    )}

                    {selectedModel === "linear-regression" && (
                      <>
                        <div className="space-y-2">
                          <div className="flex justify-between">
                            <Label>Regularization</Label>
                          </div>
                          <RadioGroup defaultValue="none" className="flex space-x-4">
                            <div className="flex items-center space-x-2">
                              <RadioGroupItem value="none" id="reg-none" />
                              <Label htmlFor="reg-none">None</Label>
                            </div>
                            <div className="flex items-center space-x-2">
                              <RadioGroupItem value="l1" id="reg-l1" />
                              <Label htmlFor="reg-l1">L1 (Lasso)</Label>
                            </div>
                            <div className="flex items-center space-x-2">
                              <RadioGroupItem value="l2" id="reg-l2" />
                              <Label htmlFor="reg-l2">L2 (Ridge)</Label>
                            </div>
                          </RadioGroup>
                        </div>
                        <div className="space-y-2">
                          <div className="flex justify-between">
                            <Label>Alpha</Label>
                            <span className="text-sm text-muted-foreground">0.01</span>
                          </div>
                          <Slider defaultValue={[0.01]} min={0.001} max={1} step={0.001} disabled={isTraining} />
                        </div>
                      </>
                    )}
                  </div>
                </CardContent>
                <CardFooter>
                  <Button onClick={handleTrainModel} disabled={isTraining} className="w-full gap-2">
                    {isTraining ? (
                      <>
                        <RefreshCw className="h-4 w-4 animate-spin" />
                        Training Model...
                      </>
                    ) : (
                      <>Train Model</>
                    )}
                  </Button>
                </CardFooter>
              </Card>
            </div>

            {isTraining && (
              <Card>
                <CardContent className="pt-6">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Training model...</span>
                      <span className="text-sm text-muted-foreground">{progress}%</span>
                    </div>
                    <Progress value={progress} />
                  </div>
                </CardContent>
              </Card>
            )}

            {modelTrained && (
              <>
                <Card>
                  <CardHeader>
                    <CardTitle>Model Performance</CardTitle>
                    <CardDescription>Evaluation metrics for the trained model</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="h-80">
                      <ChartContainer
                        config={{
                          accuracy: {
                            label: "Accuracy",
                            color: "hsl(var(--chart-1))",
                          },
                          precision: {
                            label: "Precision",
                            color: "hsl(var(--chart-2))",
                          },
                          recall: {
                            label: "Recall",
                            color: "hsl(var(--chart-3))",
                          },
                          f1: {
                            label: "F1 Score",
                            color: "hsl(var(--chart-4))",
                          },
                        }}
                      >
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={performanceData}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="name" />
                            <YAxis domain={[0, 1]} />
                            <ChartTooltip content={<ChartTooltipContent />} />
                            <Legend />
                            <Bar dataKey="accuracy" fill="var(--color-accuracy)" />
                            <Bar dataKey="precision" fill="var(--color-precision)" />
                            <Bar dataKey="recall" fill="var(--color-recall)" />
                            <Bar dataKey="f1" fill="var(--color-f1)" />
                          </BarChart>
                        </ResponsiveContainer>
                      </ChartContainer>
                    </div>
                  </CardContent>
                </Card>

                <Tabs defaultValue="predictions">
                  <TabsList className="grid w-full grid-cols-3">
                    <TabsTrigger value="predictions">Predictions</TabsTrigger>
                    <TabsTrigger value="feature-importance">Feature Importance</TabsTrigger>
                    <TabsTrigger value="confusion-matrix">Confusion Matrix</TabsTrigger>
                  </TabsList>

                  <TabsContent value="predictions">
                    <Card>
                      <CardHeader>
                        <CardTitle>Actual vs Predicted Values</CardTitle>
                        <CardDescription>Comparison of actual and predicted values</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="h-80">
                          <ChartContainer
                            config={{
                              actual: {
                                label: "Actual",
                                color: "hsl(var(--chart-1))",
                              },
                              predicted: {
                                label: "Predicted",
                                color: "hsl(var(--chart-2))",
                              },
                            }}
                          >
                            <ResponsiveContainer width="100%" height="100%">
                              <RechartsLineChart data={predictionData}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="name" />
                                <YAxis />
                                <ChartTooltip content={<ChartTooltipContent />} />
                                <Legend />
                                <Line
                                  type="monotone"
                                  dataKey="actual"
                                  stroke="var(--color-actual)"
                                  activeDot={{ r: 8 }}
                                />
                                <Line
                                  type="monotone"
                                  dataKey="predicted"
                                  stroke="var(--color-predicted)"
                                  activeDot={{ r: 8 }}
                                />
                              </RechartsLineChart>
                            </ResponsiveContainer>
                          </ChartContainer>
                        </div>
                      </CardContent>
                    </Card>
                  </TabsContent>

                  <TabsContent value="feature-importance">
                    <Card>
                      <CardHeader>
                        <CardTitle>Feature Importance</CardTitle>
                        <CardDescription>Relative importance of each feature in the model</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="h-80">
                          <ChartContainer
                            config={{
                              importance: {
                                label: "Importance",
                                color: "hsl(var(--chart-3))",
                              },
                            }}
                          >
                            <ResponsiveContainer width="100%" height="100%">
                              <BarChart
                                data={[
                                  { name: "Feature 1", importance: 0.35 },
                                  { name: "Feature 2", importance: 0.25 },
                                  { name: "Feature 3", importance: 0.2 },
                                  { name: "Feature 4", importance: 0.12 },
                                  { name: "Feature 5", importance: 0.08 },
                                ]}
                                layout="vertical"
                              >
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis type="number" />
                                <YAxis dataKey="name" type="category" />
                                <ChartTooltip content={<ChartTooltipContent />} />
                                <Legend />
                                <Bar dataKey="importance" fill="var(--color-importance)" />
                              </BarChart>
                            </ResponsiveContainer>
                          </ChartContainer>
                        </div>
                      </CardContent>
                    </Card>
                  </TabsContent>

                  <TabsContent value="confusion-matrix">
                    <Card>
                      <CardHeader>
                        <CardTitle>Confusion Matrix</CardTitle>
                        <CardDescription>Visualization of model predictions vs actual values</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="grid grid-cols-2 gap-4">
                          <div className="flex flex-col items-center justify-center p-6 border rounded-md bg-green-50 dark:bg-green-950">
                            <span className="text-3xl font-bold text-green-600 dark:text-green-400">85</span>
                            <span className="text-sm text-muted-foreground">True Positive</span>
                          </div>
                          <div className="flex flex-col items-center justify-center p-6 border rounded-md bg-red-50 dark:bg-red-950">
                            <span className="text-3xl font-bold text-red-600 dark:text-red-400">10</span>
                            <span className="text-sm text-muted-foreground">False Positive</span>
                          </div>
                          <div className="flex flex-col items-center justify-center p-6 border rounded-md bg-red-50 dark:bg-red-950">
                            <span className="text-3xl font-bold text-red-600 dark:text-red-400">15</span>
                            <span className="text-sm text-muted-foreground">False Negative</span>
                          </div>
                          <div className="flex flex-col items-center justify-center p-6 border rounded-md bg-green-50 dark:bg-green-950">
                            <span className="text-3xl font-bold text-green-600 dark:text-green-400">90</span>
                            <span className="text-sm text-muted-foreground">True Negative</span>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </TabsContent>
                </Tabs>

                <Card>
                  <CardHeader>
                    <CardTitle>Model Insights</CardTitle>
                    <CardDescription>Key findings from the model evaluation</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ul className="space-y-2">
                      <li className="flex items-start gap-2">
                        <div className="mt-0.5 rounded-full bg-green-100 p-1 dark:bg-green-900">
                          <CheckCircle2 className="h-4 w-4 text-green-600 dark:text-green-400" />
                        </div>
                        <span>Model achieved 89% accuracy on the test set</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <div className="mt-0.5 rounded-full bg-green-100 p-1 dark:bg-green-900">
                          <CheckCircle2 className="h-4 w-4 text-green-600 dark:text-green-400" />
                        </div>
                        <span>Feature 1 is the most important predictor with 35% importance</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <div className="mt-0.5 rounded-full bg-yellow-100 p-1 dark:bg-yellow-900">
                          <AlertCircle className="h-4 w-4 text-yellow-600 dark:text-yellow-400" />
                        </div>
                        <span>Model tends to slightly underpredict high values</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <div className="mt-0.5 rounded-full bg-red-100 p-1 dark:bg-red-900">
                          <XCircle className="h-4 w-4 text-red-600 dark:text-red-400" />
                        </div>
                        <span>15% false negative rate may require further optimization</span>
                      </li>
                    </ul>
                  </CardContent>
                </Card>
              </>
            )}

            <div className="flex justify-end gap-4">
              <Button variant="outline" onClick={() => router.push("/analysis")}>
                Back to Analysis
              </Button>
              <Button onClick={handleContinue} disabled={!modelTrained} className="gap-2">
                Continue to Reports
                <ArrowRight className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
      </div>
    </PageLayout>
  )
}
