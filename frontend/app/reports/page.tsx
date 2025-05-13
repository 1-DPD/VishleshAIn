"use client"

import { useState } from "react"
import PageLayout from "../page-layout"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { useToast } from "@/hooks/use-toast"
import { useRouter } from "next/navigation"
import { FileText, Download, ArrowRight, RefreshCw, BarChart2, LineChart, PieChart, ScatterChart } from "lucide-react"
import { Progress } from "@/components/ui/progress"

export default function ReportsPage() {
  const [isGenerating, setIsGenerating] = useState(false)
  const [progress, setProgress] = useState(0)
  const [reportGenerated, setReportGenerated] = useState(false)
  const { toast } = useToast()
  const router = useRouter()

  const handleGenerateReport = async () => {
    setIsGenerating(true)
    setReportGenerated(false)

    // Simulate report generation with progress updates
    for (let i = 0; i <= 100; i += 5) {
      await new Promise((resolve) => setTimeout(resolve, 150))
      setProgress(i)
    }

    try {
      toast({
        title: "Report generation complete",
        description: "Your comprehensive report has been successfully generated.",
      })

      setReportGenerated(true)
    } catch (error) {
      toast({
        title: "Report generation failed",
        description: "There was an error generating your report. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsGenerating(false)
      setProgress(0)
    }
  }

  const handleDownload = () => {
    toast({
      title: "Download started",
      description: "Your report and associated files are being prepared for download.",
    })

    // Simulate download
    setTimeout(() => {
      toast({
        title: "Download complete",
        description: "Your files have been downloaded successfully.",
      })
    }, 2000)
  }

  return (
    <PageLayout>
      <div className="container py-10">
        <div className="mx-auto max-w-6xl">
          <div className="mb-8 text-center">
            <h1 className="text-3xl font-bold mb-2">Report Generation</h1>
            <p className="text-muted-foreground">Generate comprehensive reports with visualizations and insights</p>
          </div>

          <div className="grid gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Report Configuration</CardTitle>
                <CardDescription>Configure your report settings</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid gap-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <h3 className="text-lg font-medium">Report Sections</h3>
                      <div className="space-y-1">
                        <div className="flex items-center gap-2">
                          <input type="checkbox" id="title-page" className="rounded" defaultChecked />
                          <label htmlFor="title-page">Title Page</label>
                        </div>
                        <div className="flex items-center gap-2">
                          <input type="checkbox" id="index" className="rounded" defaultChecked />
                          <label htmlFor="index">Index</label>
                        </div>
                        <div className="flex items-center gap-2">
                          <input type="checkbox" id="abstract" className="rounded" defaultChecked />
                          <label htmlFor="abstract">Abstract</label>
                        </div>
                        <div className="flex items-center gap-2">
                          <input type="checkbox" id="introduction" className="rounded" defaultChecked />
                          <label htmlFor="introduction">Introduction</label>
                        </div>
                        <div className="flex items-center gap-2">
                          <input type="checkbox" id="methodology" className="rounded" defaultChecked />
                          <label htmlFor="methodology">Methodology</label>
                        </div>
                        <div className="flex items-center gap-2">
                          <input type="checkbox" id="results" className="rounded" defaultChecked />
                          <label htmlFor="results">Results</label>
                        </div>
                        <div className="flex items-center gap-2">
                          <input type="checkbox" id="discussion" className="rounded" defaultChecked />
                          <label htmlFor="discussion">Discussion</label>
                        </div>
                        <div className="flex items-center gap-2">
                          <input type="checkbox" id="conclusion" className="rounded" defaultChecked />
                          <label htmlFor="conclusion">Conclusion</label>
                        </div>
                        <div className="flex items-center gap-2">
                          <input type="checkbox" id="references" className="rounded" defaultChecked />
                          <label htmlFor="references">References</label>
                        </div>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <h3 className="text-lg font-medium">Visualizations</h3>
                      <div className="space-y-1">
                        <div className="flex items-center gap-2">
                          <input type="checkbox" id="data-distribution" className="rounded" defaultChecked />
                          <label htmlFor="data-distribution">Data Distribution Charts</label>
                        </div>
                        <div className="flex items-center gap-2">
                          <input type="checkbox" id="correlation-matrix" className="rounded" defaultChecked />
                          <label htmlFor="correlation-matrix">Correlation Matrix</label>
                        </div>
                        <div className="flex items-center gap-2">
                          <input type="checkbox" id="feature-importance" className="rounded" defaultChecked />
                          <label htmlFor="feature-importance">Feature Importance</label>
                        </div>
                        <div className="flex items-center gap-2">
                          <input type="checkbox" id="model-performance" className="rounded" defaultChecked />
                          <label htmlFor="model-performance">Model Performance Metrics</label>
                        </div>
                        <div className="flex items-center gap-2">
                          <input type="checkbox" id="prediction-vs-actual" className="rounded" defaultChecked />
                          <label htmlFor="prediction-vs-actual">Prediction vs Actual</label>
                        </div>
                        <div className="flex items-center gap-2">
                          <input type="checkbox" id="confusion-matrix" className="rounded" defaultChecked />
                          <label htmlFor="confusion-matrix">Confusion Matrix</label>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <h3 className="text-lg font-medium">Report Format</h3>
                    <div className="flex gap-4">
                      <div className="flex items-center gap-2">
                        <input type="radio" id="format-pdf" name="format" className="rounded" defaultChecked />
                        <label htmlFor="format-pdf">PDF</label>
                      </div>
                      <div className="flex items-center gap-2">
                        <input type="radio" id="format-docx" name="format" className="rounded" />
                        <label htmlFor="format-docx">DOCX</label>
                      </div>
                      <div className="flex items-center gap-2">
                        <input type="radio" id="format-html" name="format" className="rounded" />
                        <label htmlFor="format-html">HTML</label>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
              <CardFooter>
                <Button onClick={handleGenerateReport} disabled={isGenerating} className="w-full gap-2">
                  {isGenerating ? (
                    <>
                      <RefreshCw className="h-4 w-4 animate-spin" />
                      Generating Report...
                    </>
                  ) : (
                    <>Generate Report</>
                  )}
                </Button>
              </CardFooter>
            </Card>

            {isGenerating && (
              <Card>
                <CardContent className="pt-6">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Generating report...</span>
                      <span className="text-sm text-muted-foreground">{progress}%</span>
                    </div>
                    <Progress value={progress} />
                  </div>
                </CardContent>
              </Card>
            )}

            {reportGenerated && (
              <>
                <Card>
                  <CardHeader>
                    <CardTitle>Report Preview</CardTitle>
                    <CardDescription>Preview of your generated report</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <Tabs defaultValue="title">
                      <TabsList className="grid w-full grid-cols-5">
                        <TabsTrigger value="title">Title</TabsTrigger>
                        <TabsTrigger value="abstract">Abstract</TabsTrigger>
                        <TabsTrigger value="methodology">Methodology</TabsTrigger>
                        <TabsTrigger value="results">Results</TabsTrigger>
                        <TabsTrigger value="conclusion">Conclusion</TabsTrigger>
                      </TabsList>

                      <TabsContent value="title" className="p-4 border rounded-md mt-4">
                        <div className="text-center space-y-4">
                          <h1 className="text-2xl font-bold">Data Analysis Report</h1>
                          <h2 className="text-xl">Exploratory Analysis and Predictive Modeling</h2>
                          <p className="text-muted-foreground">Generated by VishleshAIn</p>
                          <p className="text-muted-foreground">May 12, 2025</p>
                        </div>
                      </TabsContent>

                      <TabsContent value="abstract" className="p-4 border rounded-md mt-4">
                        <h2 className="text-xl font-bold mb-4">Abstract</h2>
                        <p className="mb-2">
                          This report presents a comprehensive analysis of the dataset, including exploratory data
                          analysis, preprocessing techniques, and predictive modeling. The analysis revealed significant
                          patterns and relationships within the data, with Feature 1 showing the strongest correlation
                          with the target variable.
                        </p>
                        <p>
                          A Random Forest model was trained on the preprocessed data, achieving 89% accuracy on the test
                          set. The model demonstrated strong predictive performance, with key insights highlighting the
                          importance of Features 1, 2, and 3 in making accurate predictions. This report details the
                          methodology, findings, and recommendations based on the analysis.
                        </p>
                      </TabsContent>

                      <TabsContent value="methodology" className="p-4 border rounded-md mt-4">
                        <h2 className="text-xl font-bold mb-4">Methodology</h2>
                        <p className="mb-2">
                          The analysis followed a structured approach, beginning with data preprocessing to handle
                          missing values, remove duplicates, encode categorical variables, fix inconsistent data, and
                          handle outliers. Advanced preprocessing techniques including normalization and feature
                          engineering were applied to optimize the dataset for modeling.
                        </p>
                        <p className="mb-2">
                          Exploratory data analysis was conducted to understand the distribution of variables, identify
                          correlations, and detect patterns. Visualizations including bar charts, scatter plots, and
                          correlation matrices were used to gain insights into the data structure.
                        </p>
                        <p>
                          For predictive modeling, a Random Forest algorithm was selected based on its performance
                          during preliminary testing. The model was trained on 80% of the data and evaluated on the
                          remaining 20%. Hyperparameters were optimized to achieve the best performance, with 100 trees
                          and a maximum depth of 10.
                        </p>
                      </TabsContent>

                      <TabsContent value="results" className="p-4 border rounded-md mt-4">
                        <h2 className="text-xl font-bold mb-4">Results</h2>
                        <p className="mb-2">
                          The exploratory data analysis revealed several key insights. Feature 1 and Feature 2 showed a
                          strong positive correlation (0.75), while Feature 3 demonstrated negative correlations with
                          both Feature 1 and Feature 2. The data distribution across categories was relatively balanced,
                          with Categories B and C showing similar patterns.
                        </p>
                        <p className="mb-2">
                          The Random Forest model achieved an accuracy of 89% on the test set, with precision of 88%,
                          recall of 87%, and an F1 score of 87%. Feature importance analysis identified Feature 1 as the
                          most significant predictor (35% importance), followed by Feature 2 (25%) and Feature 3 (20%).
                        </p>
                        <p>
                          The confusion matrix revealed 85 true positives, 90 true negatives, 10 false positives, and 15
                          false negatives. The model demonstrated strong performance overall, though it showed a slight
                          tendency to underpredict high values.
                        </p>
                      </TabsContent>

                      <TabsContent value="conclusion" className="p-4 border rounded-md mt-4">
                        <h2 className="text-xl font-bold mb-4">Conclusion</h2>
                        <p className="mb-2">
                          This analysis has successfully identified key patterns and relationships within the dataset,
                          providing valuable insights for decision-making. The Random Forest model demonstrated strong
                          predictive performance, with an accuracy of 89% and balanced precision and recall metrics.
                        </p>
                        <p className="mb-2">
                          Feature 1 emerged as the most important predictor, suggesting that future data collection and
                          analysis efforts should prioritize this variable. The strong correlation between Features 1
                          and 2 indicates a potential underlying relationship that merits further investigation.
                        </p>
                        <p>
                          While the model performs well overall, the 15% false negative rate suggests room for
                          improvement. Future work could explore ensemble methods or deep learning approaches to
                          potentially enhance predictive accuracy, particularly for high-value predictions where the
                          current model tends to underpredict.
                        </p>
                      </TabsContent>
                    </Tabs>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Visualizations</CardTitle>
                    <CardDescription>Key visualizations included in the report</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="border rounded-md p-4 flex flex-col items-center">
                        <div className="rounded-full bg-primary/10 p-4 mb-4">
                          <BarChart2 className="h-8 w-8 text-primary" />
                        </div>
                        <h3 className="text-lg font-medium mb-2">Data Distribution</h3>
                        <p className="text-sm text-muted-foreground text-center">
                          Bar chart showing the distribution of values across categories
                        </p>
                      </div>

                      <div className="border rounded-md p-4 flex flex-col items-center">
                        <div className="rounded-full bg-primary/10 p-4 mb-4">
                          <PieChart className="h-8 w-8 text-primary" />
                        </div>
                        <h3 className="text-lg font-medium mb-2">Category Distribution</h3>
                        <p className="text-sm text-muted-foreground text-center">
                          Pie chart showing the proportion of data in each category
                        </p>
                      </div>

                      <div className="border rounded-md p-4 flex flex-col items-center">
                        <div className="rounded-full bg-primary/10 p-4 mb-4">
                          <LineChart className="h-8 w-8 text-primary" />
                        </div>
                        <h3 className="text-lg font-medium mb-2">Actual vs Predicted</h3>
                        <p className="text-sm text-muted-foreground text-center">
                          Line chart comparing actual and predicted values
                        </p>
                      </div>

                      <div className="border rounded-md p-4 flex flex-col items-center">
                        <div className="rounded-full bg-primary/10 p-4 mb-4">
                          <ScatterChart className="h-8 w-8 text-primary" />
                        </div>
                        <h3 className="text-lg font-medium mb-2">Feature Relationships</h3>
                        <p className="text-sm text-muted-foreground text-center">
                          Scatter plot showing relationships between features
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Download Options</CardTitle>
                    <CardDescription>Download your report and associated files</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <Button
                        variant="outline"
                        className="h-auto py-6 flex flex-col items-center gap-2"
                        onClick={handleDownload}
                      >
                        <FileText className="h-8 w-8" />
                        <div className="text-center">
                          <p className="font-medium">Full Report</p>
                          <p className="text-xs text-muted-foreground">PDF, 10 pages</p>
                        </div>
                      </Button>

                      <Button
                        variant="outline"
                        className="h-auto py-6 flex flex-col items-center gap-2"
                        onClick={handleDownload}
                      >
                        <BarChart2 className="h-8 w-8" />
                        <div className="text-center">
                          <p className="font-medium">Visualizations</p>
                          <p className="text-xs text-muted-foreground">PNG, 8 files</p>
                        </div>
                      </Button>

                      <Button
                        variant="outline"
                        className="h-auto py-6 flex flex-col items-center gap-2"
                        onClick={handleDownload}
                      >
                        <Download className="h-8 w-8" />
                        <div className="text-center">
                          <p className="font-medium">All Files (ZIP)</p>
                          <p className="text-xs text-muted-foreground">Report, data, visualizations</p>
                        </div>
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              </>
            )}

            <div className="flex justify-end gap-4">
              <Button variant="outline" onClick={() => router.push("/modeling")}>
                Back to Modeling
              </Button>
              <Button onClick={() => router.push("/download")} disabled={!reportGenerated} className="gap-2">
                Continue to Download
                <ArrowRight className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
      </div>
    </PageLayout>
  )
}
