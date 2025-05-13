"use client"

import { useState } from "react"
import PageLayout from "../page-layout"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { useToast } from "@/hooks/use-toast"
import { useRouter } from "next/navigation"
import { BarChart2, ArrowRight, RefreshCw } from "lucide-react"
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  PieChart as RechartsPieChart,
  Pie,
  Cell,
  LineChart as RechartsLineChart,
  Line,
  ScatterChart as RechartsScatterChart,
  Scatter,
  ZAxis,
} from "recharts"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"

// Sample data for charts
const barData = [
  { name: "Category A", value: 400 },
  { name: "Category B", value: 300 },
  { name: "Category C", value: 200 },
  { name: "Category D", value: 278 },
  { name: "Category E", value: 189 },
]

const pieData = [
  { name: "Group A", value: 400 },
  { name: "Group B", value: 300 },
  { name: "Group C", value: 300 },
  { name: "Group D", value: 200 },
]

const lineData = [
  { name: "Jan", value: 400 },
  { name: "Feb", value: 300 },
  { name: "Mar", value: 600 },
  { name: "Apr", value: 800 },
  { name: "May", value: 500 },
  { name: "Jun", value: 900 },
  { name: "Jul", value: 1000 },
]

const scatterData = [
  { x: 100, y: 200, z: 200 },
  { x: 120, y: 100, z: 260 },
  { x: 170, y: 300, z: 400 },
  { x: 140, y: 250, z: 280 },
  { x: 150, y: 400, z: 500 },
  { x: 110, y: 280, z: 200 },
]

const COLORS = ["#0088FE", "#00C49F", "#FFBB28", "#FF8042"]

export default function AnalysisPage() {
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const { toast } = useToast()
  const router = useRouter()

  const handleAnalysis = async () => {
    setIsAnalyzing(true)

    try {
      // Simulate analysis
      await new Promise((resolve) => setTimeout(resolve, 2000))

      toast({
        title: "Analysis complete",
        description: "Exploratory data analysis has been completed successfully.",
      })

      // Navigate to modeling page
      router.push("/modeling")
    } catch (error) {
      toast({
        title: "Analysis failed",
        description: "There was an error during the analysis. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsAnalyzing(false)
    }
  }

  return (
    <PageLayout>
      <div className="container py-10">
        <div className="mx-auto max-w-6xl">
          <div className="mb-8 text-center">
            <h1 className="text-3xl font-bold mb-2">Exploratory Data Analysis</h1>
            <p className="text-muted-foreground">
              Explore and understand your data through visualizations and statistics
            </p>
          </div>

          <div className="grid gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Data Summary</CardTitle>
                <CardDescription>Statistical summary of your preprocessed data</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full border-collapse">
                    <thead>
                      <tr className="border-b">
                        <th className="text-left p-2">Column</th>
                        <th className="text-left p-2">Count</th>
                        <th className="text-left p-2">Mean</th>
                        <th className="text-left p-2">Std</th>
                        <th className="text-left p-2">Min</th>
                        <th className="text-left p-2">25%</th>
                        <th className="text-left p-2">50%</th>
                        <th className="text-left p-2">75%</th>
                        <th className="text-left p-2">Max</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr className="border-b">
                        <td className="p-2 font-medium">Feature 1</td>
                        <td className="p-2">1,245</td>
                        <td className="p-2">42.3</td>
                        <td className="p-2">15.7</td>
                        <td className="p-2">10.0</td>
                        <td className="p-2">30.5</td>
                        <td className="p-2">40.2</td>
                        <td className="p-2">55.1</td>
                        <td className="p-2">95.0</td>
                      </tr>
                      <tr className="border-b">
                        <td className="p-2 font-medium">Feature 2</td>
                        <td className="p-2">1,245</td>
                        <td className="p-2">65.8</td>
                        <td className="p-2">20.3</td>
                        <td className="p-2">15.0</td>
                        <td className="p-2">50.2</td>
                        <td className="p-2">68.5</td>
                        <td className="p-2">82.3</td>
                        <td className="p-2">120.0</td>
                      </tr>
                      <tr className="border-b">
                        <td className="p-2 font-medium">Feature 3</td>
                        <td className="p-2">1,245</td>
                        <td className="p-2">28.4</td>
                        <td className="p-2">8.9</td>
                        <td className="p-2">5.0</td>
                        <td className="p-2">22.1</td>
                        <td className="p-2">27.5</td>
                        <td className="p-2">35.8</td>
                        <td className="p-2">60.0</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>

            <Tabs defaultValue="distribution">
              <TabsList className="grid w-full grid-cols-4">
                <TabsTrigger value="distribution">Distribution</TabsTrigger>
                <TabsTrigger value="correlation">Correlation</TabsTrigger>
                <TabsTrigger value="trends">Trends</TabsTrigger>
                <TabsTrigger value="relationships">Relationships</TabsTrigger>
              </TabsList>

              <TabsContent value="distribution">
                <Card>
                  <CardHeader>
                    <CardTitle>Data Distribution</CardTitle>
                    <CardDescription>Distribution of values across categories</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="h-80">
                      <ChartContainer
                        config={{
                          value: {
                            label: "Value",
                            color: "hsl(var(--chart-1))",
                          },
                        }}
                      >
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={barData}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="name" />
                            <YAxis />
                            <ChartTooltip content={<ChartTooltipContent />} />
                            <Legend />
                            <Bar dataKey="value" fill="var(--color-value)" />
                          </BarChart>
                        </ResponsiveContainer>
                      </ChartContainer>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="correlation">
                <Card>
                  <CardHeader>
                    <CardTitle>Category Distribution</CardTitle>
                    <CardDescription>Proportion of data in each category</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="h-80">
                      <ResponsiveContainer width="100%" height="100%">
                        <RechartsPieChart>
                          <Pie
                            data={pieData}
                            cx="50%"
                            cy="50%"
                            labelLine={false}
                            outerRadius={80}
                            fill="#8884d8"
                            dataKey="value"
                            label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                          >
                            {pieData.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                            ))}
                          </Pie>
                          <Tooltip />
                          <Legend />
                        </RechartsPieChart>
                      </ResponsiveContainer>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="trends">
                <Card>
                  <CardHeader>
                    <CardTitle>Time Series Trend</CardTitle>
                    <CardDescription>Value changes over time</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="h-80">
                      <ChartContainer
                        config={{
                          value: {
                            label: "Value",
                            color: "hsl(var(--chart-2))",
                          },
                        }}
                      >
                        <ResponsiveContainer width="100%" height="100%">
                          <RechartsLineChart data={lineData}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="name" />
                            <YAxis />
                            <ChartTooltip content={<ChartTooltipContent />} />
                            <Legend />
                            <Line type="monotone" dataKey="value" stroke="var(--color-value)" activeDot={{ r: 8 }} />
                          </RechartsLineChart>
                        </ResponsiveContainer>
                      </ChartContainer>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="relationships">
                <Card>
                  <CardHeader>
                    <CardTitle>Feature Relationships</CardTitle>
                    <CardDescription>Relationship between different features</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="h-80">
                      <ResponsiveContainer width="100%" height="100%">
                        <RechartsScatterChart>
                          <CartesianGrid />
                          <XAxis type="number" dataKey="x" name="Feature X" unit="" />
                          <YAxis type="number" dataKey="y" name="Feature Y" unit="" />
                          <ZAxis type="number" dataKey="z" range={[60, 400]} name="Feature Z" unit="" />
                          <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                          <Legend />
                          <Scatter name="Features" data={scatterData} fill="#8884d8" />
                        </RechartsScatterChart>
                      </ResponsiveContainer>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Correlation Matrix</CardTitle>
                  <CardDescription>Correlation between different features</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="overflow-x-auto">
                    <table className="w-full border-collapse">
                      <thead>
                        <tr className="border-b">
                          <th className="text-left p-2"></th>
                          <th className="text-left p-2">Feature 1</th>
                          <th className="text-left p-2">Feature 2</th>
                          <th className="text-left p-2">Feature 3</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr className="border-b">
                          <td className="p-2 font-medium">Feature 1</td>
                          <td className="p-2">1.00</td>
                          <td className="p-2">0.75</td>
                          <td className="p-2">-0.32</td>
                        </tr>
                        <tr className="border-b">
                          <td className="p-2 font-medium">Feature 2</td>
                          <td className="p-2">0.75</td>
                          <td className="p-2">1.00</td>
                          <td className="p-2">-0.18</td>
                        </tr>
                        <tr className="border-b">
                          <td className="p-2 font-medium">Feature 3</td>
                          <td className="p-2">-0.32</td>
                          <td className="p-2">-0.18</td>
                          <td className="p-2">1.00</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Key Insights</CardTitle>
                  <CardDescription>Important findings from the exploratory analysis</CardDescription>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-2">
                    <li className="flex items-start gap-2">
                      <div className="mt-0.5 rounded-full bg-primary/10 p-1">
                        <BarChart2 className="h-4 w-4 text-primary" />
                      </div>
                      <span>Strong positive correlation (0.75) between Feature 1 and Feature 2</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <div className="mt-0.5 rounded-full bg-primary/10 p-1">
                        <BarChart2 className="h-4 w-4 text-primary" />
                      </div>
                      <span>Feature 3 shows negative correlation with both Feature 1 and Feature 2</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <div className="mt-0.5 rounded-full bg-primary/10 p-1">
                        <BarChart2 className="h-4 w-4 text-primary" />
                      </div>
                      <span>Category B and Category C have similar distributions</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <div className="mt-0.5 rounded-full bg-primary/10 p-1">
                        <BarChart2 className="h-4 w-4 text-primary" />
                      </div>
                      <span>Time series data shows an increasing trend from January to July</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <div className="mt-0.5 rounded-full bg-primary/10 p-1">
                        <BarChart2 className="h-4 w-4 text-primary" />
                      </div>
                      <span>Outliers detected in Feature 2 may require further investigation</span>
                    </li>
                  </ul>
                </CardContent>
              </Card>
            </div>

            <div className="flex justify-end gap-4">
              <Button variant="outline" onClick={() => router.push("/preprocessing")}>
                Back to Preprocessing
              </Button>
              <Button onClick={handleAnalysis} disabled={isAnalyzing} className="gap-2">
                {isAnalyzing ? (
                  <>
                    <RefreshCw className="h-4 w-4 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    Continue to Modeling
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
