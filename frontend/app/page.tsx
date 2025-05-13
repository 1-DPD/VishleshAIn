import PageLayout from "./page-layout"
import { Button } from "@/components/ui/button"
import Link from "next/link"
import { ArrowRight, BarChart2, Database, FileText, LineChart, Upload } from "lucide-react"
import { Chatbot } from "@/components/chatbot"

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <div className="z-10 max-w-5xl w-full items-center justify-between font-mono text-sm lg:flex">
        <h1 className="text-4xl font-bold">VishleshAIn</h1>
        <p className="fixed left-0 top-0 flex w-full justify-center border-b border-gray-300 bg-gradient-to-b from-zinc-200 pb-6 pt-8 backdrop-blur-2xl dark:border-neutral-800 dark:bg-zinc-800/30 dark:from-inherit lg:static lg:w-auto lg:rounded-xl lg:border lg:bg-gray-200 lg:p-4 lg:dark:bg-zinc-800/30">
          Your Data Analytics Personal Assistant
        </p>
      </div>

      <PageLayout>
        <section className="py-12 md:py-24 lg:py-32 bg-background">
          <div className="container px-4 md:px-6">
            <div className="flex flex-col items-center justify-center space-y-4 text-center">
              <div className="space-y-2">
                <h1 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl lg:text-6xl">
                  Welcome to VishleshAIn
                </h1>
                <p className="mx-auto max-w-[700px] text-gray-500 md:text-xl dark:text-gray-400">
                  Your personal data analytics assistant for preprocessing, analysis, modeling, and visualization.
                </p>
              </div>
              <div className="space-x-4">
                <Link href="/upload">
                  <Button className="gap-1">
                    Get Started <ArrowRight className="h-4 w-4" />
                  </Button>
                </Link>
              </div>
            </div>
          </div>
        </section>

        <section className="py-12 md:py-24 lg:py-32 bg-muted/50">
          <div className="container px-4 md:px-6">
            <div className="grid gap-6 lg:grid-cols-3 lg:gap-12">
              <div className="flex flex-col items-center space-y-4 text-center">
                <div className="rounded-full bg-primary/10 p-4">
                  <Upload className="h-6 w-6 text-primary" />
                </div>
                <div className="space-y-2">
                  <h3 className="text-xl font-bold">Upload Data</h3>
                  <p className="text-gray-500 dark:text-gray-400">
                    Upload data in various formats (CSV, JSON, XML, PDF, DOCX) or extract from URLs.
                  </p>
                </div>
              </div>
              <div className="flex flex-col items-center space-y-4 text-center">
                <div className="rounded-full bg-primary/10 p-4">
                  <Database className="h-6 w-6 text-primary" />
                </div>
                <div className="space-y-2">
                  <h3 className="text-xl font-bold">Preprocess & Clean</h3>
                  <p className="text-gray-500 dark:text-gray-400">
                    Automatically handle missing values, duplicates, encoding, and outliers.
                  </p>
                </div>
              </div>
              <div className="flex flex-col items-center space-y-4 text-center">
                <div className="rounded-full bg-primary/10 p-4">
                  <BarChart2 className="h-6 w-6 text-primary" />
                </div>
                <div className="space-y-2">
                  <h3 className="text-xl font-bold">Analyze & Visualize</h3>
                  <p className="text-gray-500 dark:text-gray-400">
                    Perform exploratory data analysis and generate insightful visualizations.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="py-12 md:py-24 lg:py-32 bg-background">
          <div className="container px-4 md:px-6">
            <div className="grid gap-6 lg:grid-cols-2 lg:gap-12">
              <div className="flex flex-col justify-center space-y-4">
                <div className="space-y-2">
                  <h2 className="text-3xl font-bold tracking-tighter md:text-4xl">Advanced Data Modeling</h2>
                  <p className="text-gray-500 md:text-xl dark:text-gray-400">
                    Build and evaluate machine learning models based on your specific requirements.
                  </p>
                </div>
                <ul className="grid gap-2">
                  <li className="flex items-center gap-2">
                    <LineChart className="h-4 w-4 text-primary" />
                    <span>Supervised and unsupervised learning</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <LineChart className="h-4 w-4 text-primary" />
                    <span>Classification, regression, and clustering</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <LineChart className="h-4 w-4 text-primary" />
                    <span>Model evaluation and optimization</span>
                  </li>
                </ul>
              </div>
              <div className="flex flex-col justify-center space-y-4">
                <div className="space-y-2">
                  <h2 className="text-3xl font-bold tracking-tighter md:text-4xl">Comprehensive Reporting</h2>
                  <p className="text-gray-500 md:text-xl dark:text-gray-400">
                    Generate detailed professional reports with visualizations and insights.
                  </p>
                </div>
                <ul className="grid gap-2">
                  <li className="flex items-center gap-2">
                    <FileText className="h-4 w-4 text-primary" />
                    <span>Professional research paper format</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <FileText className="h-4 w-4 text-primary" />
                    <span>Methodology, results, and limitations</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <FileText className="h-4 w-4 text-primary" />
                    <span>Downloadable reports and visualizations</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </section>
      </PageLayout>

      {/* Include the chatbot component */}
      <Chatbot />
    </main>
  )
}
