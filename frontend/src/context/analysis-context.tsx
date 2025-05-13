"use client"

import type React from "react"
import { createContext, useContext, useState, useEffect } from "react"
import {
  dataProcessingService,
  type DataFile,
  type AnalysisResult,
  type Visualization,
  type PreprocessingStep,
  type ModelResult,
} from "../services/data-processing"

// Define the context type
interface AnalysisContextType {
  currentData: DataFile | null
  analysisResult: AnalysisResult | null
  processingStatus: string
  uploadAndProcessData: (file: File) => Promise<void>
  visualizations: Visualization[]
  preprocessingSteps: PreprocessingStep[]
  modelResults: ModelResult[]
  registerVisualization: (visualization: Visualization) => void
  getReportContent: () => string
}

// Create the context with default values
const AnalysisContext = createContext<AnalysisContextType>({
  currentData: null,
  analysisResult: null,
  processingStatus: "idle",
  uploadAndProcessData: async () => {},
  visualizations: [],
  preprocessingSteps: [],
  modelResults: [],
  registerVisualization: () => {},
  getReportContent: () => "",
})

// Custom hook to use the analysis context
export const useAnalysis = () => useContext(AnalysisContext)

// Provider component
export const AnalysisProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [currentData, setCurrentData] = useState<DataFile | null>(null)
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null)
  const [processingStatus, setProcessingStatus] = useState<string>("idle")
  const [visualizations, setVisualizations] = useState<Visualization[]>([])
  const [preprocessingSteps, setPreprocessingSteps] = useState<PreprocessingStep[]>([])
  const [modelResults, setModelResults] = useState<ModelResult[]>([])

  // Register a new visualization
  const registerVisualization = (visualization: Visualization) => {
    setVisualizations((prev) => {
      // Check if visualization with this ID already exists
      const exists = prev.some((v) => v.id === visualization.id)
      if (exists) {
        // Replace the existing visualization
        return prev.map((v) => (v.id === visualization.id ? visualization : v))
      } else {
        // Add the new visualization
        return [...prev, visualization]
      }
    })
  }

  // Upload and process data
  const uploadAndProcessData = async (file: File) => {
    try {
      // Use the mock function for now (replace with real API call when backend is ready)
      await dataProcessingService.mockProcessData(file)
    } catch (error) {
      console.error("Error processing data:", error)
      throw error
    }
  }

  // Generate report content based on analysis results
  const getReportContent = (): string => {
    if (!analysisResult) return ""

    const { summary, preprocessingSteps, modelResults } = analysisResult

    // Generate report content
    let content = `# Data Analysis Report\n\n`
    content += `## Dataset Summary\n\n`
    content += `- **Rows**: ${summary.rowCount}\n`
    content += `- **Columns**: ${summary.columnCount}\n`
    content += `- **Missing Values**: ${summary.missingValues}\n`
    content += `- **Duplicate Rows**: ${summary.duplicateRows}\n\n`

    content += `### Data Types\n\n`
    Object.entries(summary.dataTypes).forEach(([column, type]) => {
      content += `- **${column}**: ${type}\n`
    })

    content += `\n## Preprocessing Steps\n\n`
    preprocessingSteps.forEach((step) => {
      content += `### ${step.name}\n\n`
      content += `${step.description}\n\n`
      content += `Applied to: ${step.appliedTo.join(", ")}\n\n`
      content += `Result: ${step.result}\n\n`
    })

    content += `\n## Model Results\n\n`
    modelResults.forEach((model) => {
      content += `### ${model.name}\n\n`
      content += `Type: ${model.type}\n\n`
      content += `${model.description}\n\n`

      content += `#### Metrics\n\n`
      Object.entries(model.metrics).forEach(([metric, value]) => {
        content += `- **${metric}**: ${value}\n`
      })

      content += `\n#### Parameters\n\n`
      Object.entries(model.parameters).forEach(([param, value]) => {
        content += `- **${param}**: ${value}\n`
      })

      content += `\n`
    })

    return content
  }

  // Listen for status updates from the data processing service
  useEffect(() => {
    const handleStatusUpdate = (status: string, data?: any) => {
      setProcessingStatus(status)

      if (status === "completed" && data) {
        setAnalysisResult(data)
        setCurrentData(dataProcessingService.getCurrentData())
        setVisualizations(data.visualizations || [])
        setPreprocessingSteps(data.preprocessingSteps || [])
        setModelResults(data.modelResults || [])
      }
    }

    dataProcessingService.addListener(handleStatusUpdate)

    return () => {
      dataProcessingService.removeListener(handleStatusUpdate)
    }
  }, [])

  return (
    <AnalysisContext.Provider
      value={{
        currentData,
        analysisResult,
        processingStatus,
        uploadAndProcessData,
        visualizations,
        preprocessingSteps,
        modelResults,
        registerVisualization,
        getReportContent,
      }}
    >
      {children}
    </AnalysisContext.Provider>
  )
}
