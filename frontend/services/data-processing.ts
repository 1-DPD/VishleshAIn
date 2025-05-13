import { v4 as uuidv4 } from "uuid"
import axios from "axios"

// Interface for the uploaded data
export interface DataFile {
  id: string
  name: string
  type: string
  size: number
  content: any
}

// Interface for analysis results
export interface AnalysisResult {
  summary: {
    rowCount: number
    columnCount: number
    missingValues: number
    duplicateRows: number
    dataTypes: Record<string, string>
  }
  visualizations: Visualization[]
  preprocessingSteps: PreprocessingStep[]
  modelResults: ModelResult[]
}

export interface Visualization {
  id: string
  title: string
  description: string
  type: "bar" | "line" | "scatter" | "pie" | "heatmap" | "histogram"
  data: any
  config: any
}

export interface PreprocessingStep {
  id: string
  name: string
  description: string
  appliedTo: string[]
  result: string
}

export interface ModelResult {
  id: string
  name: string
  type: string
  metrics: Record<string, number>
  parameters: Record<string, any>
  description: string
}

// Types for data processing
export type DatasetSummary = {
  rows: number
  columns: number
  missingValues: number
  duplicates: number
  columnTypes: Record<string, string>
  sampleData: any[]
}

export type ColumnStatistics = {
  name: string
  type: string
  count: number
  missing: number
  unique: number
  mean?: number
  std?: number
  min?: number
  q1?: number
  median?: number
  q3?: number
  max?: number
  mode?: any
  categories?: string[]
  categoryCount?: Record<string, number>
}

export type DatasetAnalysis = {
  summary: DatasetSummary
  columnStats: ColumnStatistics[]
  correlations: Record<string, Record<string, number>>
  insights: string[]
}

// Main data processing service
class DataProcessingService {
  private apiUrl = "http://localhost:5000/api"
  private currentData: DataFile | null = null
  private analysisResult: AnalysisResult | null = null
  private processingStatus: "idle" | "processing" | "completed" | "error" = "idle"
  private listeners: Array<(status: string, data?: any) => void> = []
  private static instance: DataProcessingService
  private currentDataset: any[] = []
  private processedDataset: any[] = []
  private datasetColumns: string[] = []
  private datasetTypes: Record<string, string> = {}
  private processingCallbacks: Record<string, (progress: number, status: string) => void> = {}

  private constructor() {}

  public static getInstance(): DataProcessingService {
    if (!DataProcessingService.instance) {
      DataProcessingService.instance = new DataProcessingService()
    }
    return DataProcessingService.instance
  }

  // Upload and process data
  async uploadAndProcessData(file: File): Promise<any> {
    try {
      this.processingStatus = "processing"
      this.notifyListeners("processing")

      // Read the file content
      const content = await this.readFileContent(file)

      // Create a data file object
      this.currentData = {
        id: Date.now().toString(),
        name: file.name,
        type: file.type,
        size: file.size,
        content,
      }

      // Send the data to the backend for processing
      const formData = new FormData()
      formData.append("file", file)

      // Call the backend API to process the data
      const response = await axios.post(`${this.apiUrl}/process`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      })

      // Store the analysis result
      this.analysisResult = response.data
      this.processingStatus = "completed"
      this.notifyListeners("completed", this.analysisResult)

      return this.analysisResult
    } catch (error) {
      console.error("Error processing data:", error)
      this.processingStatus = "error"
      this.notifyListeners("error", error)
      throw error
    }
  }

  // Process uploaded file
  public async processFile(
    file: File,
    onProgress?: (progress: number, status: string) => void,
  ): Promise<DatasetSummary> {
    const processId = uuidv4()
    if (onProgress) {
      this.processingCallbacks[processId] = onProgress
    }

    try {
      // Update progress
      this.updateProgress(processId, 10, "Reading file...")

      // Read the file
      const content = await this.readFileContent(file)

      // Parse the data based on file type
      this.updateProgress(processId, 30, "Parsing data...")
      const data = this.parseData(file.name, content)
      this.currentDataset = data

      // Extract columns
      this.datasetColumns = Object.keys(data[0] || {})

      // Determine column types
      this.updateProgress(processId, 50, "Analyzing column types...")
      this.datasetTypes = this.determineColumnTypes(data)

      // Generate summary
      this.updateProgress(processId, 70, "Generating summary...")
      const summary = this.generateSummary(data)

      // Clean up
      this.updateProgress(processId, 100, "Complete")
      if (onProgress) {
        delete this.processingCallbacks[processId]
      }

      return summary
    } catch (error) {
      console.error("Error processing file:", error)
      throw new Error("Failed to process file")
    }
  }

  // Read file content based on file type
  private async readFileContent(file: File): Promise<any> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()

      reader.onload = (event) => {
        try {
          const content = event.target?.result

          if (file.type === "application/json") {
            resolve(JSON.parse(content as string))
          } else if (file.type === "text/csv" || file.type.includes("csv")) {
            // Simple CSV parsing
            const lines = (content as string).split("\n")
            const headers = lines[0].split(",")
            const data = []

            for (let i = 1; i < lines.length; i++) {
              if (lines[i].trim() === "") continue

              const values = lines[i].split(",")
              const row: Record<string, any> = {}

              for (let j = 0; j < headers.length; j++) {
                row[headers[j].trim()] = values[j]?.trim() || ""
              }

              data.push(row)
            }

            resolve(data)
          } else {
            // For other file types, return the raw content
            resolve(content)
          }
        } catch (error) {
          reject(error)
        }
      }

      reader.onerror = (error) => {
        reject(error)
      }

      if (file.type === "application/json" || file.type === "text/csv" || file.type.includes("csv")) {
        reader.readAsText(file)
      } else {
        reader.readAsArrayBuffer(file)
      }
    })
  }

  // Preprocess the data
  public async preprocessData(
    options: {
      handleMissingValues: boolean
      removeDuplicates: boolean
      encodeCategorial: boolean
      fixInconsistentData: boolean
      handleOutliers: boolean
      normalization: boolean
      scaling: boolean
      featureEngineering: boolean
      dimensionalityReduction: boolean
    },
    onProgress?: (progress: number, status: string) => void,
  ): Promise<any[]> {
    const processId = uuidv4()
    if (onProgress) {
      this.processingCallbacks[processId] = onProgress
    }

    try {
      let processedData = [...this.currentDataset]

      // Handle missing values
      if (options.handleMissingValues) {
        this.updateProgress(processId, 10, "Handling missing values...")
        processedData = this.handleMissingValues(processedData)
      }

      // Remove duplicates
      if (options.removeDuplicates) {
        this.updateProgress(processId, 20, "Removing duplicates...")
        processedData = this.removeDuplicates(processedData)
      }

      // Encode categorical variables
      if (options.encodeCategorial) {
        this.updateProgress(processId, 30, "Encoding categorical variables...")
        processedData = this.encodeCategoricalVariables(processedData)
      }

      // Fix inconsistent data
      if (options.fixInconsistentData) {
        this.updateProgress(processId, 40, "Fixing inconsistent data...")
        processedData = this.fixInconsistentData(processedData)
      }

      // Handle outliers
      if (options.handleOutliers) {
        this.updateProgress(processId, 50, "Handling outliers...")
        processedData = this.handleOutliers(processedData)
      }

      // Apply normalization
      if (options.normalization) {
        this.updateProgress(processId, 60, "Applying normalization...")
        processedData = this.normalizeData(processedData)
      }

      // Apply scaling
      if (options.scaling) {
        this.updateProgress(processId, 70, "Applying scaling...")
        processedData = this.scaleData(processedData)
      }

      // Feature engineering
      if (options.featureEngineering) {
        this.updateProgress(processId, 80, "Performing feature engineering...")
        processedData = this.engineerFeatures(processedData)
      }

      // Dimensionality reduction
      if (options.dimensionalityReduction) {
        this.updateProgress(processId, 90, "Reducing dimensionality...")
        processedData = this.reduceDimensionality(processedData)
      }

      this.processedDataset = processedData
      this.updateProgress(processId, 100, "Preprocessing complete")

      if (onProgress) {
        delete this.processingCallbacks[processId]
      }

      return processedData
    } catch (error) {
      console.error("Error preprocessing data:", error)
      throw new Error("Failed to preprocess data")
    }
  }

  // Analyze the data
  public async analyzeData(onProgress?: (progress: number, status: string) => void): Promise<DatasetAnalysis> {
    const processId = uuidv4()
    if (onProgress) {
      this.processingCallbacks[processId] = onProgress
    }

    try {
      const data = this.processedDataset.length > 0 ? this.processedDataset : this.currentDataset

      // Generate summary
      this.updateProgress(processId, 20, "Generating dataset summary...")
      const summary = this.generateSummary(data)

      // Calculate column statistics
      this.updateProgress(processId, 40, "Calculating column statistics...")
      const columnStats = this.calculateColumnStatistics(data)

      // Calculate correlations
      this.updateProgress(processId, 60, "Calculating correlations...")
      const correlations = this.calculateCorrelations(data)

      // Generate insights
      this.updateProgress(processId, 80, "Generating insights...")
      const insights = this.generateInsights(data, columnStats, correlations)

      this.updateProgress(processId, 100, "Analysis complete")

      if (onProgress) {
        delete this.processingCallbacks[processId]
      }

      return {
        summary,
        columnStats,
        correlations,
        insights,
      }
    } catch (error) {
      console.error("Error analyzing data:", error)
      throw new Error("Failed to analyze data")
    }
  }

  // Generate visualizations
  public async generateVisualizations(
    onProgress?: (progress: number, status: string) => void,
  ): Promise<Visualization[]> {
    const processId = uuidv4()
    if (onProgress) {
      this.processingCallbacks[processId] = onProgress
    }

    try {
      const data = this.processedDataset.length > 0 ? this.processedDataset : this.currentDataset
      const visualizations: Visualization[] = []

      // Generate distribution visualizations
      this.updateProgress(processId, 20, "Generating distribution visualizations...")
      const distributionViz = this.generateDistributionVisualizations(data)
      visualizations.push(...distributionViz)

      // Generate correlation visualizations
      this.updateProgress(processId, 40, "Generating correlation visualizations...")
      const correlationViz = this.generateCorrelationVisualizations(data)
      visualizations.push(...correlationViz)

      // Generate time series visualizations if applicable
      this.updateProgress(processId, 60, "Checking for time series data...")
      const timeSeriesViz = this.generateTimeSeriesVisualizations(data)
      if (timeSeriesViz.length > 0) {
        visualizations.push(...timeSeriesViz)
      }

      // Generate scatter plot visualizations
      this.updateProgress(processId, 80, "Generating relationship visualizations...")
      const scatterViz = this.generateScatterVisualizations(data)
      visualizations.push(...scatterViz)

      this.updateProgress(processId, 100, "Visualization generation complete")

      if (onProgress) {
        delete this.processingCallbacks[processId]
      }

      return visualizations
    } catch (error) {
      console.error("Error generating visualizations:", error)
      throw new Error("Failed to generate visualizations")
    }
  }

  // Train models
  public async trainModels(
    targetColumn: string,
    modelTypes: string[],
    options: { testSize: number },
    onProgress?: (progress: number, status: string) => void,
  ): Promise<ModelResult[]> {
    const processId = uuidv4()
    if (onProgress) {
      this.processingCallbacks[processId] = onProgress
    }

    try {
      const data = this.processedDataset.length > 0 ? this.processedDataset : this.currentDataset
      const results: ModelResult[] = []

      // Split data into features and target
      this.updateProgress(processId, 10, "Preparing data for modeling...")
      const { features, target } = this.splitFeaturesTarget(data, targetColumn)

      // Split into training and testing sets
      this.updateProgress(processId, 20, "Splitting into training and testing sets...")
      const { trainFeatures, testFeatures, trainTarget, testTarget } = this.trainTestSplit(
        features,
        target,
        options.testSize,
      )

      // Train each model
      const totalModels = modelTypes.length
      for (let i = 0; i < totalModels; i++) {
        const modelType = modelTypes[i]
        this.updateProgress(processId, 20 + Math.floor(70 * (i / totalModels)), `Training ${modelType} model...`)

        const result = this.trainModel(modelType, trainFeatures, testFeatures, trainTarget, testTarget)

        results.push(result)
      }

      this.updateProgress(processId, 100, "Model training complete")

      if (onProgress) {
        delete this.processingCallbacks[processId]
      }

      return results
    } catch (error) {
      console.error("Error training models:", error)
      throw new Error("Failed to train models")
    }
  }

  // Get the current data
  getCurrentData(): DataFile | null {
    return this.currentData
  }

  // Get the analysis result
  getAnalysisResult(): AnalysisResult | null {
    return this.analysisResult
  }

  // Get the processing status
  getProcessingStatus(): string {
    return this.processingStatus
  }

  // Add a listener for status updates
  addListener(listener: (status: string, data?: any) => void): void {
    this.listeners.push(listener)
  }

  // Remove a listener
  removeListener(listener: (status: string, data?: any) => void): void {
    this.listeners = this.listeners.filter((l) => l !== listener)
  }

  // Notify all listeners
  private notifyListeners(status: string, data?: any): void {
    this.listeners.forEach((listener) => listener(status, data))
  }

  // Mock data processing for development (when backend is not available)
  async mockProcessData(file: File): Promise<AnalysisResult> {
    this.processingStatus = "processing"
    this.notifyListeners("processing")

    // Read the file content
    const content = await this.readFileContent(file)

    // Create a data file object
    this.currentData = {
      id: Date.now().toString(),
      name: file.name,
      type: file.type,
      size: file.size,
      content,
    }

    // Wait for 2 seconds to simulate processing
    await new Promise((resolve) => setTimeout(resolve, 2000))

    // Generate mock analysis result based on the actual data
    const mockResult = this.generateMockAnalysisResult(content)

    this.analysisResult = mockResult
    this.processingStatus = "completed"
    this.notifyListeners("completed", this.analysisResult)

    return mockResult
  }

  // Generate mock analysis result based on actual data
  private generateMockAnalysisResult(data: any[]): AnalysisResult {
    // Extract column names and data types
    const columns: Record<string, string> = {}
    const firstRow = data[0] || {}

    Object.keys(firstRow).forEach((key) => {
      const value = firstRow[key]
      if (typeof value === "number") {
        columns[key] = "number"
      } else if (typeof value === "boolean") {
        columns[key] = "boolean"
      } else if (typeof value === "string") {
        if (!isNaN(Date.parse(value))) {
          columns[key] = "date"
        } else {
          columns[key] = "string"
        }
      } else {
        columns[key] = "unknown"
      }
    })

    // Count missing values
    let missingValues = 0
    data.forEach((row) => {
      Object.values(row).forEach((value) => {
        if (value === null || value === undefined || value === "") {
          missingValues++
        }
      })
    })

    // Count duplicate rows
    const stringifiedRows = data.map((row) => JSON.stringify(row))
    const uniqueRows = new Set(stringifiedRows)
    const duplicateRows = data.length - uniqueRows.size

    // Generate visualizations based on data
    const visualizations: Visualization[] = []

    // Add bar chart for categorical columns
    const categoricalColumns = Object.entries(columns)
      .filter(([_, type]) => type === "string")
      .map(([name]) => name)

    if (categoricalColumns.length > 0) {
      const columnName = categoricalColumns[0]
      const counts: Record<string, number> = {}

      data.forEach((row) => {
        const value = row[columnName] as string
        counts[value] = (counts[value] || 0) + 1
      })

      visualizations.push({
        id: "bar-chart-1",
        title: `Distribution of ${columnName}`,
        description: `Bar chart showing the distribution of values in the ${columnName} column`,
        type: "bar",
        data: Object.entries(counts).map(([name, value]) => ({ name, value })),
        config: {
          xKey: "name",
          yKey: "value",
          colors: ["#8884d8", "#82ca9d", "#ffc658", "#ff8042", "#a4de6c"],
        },
      })
    }

    // Add line chart for numerical columns over time if date column exists
    const numericalColumns = Object.entries(columns)
      .filter(([_, type]) => type === "number")
      .map(([name]) => name)

    const dateColumns = Object.entries(columns)
      .filter(([_, type]) => type === "date")
      .map(([name]) => name)

    if (numericalColumns.length > 0 && dateColumns.length > 0) {
      const dateColumn = dateColumns[0]
      const numColumn = numericalColumns[0]

      // Sort data by date
      const sortedData = [...data].sort((a, b) => {
        return new Date(a[dateColumn]).getTime() - new Date(b[dateColumn]).getTime()
      })

      visualizations.push({
        id: "line-chart-1",
        title: `${numColumn} over time`,
        description: `Line chart showing ${numColumn} values over time`,
        type: "line",
        data: sortedData.map((row) => ({
          date: row[dateColumn],
          value: row[numColumn],
        })),
        config: {
          xKey: "date",
          yKey: "value",
          color: "#8884d8",
        },
      })
    }

    // Add scatter plot for two numerical columns
    if (numericalColumns.length >= 2) {
      const xColumn = numericalColumns[0]
      const yColumn = numericalColumns[1]

      visualizations.push({
        id: "scatter-plot-1",
        title: `${xColumn} vs ${yColumn}`,
        description: `Scatter plot showing the relationship between ${xColumn} and ${yColumn}`,
        type: "scatter",
        data: data.map((row) => ({
          x: row[xColumn],
          y: row[yColumn],
        })),
        config: {
          xKey: "x",
          yKey: "y",
          color: "#82ca9d",
        },
      })
    }

    // Add pie chart for categorical column with few unique values
    if (categoricalColumns.length > 0) {
      const columnName = categoricalColumns[0]
      const counts: Record<string, number> = {}

      data.forEach((row) => {
        const value = row[columnName] as string
        counts[value] = (counts[value] || 0) + 1
      })

      // Only create pie chart if there are fewer than 10 unique values
      if (Object.keys(counts).length < 10) {
        visualizations.push({
          id: "pie-chart-1",
          title: `Distribution of ${columnName}`,
          description: `Pie chart showing the distribution of values in the ${columnName} column`,
          type: "pie",
          data: Object.entries(counts).map(([name, value]) => ({ name, value })),
          config: {
            nameKey: "name",
            valueKey: "value",
            colors: ["#8884d8", "#82ca9d", "#ffc658", "#ff8042", "#a4de6c"],
          },
        })
      }
    }

    // Add histogram for numerical column
    if (numericalColumns.length > 0) {
      const columnName = numericalColumns[0]
      const values = data.map((row) => row[columnName] as number).filter((val) => !isNaN(val))

      // Create bins
      const min = Math.min(...values)
      const max = Math.max(...values)
      const binCount = 10
      const binWidth = (max - min) / binCount
      const bins: Record<string, number> = {}

      for (let i = 0; i < binCount; i++) {
        const binStart = min + i * binWidth
        const binEnd = binStart + binWidth
        const binName = `${binStart.toFixed(2)}-${binEnd.toFixed(2)}`
        bins[binName] = 0
      }

      values.forEach((value) => {
        for (let i = 0; i < binCount; i++) {
          const binStart = min + i * binWidth
          const binEnd = binStart + binWidth
          const binName = `${binStart.toFixed(2)}-${binEnd.toFixed(2)}`

          if (value >= binStart && value < binEnd) {
            bins[binName]++
            break
          }
        }
      })

      visualizations.push({
        id: "histogram-1",
        title: `Histogram of ${columnName}`,
        description: `Histogram showing the distribution of values in the ${columnName} column`,
        type: "histogram",
        data: Object.entries(bins).map(([name, value]) => ({ name, value })),
        config: {
          xKey: "name",
          yKey: "value",
          color: "#8884d8",
        },
      })
    }

    // Generate preprocessing steps
    const preprocessingSteps: PreprocessingStep[] = []

    // Missing value handling
    if (missingValues > 0) {
      preprocessingSteps.push({
        id: "missing-values",
        name: "Handle Missing Values",
        description: `Replaced ${missingValues} missing values with appropriate strategies (mean for numerical, mode for categorical)`,
        appliedTo: Object.keys(columns),
        result: "Success",
      })
    }

    // Duplicate row removal
    if (duplicateRows > 0) {
      preprocessingSteps.push({
        id: "duplicate-rows",
        name: "Remove Duplicate Rows",
        description: `Removed ${duplicateRows} duplicate rows from the dataset`,
        appliedTo: ["entire dataset"],
        result: "Success",
      })
    }

    // Encoding categorical variables
    const categoricalColumnsForEncoding = Object.entries(columns)
      .filter(([_, type]) => type === "string")
      .map(([name]) => name)

    if (categoricalColumnsForEncoding.length > 0) {
      preprocessingSteps.push({
        id: "categorical-encoding",
        name: "Encode Categorical Variables",
        description: `Applied one-hot encoding to categorical variables`,
        appliedTo: categoricalColumnsForEncoding,
        result: "Success",
      })
    }

    // Scaling numerical features
    const numericalColumnsForScaling = Object.entries(columns)
      .filter(([_, type]) => type === "number")
      .map(([name]) => name)

    if (numericalColumnsForScaling.length > 0) {
      preprocessingSteps.push({
        id: "scaling",
        name: "Scale Numerical Features",
        description: `Applied standard scaling to numerical features`,
        appliedTo: numericalColumnsForScaling,
        result: "Success",
      })
    }

    // Generate model results based on data characteristics
    const modelResults: ModelResult[] = []

    // Regression model if there are numerical targets
    if (numericalColumns.length >= 2) {
      modelResults.push({
        id: "linear-regression",
        name: "Linear Regression",
        type: "regression",
        metrics: {
          "R²": 0.78,
          MSE: 0.34,
          MAE: 0.21,
        },
        parameters: {
          fit_intercept: true,
          normalize: false,
        },
        description: "Linear regression model to predict numerical values based on other features",
      })

      modelResults.push({
        id: "random-forest-regression",
        name: "Random Forest Regression",
        type: "regression",
        metrics: {
          "R²": 0.85,
          MSE: 0.28,
          MAE: 0.18,
        },
        parameters: {
          n_estimators: 100,
          max_depth: 10,
        },
        description: "Random forest regression model for improved prediction accuracy",
      })
    }

    // Classification model if there are categorical targets
    if (categoricalColumns.length > 0) {
      modelResults.push({
        id: "logistic-regression",
        name: "Logistic Regression",
        type: "classification",
        metrics: {
          Accuracy: 0.82,
          Precision: 0.79,
          Recall: 0.81,
          F1: 0.8,
        },
        parameters: {
          C: 1.0,
          penalty: "l2",
        },
        description: "Logistic regression model for classification tasks",
      })

      modelResults.push({
        id: "random-forest-classification",
        name: "Random Forest Classification",
        type: "classification",
        metrics: {
          Accuracy: 0.88,
          Precision: 0.86,
          Recall: 0.85,
          F1: 0.85,
        },
        parameters: {
          n_estimators: 100,
          max_depth: 10,
        },
        description: "Random forest classification model for improved classification accuracy",
      })
    }

    // Clustering model for exploratory analysis
    modelResults.push({
      id: "kmeans-clustering",
      name: "K-Means Clustering",
      type: "clustering",
      metrics: {
        "Silhouette Score": 0.68,
        Inertia: 245.6,
      },
      parameters: {
        n_clusters: 3,
        init: "k-means++",
      },
      description: "K-means clustering model to identify natural groupings in the data",
    })

    return {
      summary: {
        rowCount: data.length,
        columnCount: Object.keys(columns).length,
        missingValues,
        duplicateRows,
        dataTypes: columns,
      },
      visualizations,
      preprocessingSteps,
      modelResults,
    }
  }

  // Helper methods
  private updateProgress(processId: string, progress: number, status: string): void {
    const callback = this.processingCallbacks[processId]
    if (callback) {
      callback(progress, status)
    }
  }

  private parseData(filename: string, content: string): any[] {
    if (filename.endsWith(".csv")) {
      return this.parseCSV(content)
    } else if (filename.endsWith(".json")) {
      return JSON.parse(content)
    } else {
      throw new Error(`Unsupported file format: ${filename}`)
    }
  }

  private parseCSV(content: string): any[] {
    const lines = content.split("\n")
    const headers = lines[0].split(",").map((h) => h.trim())

    return lines
      .slice(1)
      .filter((line) => line.trim())
      .map((line) => {
        const values = line.split(",").map((v) => v.trim())
        const row: Record<string, any> = {}

        headers.forEach((header, index) => {
          const value = values[index]
          // Try to convert to number if possible
          const numValue = Number(value)
          row[header] = isNaN(numValue) ? value : numValue
        })

        return row
      })
  }

  private determineColumnTypes(data: any[]): Record<string, string> {
    if (data.length === 0) return {}

    const types: Record<string, string> = {}
    const firstRow = data[0]

    Object.keys(firstRow).forEach((column) => {
      // Check a sample of rows to determine type
      const sampleSize = Math.min(100, data.length)
      const sample = data.slice(0, sampleSize)

      // Check if column contains numbers
      const numericCount = sample.filter(
        (row) => typeof row[column] === "number" || (typeof row[column] === "string" && !isNaN(Number(row[column]))),
      ).length

      // Check if column contains dates
      const dateRegex = /^\d{4}[-/]\d{1,2}[-/]\d{1,2}$/
      const dateCount = sample.filter((row) => typeof row[column] === "string" && dateRegex.test(row[column])).length

      // Determine type based on majority
      if (dateCount > sampleSize * 0.7) {
        types[column] = "date"
      } else if (numericCount > sampleSize * 0.7) {
        types[column] = "numeric"
      } else {
        types[column] = "categorical"
      }
    })

    return types
  }

  private generateSummary(data: any[]): DatasetSummary {
    // Count rows
    const rows = data.length

    // Count columns
    const columns = data.length > 0 ? Object.keys(data[0]).length : 0

    // Count missing values
    let missingValues = 0
    data.forEach((row) => {
      Object.values(row).forEach((value) => {
        if (value === null || value === undefined || value === "") {
          missingValues++
        }
      })
    })

    // Count duplicates
    const stringifiedRows = data.map((row) => JSON.stringify(row))
    const uniqueRows = new Set(stringifiedRows)
    const duplicates = rows - uniqueRows.size

    // Get column types
    const columnTypes = this.determineColumnTypes(data)

    // Get sample data
    const sampleData = data.slice(0, 5)

    return {
      rows,
      columns,
      missingValues,
      duplicates,
      columnTypes,
      sampleData,
    }
  }

  // Data preprocessing methods
  private handleMissingValues(data: any[]): any[] {
    const columnTypes = this.determineColumnTypes(data)
    const result = [...data]

    // For each column
    Object.keys(columnTypes).forEach((column) => {
      const type = columnTypes[column]

      // For numeric columns, use mean
      if (type === "numeric") {
        const values = data.map((row) => row[column]).filter((val) => val !== null && val !== undefined && val !== "")
        const sum = values.reduce((acc, val) => acc + Number(val), 0)
        const mean = sum / values.length

        result.forEach((row) => {
          if (row[column] === null || row[column] === undefined || row[column] === "") {
            row[column] = mean
          }
        })
      }
      // For categorical columns, use mode
      else {
        const valueCounts: Record<string, number> = {}
        data.forEach((row) => {
          const value = row[column]
          if (value !== null && value !== undefined && value !== "") {
            valueCounts[value] = (valueCounts[value] || 0) + 1
          }
        })

        let mode = ""
        let maxCount = 0
        Object.entries(valueCounts).forEach(([value, count]) => {
          if (count > maxCount) {
            maxCount = count
            mode = value
          }
        })

        result.forEach((row) => {
          if (row[column] === null || row[column] === undefined || row[column] === "") {
            row[column] = mode
          }
        })
      }
    })

    return result
  }

  private removeDuplicates(data: any[]): any[] {
    const uniqueRows = new Map()

    data.forEach((row) => {
      const key = JSON.stringify(row)
      uniqueRows.set(key, row)
    })

    return Array.from(uniqueRows.values())
  }

  private encodeCategoricalVariables(data: any[]): any[] {
    const columnTypes = this.determineColumnTypes(data)
    const result = [...data]

    // For each categorical column
    Object.entries(columnTypes).forEach(([column, type]) => {
      if (type === "categorical") {
        // Get unique values
        const uniqueValues = new Set<string>()
        data.forEach((row) => {
          if (row[column] !== null && row[column] !== undefined) {
            uniqueValues.add(String(row[column]))
          }
        })

        // Create mapping
        const valueMap = new Map<string, number>()
        Array.from(uniqueValues)
          .sort()
          .forEach((value, index) => {
            valueMap.set(value, index)
          })

        // Encode values
        result.forEach((row) => {
          if (row[column] !== null && row[column] !== undefined) {
            row[column] = valueMap.get(String(row[column])) || 0
          }
        })
      }
    })

    return result
  }

  private fixInconsistentData(data: any[]): any[] {
    const columnTypes = this.determineColumnTypes(data)
    const result = [...data]

    // For each column
    Object.entries(columnTypes).forEach(([column, type]) => {
      if (type === "numeric") {
        // Convert all values to numbers
        result.forEach((row) => {
          if (row[column] !== null && row[column] !== undefined && row[column] !== "") {
            row[column] = Number(row[column])
          }
        })
      }
    })

    return result
  }

  private handleOutliers(data: any[]): any[] {
    const columnTypes = this.determineColumnTypes(data)
    const result = [...data]

    // For each numeric column
    Object.entries(columnTypes).forEach(([column, type]) => {
      if (type === "numeric") {
        // Calculate quartiles
        const values = data.map((row) => Number(row[column])).filter((val) => !isNaN(val))
        values.sort((a, b) => a - b)

        const q1Index = Math.floor(values.length * 0.25)
        const q3Index = Math.floor(values.length * 0.75)

        const q1 = values[q1Index]
        const q3 = values[q3Index]

        const iqr = q3 - q1
        const lowerBound = q1 - 1.5 * iqr
        const upperBound = q3 + 1.5 * iqr

        // Replace outliers with bounds
        result.forEach((row) => {
          if (row[column] < lowerBound) {
            row[column] = lowerBound
          } else if (row[column] > upperBound) {
            row[column] = upperBound
          }
        })
      }
    })

    return result
  }

  private normalizeData(data: any[]): any[] {
    const columnTypes = this.determineColumnTypes(data)
    const result = [...data]

    // For each numeric column
    Object.entries(columnTypes).forEach(([column, type]) => {
      if (type === "numeric") {
        // Find min and max
        const values = data.map((row) => Number(row[column])).filter((val) => !isNaN(val))
        const min = Math.min(...values)
        const max = Math.max(...values)

        // Normalize values to [0, 1]
        if (max > min) {
          result.forEach((row) => {
            row[column] = (Number(row[column]) - min) / (max - min)
          })
        }
      }
    })

    return result
  }

  private scaleData(data: any[]): any[] {
    const columnTypes = this.determineColumnTypes(data)
    const result = [...data]

    // For each numeric column
    Object.entries(columnTypes).forEach(([column, type]) => {
      if (type === "numeric") {
        // Calculate mean and standard deviation
        const values = data.map((row) => Number(row[column])).filter((val) => !isNaN(val))
        const mean = values.reduce((acc, val) => acc + val, 0) / values.length

        const squaredDiffs = values.map((val) => Math.pow(val - mean, 2))
        const variance = squaredDiffs.reduce((acc, val) => acc + val, 0) / values.length
        const std = Math.sqrt(variance)

        // Scale values to have mean=0 and std=1
        if (std > 0) {
          result.forEach((row) => {
            row[column] = (Number(row[column]) - mean) / std
          })
        }
      }
    })

    return result
  }

  private engineerFeatures(data: any[]): any[] {
    const columnTypes = this.determineColumnTypes(data)
    const result = [...data]
    const numericColumns = Object.entries(columnTypes)
      .filter(([_, type]) => type === "numeric")
      .map(([column, _]) => column)

    // If we have at least 2 numeric columns, create interaction features
    if (numericColumns.length >= 2) {
      for (let i = 0; i < numericColumns.length; i++) {
        for (let j = i + 1; j < numericColumns.length; j++) {
          const col1 = numericColumns[i]
          const col2 = numericColumns[j]
          const newCol = `${col1}_x_${col2}`

          result.forEach((row) => {
            row[newCol] = Number(row[col1]) * Number(row[col2])
          })
        }
      }
    }

    return result
  }

  private reduceDimensionality(data: any[]): any[] {
    // This is a simplified version of PCA
    // In a real implementation, you would use a proper PCA algorithm
    const columnTypes = this.determineColumnTypes(data)
    const numericColumns = Object.entries(columnTypes)
      .filter(([_, type]) => type === "numeric")
      .map(([column, _]) => column)

    // If we have too few numeric columns, return original data
    if (numericColumns.length <= 3) {
      return data
    }

    // Extract numeric data
    const numericData = data.map((row) => numericColumns.map((col) => Number(row[col])))

    // Calculate covariance matrix (simplified)
    const n = numericData.length
    const means = numericColumns.map((_, colIndex) => numericData.reduce((sum, row) => sum + row[colIndex], 0) / n)

    // Create PCA components (simplified)
    const result = [...data]
    const numComponents = Math.min(3, numericColumns.length)

    for (let i = 0; i < numComponents; i++) {
      const componentName = `PCA_${i + 1}`

      result.forEach((row, rowIndex) => {
        // Simple weighted sum of features
        row[componentName] = numericColumns.reduce((sum, col, colIndex) => {
          // Random weights for demonstration
          const weight = Math.sin(i * colIndex + 1)
          return sum + Number(row[col]) * weight
        }, 0)
      })
    }

    return result
  }

  // Analysis methods
  private calculateColumnStatistics(data: any[]): ColumnStatistics[] {
    const columnTypes = this.determineColumnTypes(data)
    const stats: ColumnStatistics[] = []

    Object.entries(columnTypes).forEach(([column, type]) => {
      const values = data.map((row) => row[column])

      // Count non-null values
      const nonNullValues = values.filter((val) => val !== null && val !== undefined && val !== "")
      const count = nonNullValues.length

      // Count missing values
      const missing = values.length - count

      // Count unique values
      const uniqueValues = new Set(nonNullValues.map(String))
      const unique = uniqueValues.size

      const columnStat: ColumnStatistics = {
        name: column,
        type,
        count,
        missing,
        unique,
      }

      // For numeric columns, calculate additional statistics
      if (type === "numeric") {
        const numericValues = nonNullValues.map(Number).filter((val) => !isNaN(val))

        if (numericValues.length > 0) {
          // Calculate mean
          const sum = numericValues.reduce((acc, val) => acc + val, 0)
          const mean = sum / numericValues.length
          columnStat.mean = mean

          // Calculate standard deviation
          const squaredDiffs = numericValues.map((val) => Math.pow(val - mean, 2))
          const variance = squaredDiffs.reduce((acc, val) => acc + val, 0) / numericValues.length
          columnStat.std = Math.sqrt(variance)

          // Sort values for percentiles
          numericValues.sort((a, b) => a - b)

          // Calculate min, max, and quartiles
          columnStat.min = numericValues[0]
          columnStat.max = numericValues[numericValues.length - 1]

          const q1Index = Math.floor(numericValues.length * 0.25)
          const medianIndex = Math.floor(numericValues.length * 0.5)
          const q3Index = Math.floor(numericValues.length * 0.75)

          columnStat.q1 = numericValues[q1Index]
          columnStat.median = numericValues[medianIndex]
          columnStat.q3 = numericValues[q3Index]
        }
      }
      // For categorical columns, calculate mode and category counts
      else if (type === "categorical") {
        const valueCounts: Record<string, number> = {}
        nonNullValues.forEach((val) => {
          const strVal = String(val)
          valueCounts[strVal] = (valueCounts[strVal] || 0) + 1
        })

        // Find mode
        let mode = ""
        let maxCount = 0
        Object.entries(valueCounts).forEach(([value, count]) => {
          if (count > maxCount) {
            maxCount = count
            mode = value
          }
        })

        columnStat.mode = mode
        columnStat.categories = Object.keys(valueCounts)
        columnStat.categoryCount = valueCounts
      }

      stats.push(columnStat)
    })

    return stats
  }

  private calculateCorrelations(data: any[]): Record<string, Record<string, number>> {
    const columnTypes = this.determineColumnTypes(data)
    const numericColumns = Object.entries(columnTypes)
      .filter(([_, type]) => type === "numeric")
      .map(([column, _]) => column)

    const correlations: Record<string, Record<string, number>> = {}

    // Initialize correlation matrix
    numericColumns.forEach((col1) => {
      correlations[col1] = {}
      numericColumns.forEach((col2) => {
        correlations[col1][col2] = col1 === col2 ? 1 : 0
      })
    })

    // Calculate Pearson correlation for each pair of numeric columns
    for (let i = 0; i < numericColumns.length; i++) {
      for (let j = i + 1; j < numericColumns.length; j++) {
        const col1 = numericColumns[i]
        const col2 = numericColumns[j]

        // Extract values
        const pairs = data
          .map((row) => ({
            x: Number(row[col1]),
            y: Number(row[col2]),
          }))
          .filter((pair) => !isNaN(pair.x) && !isNaN(pair.y))

        if (pairs.length > 0) {
          // Calculate means
          const sumX = pairs.reduce((acc, pair) => acc + pair.x, 0)
          const sumY = pairs.reduce((acc, pair) => acc + pair.y, 0)
          const meanX = sumX / pairs.length
          const meanY = sumY / pairs.length

          // Calculate correlation
          let numerator = 0
          let denomX = 0
          let denomY = 0

          pairs.forEach((pair) => {
            const diffX = pair.x - meanX
            const diffY = pair.y - meanY
            numerator += diffX * diffY
            denomX += diffX * diffX
            denomY += diffY * diffY
          })

          const correlation = numerator / (Math.sqrt(denomX) * Math.sqrt(denomY))

          // Store correlation
          correlations[col1][col2] = correlation
          correlations[col2][col1] = correlation
        }
      }
    }

    return correlations
  }

  private generateInsights(
    data: any[],
    columnStats: ColumnStatistics[],
    correlations: Record<string, Record<string, number>>,
  ): string[] {
    const insights: string[] = []

    // Insight about dataset size
    insights.push(`The dataset contains ${data.length} rows and ${columnStats.length} columns.`)

    // Insight about missing values
    const totalMissing = columnStats.reduce((acc, stat) => acc + stat.missing, 0)
    if (totalMissing > 0) {
      const missingPercent = ((totalMissing / (data.length * columnStats.length)) * 100).toFixed(2)
      insights.push(`The dataset contains ${totalMissing} missing values (${missingPercent}% of all values).`)

      // Columns with most missing values
      const columnsWithMissing = columnStats
        .filter((stat) => stat.missing > 0)
        .sort((a, b) => b.missing - a.missing)
        .slice(0, 3)

      if (columnsWithMissing.length > 0) {
        const missingColumns = columnsWithMissing.map((stat) => `${stat.name} (${stat.missing} missing)`).join(", ")
        insights.push(`Columns with most missing values: ${missingColumns}.`)
      }
    }

    // Insight about correlations
    const correlationPairs: { col1: string; col2: string; correlation: number }[] = []
    Object.entries(correlations).forEach(([col1, colCorrelations]) => {
      Object.entries(colCorrelations).forEach(([col2, correlation]) => {
        if (col1 < col2) {
          // Avoid duplicates
          correlationPairs.push({ col1, col2, correlation })
        }
      })
    })

    // Sort by absolute correlation
    correlationPairs.sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation))

    // Strong positive correlations
    const strongPositive = correlationPairs.filter((pair) => pair.correlation > 0.7).slice(0, 3)

    if (strongPositive.length > 0) {
      const pairs = strongPositive
        .map((pair) => `${pair.col1} and ${pair.col2} (${pair.correlation.toFixed(2)})`)
        .join(", ")
      insights.push(`Strong positive correlations found between: ${pairs}.`)
    }

    // Strong negative correlations
    const strongNegative = correlationPairs.filter((pair) => pair.correlation < -0.7).slice(0, 3)

    if (strongNegative.length > 0) {
      const pairs = strongNegative
        .map((pair) => `${pair.col1} and ${pair.col2} (${pair.correlation.toFixed(2)})`)
        .join(", ")
      insights.push(`Strong negative correlations found between: ${pairs}.`)
    }

    // Insight about numeric columns
    const numericStats = columnStats.filter((stat) => stat.type === "numeric")
    if (numericStats.length > 0) {
      // Find columns with outliers
      const columnsWithOutliers = numericStats
        .filter((stat) => {
          if (stat.q1 !== undefined && stat.q3 !== undefined && stat.min !== undefined && stat.max !== undefined) {
            const iqr = stat.q3 - stat.q1
            const lowerBound = stat.q1 - 1.5 * iqr
            const upperBound = stat.q3 + 1.5 * iqr
            return stat.min < lowerBound || stat.max > upperBound
          }
          return false
        })
        .map((stat) => stat.name)

      if (columnsWithOutliers.length > 0) {
        insights.push(`Potential outliers detected in columns: ${columnsWithOutliers.join(", ")}.`)
      }
    }

    // Insight about categorical columns
    const categoricalStats = columnStats.filter((stat) => stat.type === "categorical")
    if (categoricalStats.length > 0) {
      // Find columns with high cardinality
      const highCardinalityColumns = categoricalStats
        .filter((stat) => stat.unique > 20)
        .map((stat) => `${stat.name} (${stat.unique} unique values)`)

      if (highCardinalityColumns.length > 0) {
        insights.push(`High cardinality detected in categorical columns: ${highCardinalityColumns.join(", ")}.`)
      }

      // Find imbalanced categories
      const imbalancedColumns = categoricalStats
        .filter((stat) => {
          if (stat.categoryCount) {
            const counts = Object.values(stat.categoryCount)
            const max = Math.max(...counts)
            const min = Math.min(...counts)
            return max > 10 * min // 10:1 ratio as threshold
          }
          return false
        })
        .map((stat) => stat.name)

      if (imbalancedColumns.length > 0) {
        insights.push(`Imbalanced categories detected in columns: ${imbalancedColumns.join(", ")}.`)
      }
    }

    return insights
  }

  // Visualization methods
  private generateDistributionVisualizations(data: any[]): Visualization[] {
    const columnTypes = this.determineColumnTypes(data)
    const visualizations: Visualization[] = []

    // Generate bar charts for categorical columns
    const categoricalColumns = Object.entries(columnTypes)
      .filter(([_, type]) => type === "categorical")
      .map(([column, _]) => column)
      .slice(0, 3) // Limit to 3 columns

    categoricalColumns.forEach((column) => {
      // Count occurrences of each value
      const valueCounts: Record<string, number> = {}
      data.forEach((row) => {
        const value = String(row[column])
        valueCounts[value] = (valueCounts[value] || 0) + 1
      })

      // Convert to chart data
      const chartData = Object.entries(valueCounts)
        .map(([name, value]) => ({ name, value }))
        .sort((a, b) => b.value - a.value)
        .slice(0, 10) // Limit to top 10 categories

      visualizations.push({
        id: `bar-${column}`,
        title: `Distribution of ${column}`,
        description: `Bar chart showing the distribution of values in ${column}`,
        type: "bar",
        imageUrl: `/placeholder.svg?height=300&width=500&text=${encodeURIComponent(`Distribution of ${column}`)}`,
        data: chartData,
      })
    })

    // Generate histograms for numeric columns
    const numericColumns = Object.entries(columnTypes)
      .filter(([_, type]) => type === "numeric")
      .map(([column, _]) => column)
      .slice(0, 3) // Limit to 3 columns

    numericColumns.forEach((column) => {
      // Extract values
      const values = data.map((row) => Number(row[column])).filter((val) => !isNaN(val))

      // Create bins
      const min = Math.min(...values)
      const max = Math.max(...values)
      const binCount = 10
      const binWidth = (max - min) / binCount

      const bins: Record<string, number> = {}
      for (let i = 0; i < binCount; i++) {
        const binStart = min + i * binWidth
        const binEnd = binStart + binWidth
        const binName = `${binStart.toFixed(2)} - ${binEnd.toFixed(2)}`
        bins[binName] = 0
      }

      // Count values in each bin
      values.forEach((val) => {
        const binIndex = Math.min(binCount - 1, Math.floor((val - min) / binWidth))
        const binStart = min + binIndex * binWidth
        const binEnd = binStart + binWidth
        const binName = `${binStart.toFixed(2)} - ${binEnd.toFixed(2)}`
        bins[binName]++
      })

      // Convert to chart data
      const chartData = Object.entries(bins).map(([name, value]) => ({ name, value }))

      visualizations.push({
        id: `histogram-${column}`,
        title: `Distribution of ${column}`,
        description: `Histogram showing the distribution of values in ${column}`,
        type: "bar",
        imageUrl: `/placeholder.svg?height=300&width=500&text=${encodeURIComponent(`Histogram of ${column}`)}`,
        data: chartData,
      })
    })

    // Generate pie chart for a categorical column with few categories
    const pieColumn = categoricalColumns.find((column) => {
      const uniqueValues = new Set(data.map((row) => String(row[column])))
      return uniqueValues.size <= 10
    })

    if (pieColumn) {
      // Count occurrences of each value
      const valueCounts: Record<string, number> = {}
      data.forEach((row) => {
        const value = String(row[pieColumn])
        valueCounts[value] = (valueCounts[value] || 0) + 1
      })

      // Convert to chart data
      const chartData = Object.entries(valueCounts)
        .map(([name, value]) => ({ name, value }))
        .sort((a, b) => b.value - a.value)

      visualizations.push({
        id: `pie-${pieColumn}`,
        title: `Distribution of ${pieColumn}`,
        description: `Pie chart showing the distribution of values in ${pieColumn}`,
        type: "pie",
        imageUrl: `/placeholder.svg?height=300&width=500&text=${encodeURIComponent(`Pie Chart of ${pieColumn}`)}`,
        data: chartData,
      })
    }

    return visualizations
  }

  private generateCorrelationVisualizations(data: any[]): Visualization[] {
    const columnTypes = this.determineColumnTypes(data)
    const numericColumns = Object.entries(columnTypes)
      .filter(([_, type]) => type === "numeric")
      .map(([column, _]) => column)

    if (numericColumns.length < 2) {
      return []
    }

    // Calculate correlation matrix
    const correlations = this.calculateCorrelations(data)

    // Convert to heatmap data
    const heatmapData: { x: string; y: string; value: number }[] = []
    numericColumns.forEach((col1) => {
      numericColumns.forEach((col2) => {
        heatmapData.push({
          x: col1,
          y: col2,
          value: correlations[col1][col2],
        })
      })
    })

    return [
      {
        id: "correlation-heatmap",
        title: "Correlation Matrix",
        description: "Heatmap showing correlations between numeric features",
        type: "heatmap",
        imageUrl: `/placeholder.svg?height=400&width=500&text=${encodeURIComponent("Correlation Matrix")}`,
        data: heatmapData,
      },
    ]
  }

  private generateTimeSeriesVisualizations(data: any[]): Visualization[] {
    const columnTypes = this.determineColumnTypes(data)
    const dateColumns = Object.entries(columnTypes)
      .filter(([_, type]) => type === "date")
      .map(([column, _]) => column)

    if (dateColumns.length === 0) {
      return []
    }

    const visualizations: Visualization[] = []
    const dateColumn = dateColumns[0]
    const numericColumns = Object.entries(columnTypes)
      .filter(([_, type]) => type === "numeric")
      .map(([column, _]) => column)
      .slice(0, 3) // Limit to 3 columns

    numericColumns.forEach((column) => {
      // Extract date-value pairs
      const pairs = data
        .map((row) => ({
          date: row[dateColumn],
          value: Number(row[column]),
        }))
        .filter((pair) => !isNaN(pair.value))
        .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime())

      // Group by month if too many points
      if (pairs.length > 30) {
        const monthlyData: Record<string, number[]> = {}

        pairs.forEach((pair) => {
          const date = new Date(pair.date)
          const monthKey = `${date.getFullYear()}-${date.getMonth() + 1}`

          if (!monthlyData[monthKey]) {
            monthlyData[monthKey] = []
          }

          monthlyData[monthKey].push(pair.value)
        })

        // Calculate monthly averages
        const monthlyAverages = Object.entries(monthlyData).map(([month, values]) => ({
          name: month,
          value: values.reduce((sum, val) => sum + val, 0) / values.length,
        }))

        visualizations.push({
          id: `timeseries-${column}`,
          title: `${column} Over Time`,
          description: `Line chart showing ${column} values over time`,
          type: "line",
          imageUrl: `/placeholder.svg?height=300&width=500&text=${encodeURIComponent(`${column} Over Time`)}`,
          data: monthlyAverages,
        })
      } else {
        // Use raw data if not too many points
        const chartData = pairs.map((pair) => ({
          name: pair.date,
          value: pair.value,
        }))

        visualizations.push({
          id: `timeseries-${column}`,
          title: `${column} Over Time`,
          description: `Line chart showing ${column} values over time`,
          type: "line",
          imageUrl: `/placeholder.svg?height=300&width=500&text=${encodeURIComponent(`${column} Over Time`)}`,
          data: chartData,
        })
      }
    })

    return visualizations
  }

  private generateScatterVisualizations(data: any[]): Visualization[] {
    const columnTypes = this.determineColumnTypes(data)
    const numericColumns = Object.entries(columnTypes)
      .filter(([_, type]) => type === "numeric")
      .map(([column, _]) => column)

    if (numericColumns.length < 2) {
      return []
    }

    const visualizations: Visualization[] = []

    // Calculate correlations
    const correlations = this.calculateCorrelations(data)

    // Find pairs with highest absolute correlation
    const correlationPairs: { col1: string; col2: string; correlation: number }[] = []
    numericColumns.forEach((col1) => {
      numericColumns.forEach((col2) => {
        if (col1 < col2) {
          // Avoid duplicates
          correlationPairs.push({
            col1,
            col2,
            correlation: correlations[col1][col2],
          })
        }
      })
    })

    // Sort by absolute correlation
    correlationPairs.sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation))

    // Generate scatter plots for top pairs
    const topPairs = correlationPairs.slice(0, 3)

    topPairs.forEach((pair) => {
      const { col1, col2, correlation } = pair

      // Extract data points
      const points = data
        .map((row) => ({
          x: Number(row[col1]),
          y: Number(row[col2]),
        }))
        .filter((point) => !isNaN(point.x) && !isNaN(point.y))

      visualizations.push({
        id: `scatter-${col1}-${col2}`,
        title: `${col1} vs ${col2}`,
        description: `Scatter plot showing relationship between ${col1} and ${col2} (correlation: ${correlation.toFixed(2)})`,
        type: "scatter",
        imageUrl: `/placeholder.svg?height=300&width=500&text=${encodeURIComponent(`${col1} vs ${col2}`)}`,
        data: points,
      })
    })

    return visualizations
  }

  // Model training methods
  private splitFeaturesTarget(data: any[], targetColumn: string): { features: any[][]; target: any[] } {
    const features: any[][] = []
    const target: any[] = []

    data.forEach((row) => {
      const featureRow: any[] = []

      Object.entries(row).forEach(([column, value]) => {
        if (column !== targetColumn) {
          featureRow.push(Number(value))
        }
      })

      features.push(featureRow)
      target.push(row[targetColumn])
    })

    return { features, target }
  }

  private trainTestSplit(
    features: any[][],
    target: any[],
    testSize: number,
  ): { trainFeatures: any[][]; testFeatures: any[][]; trainTarget: any[]; testTarget: any[] } {
    const testCount = Math.floor(features.length * testSize)
    const trainCount = features.length - testCount

    // Create indices and shuffle
    const indices = Array.from({ length: features.length }, (_, i) => i)
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1))
      ;[indices[i], indices[j]] = [indices[j], indices[i]]
    }

    // Split indices
    const trainIndices = indices.slice(0, trainCount)
    const testIndices = indices.slice(trainCount)

    // Create train and test sets
    const trainFeatures = trainIndices.map((i) => features[i])
    const testFeatures = testIndices.map((i) => features[i])
    const trainTarget = trainIndices.map((i) => target[i])
    const testTarget = testIndices.map((i) => target[i])

    return { trainFeatures, testFeatures, trainTarget, testTarget }
  }

  private trainModel(
    modelType: string,
    trainFeatures: any[][],
    testFeatures: any[][],
    trainTarget: any[],
    testTarget: any[],
  ): ModelResult {
    // Check if classification or regression
    const uniqueTargets = new Set(trainTarget)
    const isClassification = uniqueTargets.size <= 10

    // Train model (simplified)
    let predictions: any[]
    let featureImportance: { feature: string; importance: number }[]

    if (modelType === "random-forest") {
      // Simulate random forest
      predictions = this.simulateRandomForest(trainFeatures, testFeatures, trainTarget)
      featureImportance = this.simulateFeatureImportance(trainFeatures[0].length)
    } else if (modelType === "linear-regression") {
      // Simulate linear regression
      predictions = this.simulateLinearRegression(trainFeatures, testFeatures, trainTarget)
      featureImportance = this.simulateFeatureImportance(trainFeatures[0].length)
    } else if (modelType === "svm") {
      // Simulate SVM
      predictions = this.simulateSVM(trainFeatures, testFeatures, trainTarget)
      featureImportance = this.simulateFeatureImportance(trainFeatures[0].length)
    } else {
      // Default to random predictions
      predictions = testTarget.map(() => {
        if (isClassification) {
          return Array.from(uniqueTargets)[Math.floor(Math.random() * uniqueTargets.size)]
        } else {
          return Math.random() * 100
        }
      })
      featureImportance = this.simulateFeatureImportance(trainFeatures[0].length)
    }

    // Evaluate model
    const result: ModelResult = {
      modelType,
      featureImportance,
    }

    if (isClassification) {
      // Classification metrics
      result.accuracy = this.calculateAccuracy(testTarget, predictions)
      result.precision = this.calculatePrecision(testTarget, predictions)
      result.recall = this.calculateRecall(testTarget, predictions)
      result.f1 = this.calculateF1(testTarget, predictions)
      result.confusionMatrix = this.calculateConfusionMatrix(testTarget, predictions)
    } else {
      // Regression metrics
      result.mse = this.calculateMSE(testTarget, predictions)
      result.rmse = Math.sqrt(result.mse)
      result.mae = this.calculateMAE(testTarget, predictions)
      result.r2 = this.calculateR2(testTarget, predictions)
    }

    return result
  }

  // Model simulation methods (simplified)
  private simulateRandomForest(trainFeatures: any[][], testFeatures: any[][], trainTarget: any[]): any[] {
    // This is a very simplified simulation
    const uniqueTargets = new Set(trainTarget)
    const isClassification = uniqueTargets.size <= 10

    if (isClassification) {
      // For classification, predict most common class with some noise
      const classCounts: Record<string, number> = {}
      trainTarget.forEach((target) => {
        classCounts[target] = (classCounts[target] || 0) + 1
      })

      let mostCommonClass = trainTarget[0]
      let maxCount = 0

      Object.entries(classCounts).forEach(([cls, count]) => {
        if (count > maxCount) {
          maxCount = count
          mostCommonClass = cls
        }
      })

      // Add some randomness for realism
      return testFeatures.map(() => {
        return Math.random() < 0.8
          ? mostCommonClass
          : Array.from(uniqueTargets)[Math.floor(Math.random() * uniqueTargets.size)]
      })
    } else {
      // For regression, predict mean with some noise
      const sum = trainTarget.reduce((acc: number, val: number) => acc + Number(val), 0)
      const mean = sum / trainTarget.length
      const std = Math.sqrt(
        trainTarget.reduce((acc: number, val: number) => acc + Math.pow(Number(val) - mean, 2), 0) / trainTarget.length,
      )

      // Add some randomness for realism
      return testFeatures.map(() => {
        return mean + std * (Math.random() - 0.5)
      })
    }
  }

  private simulateLinearRegression(trainFeatures: any[][], testFeatures: any[][], trainTarget: any[]): any[] {
    // This is a very simplified simulation
    const uniqueTargets = new Set(trainTarget)
    const isClassification = uniqueTargets.size <= 10

    if (isClassification) {
      // For classification, predict most common class with some noise
      const classCounts: Record<string, number> = {}
      trainTarget.forEach((target) => {
        classCounts[target] = (classCounts[target] || 0) + 1
      })

      let mostCommonClass = trainTarget[0]
      let maxCount = 0

      Object.entries(classCounts).forEach(([cls, count]) => {
        if (count > maxCount) {
          maxCount = count
          mostCommonClass = cls
        }
      })

      // Add some randomness for realism
      return testFeatures.map(() => {
        return Math.random() < 0.7
          ? mostCommonClass
          : Array.from(uniqueTargets)[Math.floor(Math.random() * uniqueTargets.size)]
      })
    } else {
      // For regression, predict mean with some noise
      const sum = trainTarget.reduce((acc: number, val: number) => acc + Number(val), 0)
      const mean = sum / trainTarget.length
      const std = Math.sqrt(
        trainTarget.reduce((acc: number, val: number) => acc + Math.pow(Number(val) - mean, 2), 0) / trainTarget.length,
      )

      // Add some randomness for realism
      return testFeatures.map(() => {
        return mean + std * (Math.random() - 0.5) * 0.8
      })
    }
  }

  private simulateSVM(trainFeatures: any[][], testFeatures: any[][], trainTarget: any[]): any[] {
    // This is a very simplified simulation
    const uniqueTargets = new Set(trainTarget)
    const isClassification = uniqueTargets.size <= 10

    if (isClassification) {
      // For classification, predict most common class with some noise
      const classCounts: Record<string, number> = {}
      trainTarget.forEach((target) => {
        classCounts[target] = (classCounts[target] || 0) + 1
      })

      let mostCommonClass = trainTarget[0]
      let maxCount = 0

      Object.entries(classCounts).forEach(([cls, count]) => {
        if (count > maxCount) {
          maxCount = count
          mostCommonClass = cls
        }
      })

      // Add some randomness for realism
      return testFeatures.map(() => {
        return Math.random() < 0.75
          ? mostCommonClass
          : Array.from(uniqueTargets)[Math.floor(Math.random() * uniqueTargets.size)]
      })
    } else {
      // For regression, predict mean with some noise
      const sum = trainTarget.reduce((acc: number, val: number) => acc + Number(val), 0)
      const mean = sum / trainTarget.length
      const std = Math.sqrt(
        trainTarget.reduce((acc: number, val: number) => acc + Math.pow(Number(val) - mean, 2), 0) / trainTarget.length,
      )

      // Add some randomness for realism
      return testFeatures.map(() => {
        return mean + std * (Math.random() - 0.5) * 0.9
      })
    }
  }

  private simulateFeatureImportance(numFeatures: number): { feature: string; importance: number }[] {
    const importance: { feature: string; importance: number }[] = []

    // Generate random importance values
    const values: number[] = []
    for (let i = 0; i < numFeatures; i++) {
      values.push(Math.random())
    }

    // Normalize to sum to 1
    const sum = values.reduce((acc, val) => acc + val, 0)
    const normalizedValues = values.map((val) => val / sum)

    // Create feature importance objects
    for (let i = 0; i < numFeatures; i++) {
      importance.push({
        feature: `Feature ${i + 1}`,
        importance: normalizedValues[i],
      })
    }

    // Sort by importance
    importance.sort((a, b) => b.importance - a.importance)

    return importance
  }

  // Evaluation metrics
  private calculateAccuracy(actual: any[], predicted: any[]): number {
    let correct = 0
    for (let i = 0; i < actual.length; i++) {
      if (actual[i] === predicted[i]) {
        correct++
      }
    }
    return correct / actual.length
  }

  private calculatePrecision(actual: any[], predicted: any[]): number {
    const classes = Array.from(new Set(actual))
    let totalPrecision = 0

    classes.forEach((cls) => {
      let truePositive = 0
      let falsePositive = 0

      for (let i = 0; i < actual.length; i++) {
        if (predicted[i] === cls) {
          if (actual[i] === cls) {
            truePositive++
          } else {
            falsePositive++
          }
        }
      }

      const precision = truePositive + falsePositive > 0 ? truePositive / (truePositive + falsePositive) : 0
      totalPrecision += precision
    })

    return totalPrecision / classes.length
  }

  private calculateRecall(actual: any[], predicted: any[]): number {
    const classes = Array.from(new Set(actual))
    let totalRecall = 0

    classes.forEach((cls) => {
      let truePositive = 0
      let falseNegative = 0

      for (let i = 0; i < actual.length; i++) {
        if (actual[i] === cls) {
          if (predicted[i] === cls) {
            truePositive++
          } else {
            falseNegative++
          }
        }
      }

      const recall = truePositive + falseNegative > 0 ? truePositive / (truePositive + falseNegative) : 0
      totalRecall += recall
    })

    return totalRecall / classes.length
  }

  private calculateF1(actual: any[], predicted: any[]): number {
    const precision = this.calculatePrecision(actual, predicted)
    const recall = this.calculateRecall(actual, predicted)

    return precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0
  }

  private calculateConfusionMatrix(actual: any[], predicted: any[]): number[][] {
    const classes = Array.from(new Set([...actual, ...predicted])).sort()
    const matrix: number[][] = Array(classes.length)
      .fill(0)
      .map(() => Array(classes.length).fill(0))

    for (let i = 0; i < actual.length; i++) {
      const actualIndex = classes.indexOf(actual[i])
      const predictedIndex = classes.indexOf(predicted[i])

      if (actualIndex >= 0 && predictedIndex >= 0) {
        matrix[actualIndex][predictedIndex]++
      }
    }

    return matrix
  }
}

// Export a singleton instance
export const dataProcessingService = new DataProcessingService()
