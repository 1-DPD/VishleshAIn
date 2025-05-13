"use client"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { MessageCircle, Send, X, Minimize2, Maximize2 } from "lucide-react"
import { cn } from "@/lib/utils"
import { useToast } from "@/hooks/use-toast"

type Message = {
  role: "user" | "assistant"
  content: string
}

// Mock responses for the chatbot when API key is not available
const mockResponses = [
  "I'm analyzing your data now. The dataset shows some interesting patterns!",
  "Based on your data, I recommend cleaning the null values and normalizing the numeric columns.",
  "Your data seems suitable for a regression model. Would you like me to explain why?",
  "I've detected some outliers in your dataset. Consider using IQR or z-score methods to handle them.",
  "The visualizations suggest a strong correlation between variables X and Y.",
  "Your model performance could be improved by feature engineering or trying different algorithms.",
  "I've completed the analysis. You can view the detailed report in the Reports section.",
  "The data preprocessing is complete. Would you like to proceed with the analysis?",
  "Based on the model evaluation, the Random Forest algorithm performed best with 92% accuracy.",
  "I've generated visualizations that highlight the key insights from your data.",
]

export function Chatbot() {
  const [isOpen, setIsOpen] = useState(false)
  const [isMinimized, setIsMinimized] = useState(false)
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content: "Hello! I'm your VishleshAIn assistant. How can I help you with your data analysis today?",
    },
  ])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const { toast } = useToast()

  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" })
    }
  }, [messages])

  const handleSendMessage = async () => {
    if (!input.trim()) return

    const userMessage = { role: "user" as const, content: input }
    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setIsLoading(true)

    try {
      // Check if we're in a browser environment and if the OpenAI API key is available
      const hasApiKey =
        typeof window !== "undefined" &&
        (process.env.NEXT_PUBLIC_OPENAI_API_KEY || window.localStorage.getItem("OPENAI_API_KEY"))

      if (!hasApiKey) {
        // If no API key is available, use mock responses
        setTimeout(() => {
          const randomIndex = Math.floor(Math.random() * mockResponses.length)
          const mockResponse = mockResponses[randomIndex]
          setMessages((prev) => [...prev, { role: "assistant", content: mockResponse }])
          setIsLoading(false)
        }, 1000)
        return
      }

      // If we have an API key, try to use the OpenAI API
      // We need to dynamically import these to avoid the error when the API key is not available
      const { generateText } = await import("ai")
      const { openai } = await import("@ai-sdk/openai")

      // Construct the prompt with conversation history
      const conversationHistory = messages
        .map((msg) => `${msg.role === "user" ? "User" : "Assistant"}: ${msg.content}`)
        .join("\n")

      const prompt = `${conversationHistory}\nUser: ${input}\nAssistant:`

      const apiKey = process.env.NEXT_PUBLIC_OPENAI_API_KEY || window.localStorage.getItem("OPENAI_API_KEY")

      const { text } = await generateText({
        model: openai("gpt-4o", { apiKey }),
        prompt,
        system:
          "You are VishleshAIn, a helpful data analytics assistant. Provide concise, accurate responses about data preprocessing, analysis, modeling, and visualization.",
      })

      setMessages((prev) => [...prev, { role: "assistant", content: text }])
    } catch (error) {
      console.error("Error generating response:", error)
      toast({
        title: "API Error",
        description: "Could not connect to the AI service. Using simulated responses instead.",
        variant: "destructive",
      })

      // Fallback to mock response
      const randomIndex = Math.floor(Math.random() * mockResponses.length)
      const mockResponse = mockResponses[randomIndex]
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: mockResponse,
        },
      ])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <>
      <Button
        className="fixed bottom-4 right-4 rounded-full h-12 w-12 p-0 shadow-lg bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 transition-all duration-300"
        onClick={() => {
          setIsOpen(!isOpen)
          setIsMinimized(false)
        }}
      >
        {isOpen ? <X className="h-5 w-5" /> : <MessageCircle className="h-5 w-5" />}
      </Button>

      {isOpen && (
        <Card
          className={cn(
            "fixed bottom-20 right-4 w-80 md:w-96 shadow-lg transition-all duration-300 ease-in-out border border-purple-200 dark:border-purple-800",
            isMinimized ? "h-14" : "h-96",
          )}
        >
          <CardHeader className="p-3 border-b flex flex-row items-center justify-between bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-950 dark:to-purple-950">
            <CardTitle className="text-sm font-medium">VishleshAIn Assistant</CardTitle>
            <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => setIsMinimized(!isMinimized)}>
              {isMinimized ? <Maximize2 className="h-4 w-4" /> : <Minimize2 className="h-4 w-4" />}
            </Button>
          </CardHeader>

          {!isMinimized && (
            <>
              <CardContent className="p-0">
                <ScrollArea className="h-72 p-4">
                  {messages.map((message, index) => (
                    <div
                      key={index}
                      className={cn(
                        "mb-4 max-w-[80%] rounded-lg p-3",
                        message.role === "user"
                          ? "ml-auto bg-gradient-to-r from-blue-500 to-purple-600 text-white"
                          : "bg-muted",
                      )}
                    >
                      {message.content}
                    </div>
                  ))}
                  <div ref={messagesEndRef} />
                </ScrollArea>
              </CardContent>

              <CardFooter className="p-3 pt-0">
                <div className="flex w-full items-center space-x-2">
                  <Input
                    placeholder="Type your message..."
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !isLoading) {
                        handleSendMessage()
                      }
                    }}
                    disabled={isLoading}
                    className="border-purple-200 dark:border-purple-800 focus:ring-purple-500"
                  />
                  <Button
                    size="icon"
                    onClick={handleSendMessage}
                    disabled={isLoading || !input.trim()}
                    className="bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700"
                  >
                    <Send className="h-4 w-4" />
                  </Button>
                </div>
              </CardFooter>
            </>
          )}
        </Card>
      )}
    </>
  )
}
