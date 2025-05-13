import type React from "react"
import { Header } from "@/components/header"
import { Chatbot } from "@/components/chatbot"

export default function PageLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div className="flex min-h-screen flex-col">
      <Header />
      <main className="flex-1">{children}</main>
      <Chatbot />
    </div>
  )
}
