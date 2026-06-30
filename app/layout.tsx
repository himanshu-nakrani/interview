import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { Toaster } from "sonner";

const inter = Inter({
  variable: "--font-sans",
  subsets: ["latin"],
  weight: ["400", "500"],
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
  weight: ["400", "500"],
});

export const metadata: Metadata = {
  title: "AI Engineering Interview Prep",
  description: "Comprehensive interview preparation guide with 456 questions across 14 topics: LLMs, RAG, Agents, Fine-Tuning, Vector DBs, System Design, and more.",
  icons: {
    icon: "/favicon.ico",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${inter.variable} ${jetbrainsMono.variable} h-full antialiased dark`}
    >
      <body className="min-h-screen overflow-x-hidden bg-zinc-950 text-zinc-300 font-normal leading-relaxed">
        {children}
        <Toaster position="top-center" richColors closeButton />
      </body>
    </html>
  );
}
