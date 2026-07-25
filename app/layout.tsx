import type { Metadata, Viewport } from "next";
import { Fraunces, Inter, JetBrains_Mono, Newsreader } from "next/font/google";
import "./globals.css";
import { Toaster } from "sonner";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  weight: ["400", "500"],
  display: "swap",
});

const fraunces = Fraunces({
  variable: "--font-fraunces",
  subsets: ["latin"],
  weight: ["500", "600"],
  display: "swap",
});

const newsreader = Newsreader({
  variable: "--font-newsreader",
  subsets: ["latin"],
  weight: ["400", "500"],
  style: ["normal", "italic"],
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-jetbrains",
  subsets: ["latin"],
  weight: ["400", "500"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "AI Engineering Interview Prep",
  description: "Comprehensive interview preparation guide with 456 questions across 14 topics: LLMs, RAG, Agents, Fine-Tuning, Vector DBs, System Design, and more.",
  icons: {
    icon: "/favicon.ico",
  },
};

export const viewport: Viewport = {
  themeColor: "#131110",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${inter.variable} ${fraunces.variable} ${newsreader.variable} ${jetbrainsMono.variable} h-full antialiased`}
    >
      <body className="min-h-screen overflow-x-hidden bg-page text-ink-mid font-normal leading-relaxed">
        {children}
        <Toaster
          position="top-center"
          theme="dark"
          closeButton
          toastOptions={{
            style: {
              background: "var(--surface-3)",
              border: "1px solid var(--border-strong)",
              color: "var(--text-hi)",
              borderRadius: "12px",
              boxShadow: "var(--shadow-popover)",
              fontFamily: "var(--font-inter), ui-sans-serif, system-ui, sans-serif",
              fontSize: "13.5px",
            },
          }}
        />
      </body>
    </html>
  );
}
