import AudioRecorder from "@/components/AudioRecorder"
import SongsShowcase from "@/components/SongShowcase"
import { MusicIcon } from "lucide-react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-b from-background to-muted/50">
      <main className="flex-1 container mx-auto px-4 py-12">
        <div className="flex flex-col items-center justify-center mb-12 text-center">
          <div className="flex items-center gap-2 mb-4">
            <div className="bg-primary/10 p-3 rounded-full">
              <MusicIcon className="h-8 w-8 text-primary" />
            </div>
          </div>
          <h1 className="text-4xl font-bold tracking-tight mb-2">Reverse Engineering Shazam</h1>
          <p className="text-muted-foreground max-w-md">
            Record a sample of music from a predefined list and my algorithm will identify the song for you.
          </p>
        </div>

        <Tabs defaultValue="recorder" className="max-w-5xl mx-auto">
          <div className="flex justify-center mb-6">
            <TabsList>
              <TabsTrigger value="recorder">Audio Recorder</TabsTrigger>
              <TabsTrigger value="songs">Available Songs</TabsTrigger>
            </TabsList>
          </div>

          <TabsContent value="recorder" className="flex justify-center">
            <div className="max-w-md w-full">
              <AudioRecorder />
            </div>
          </TabsContent>

          <TabsContent value="songs">
            <SongsShowcase />
          </TabsContent>
        </Tabs>
      </main>

      <footer className="py-6 text-center text-sm text-muted-foreground border-t bg-background">
        <div className="container">
          <p>{new Date().getFullYear()} Heet Patel. N1094777. Department of Maths</p>
        </div>
      </footer>
    </div>
  )
}

