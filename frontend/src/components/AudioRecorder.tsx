"use client"

import { useState, useRef, useEffect } from "react"
import { Mic, Square, Loader2, Volume2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription } from "@/components/ui/alert"

interface MatchResult {
    song_id: number
    song_name: string
    confidence: number
    offset: number
    match_count: number
}

interface RecognitionResult {
    matches: MatchResult
}

export default function AudioRecorder() {
    const [recording, setRecording] = useState(false)
    const [audioUrl, setAudioUrl] = useState<string>("")
    const [result, setResult] = useState<RecognitionResult | null>(null)
    const [isLoading, setIsLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [recordingTime, setRecordingTime] = useState(0)
    const mediaRecorderRef = useRef<MediaRecorder | null>(null)
    const chunksRef = useRef<Blob[]>([])
    const timerRef = useRef<NodeJS.Timeout | null>(null)

    useEffect(() => {
        if (recording) {
            timerRef.current = setInterval(() => {
                setRecordingTime((prev) => prev + 1)
            }, 1000)
        } else {
            if (timerRef.current) {
                clearInterval(timerRef.current)
                timerRef.current = null
            }
            setRecordingTime(0)
        }

        return () => {
            if (timerRef.current) {
                clearInterval(timerRef.current)
            }
        }
    }, [recording])

    const startRecording = async () => {
        try {
            setError(null)
            setResult(null)
            setAudioUrl("")

            const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
            const mediaRecorder = new MediaRecorder(stream)
            mediaRecorderRef.current = mediaRecorder
            chunksRef.current = []

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    chunksRef.current.push(event.data)
                }
            }

            mediaRecorder.onstop = handleStop

            mediaRecorder.start()
            setRecording(true)
        } catch (err) {
            console.error("Error accessing microphone:", err)
            setError("Could not access microphone. Please check your permissions.")
        }
    }

    const stopRecording = () => {
        if (mediaRecorderRef.current) {
            mediaRecorderRef.current.stop()
            setRecording(false)

            mediaRecorderRef.current.stream.getTracks().forEach((track) => track.stop())
        }
    }

    const handleStop = async () => {
        const blob = new Blob(chunksRef.current, { type: "audio/wav" })
        const url = URL.createObjectURL(blob)
        setAudioUrl(url)

        const formData = new FormData()
        formData.append("file", blob, "recording.wav")
        formData.append("duration", recordingTime.toString())

        try {
            setIsLoading(true)
            const response = await fetch("http://localhost:8000/identify", {
                method: "POST",
                body: formData,
            })

            if (!response.ok) {
                throw new Error(`Server responded with status: ${response.status}`)
            }

            const data = await response.json()
            console.log("API response:", data)

            setResult(data)
        } catch (err) {
            console.error("Error sending audio to backend:", err)
            setError("Failed to identify audio. Please try again.")
        } finally {
            setIsLoading(false)
        }
    }

    const formatTime = (seconds: number): string => {
        const mins = Math.floor(seconds / 60)
        const secs = seconds % 60
        return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`
    }

    return (
        <Card className="w-full max-w-md mx-auto">
            <CardHeader>
                <CardTitle className="flex items-center gap-2">
                    <Volume2 className="h-5 w-5" />
                    Audio Recognition
                </CardTitle>
                <CardDescription>Record audio to identify songs and music</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
                {error && (
                    <Alert variant="destructive" className="mb-4">
                        <AlertDescription>{error}</AlertDescription>
                    </Alert>
                )}

                <div className="flex flex-col items-center gap-4">
                    <div className={`relative rounded-full p-4 ${recording ? "bg-red-100 dark:bg-red-900/20" : "bg-muted"}`}>
                        <div
                            className={`absolute inset-0 rounded-full ${recording ? "animate-pulse bg-red-100/50 dark:bg-red-900/10" : ""}`}
                        ></div>
                        <Button
                            variant={recording ? "destructive" : "default"}
                            size="icon"
                            className="relative h-16 w-16 rounded-full"
                            onClick={recording ? stopRecording : startRecording}
                            disabled={isLoading}
                        >
                            {recording ? <Square className="h-6 w-6" /> : <Mic className="h-6 w-6" />}
                        </Button>
                    </div>

                    {recording && (
                        <div className="text-center">
                            <div className="text-sm text-muted-foreground">Recording</div>
                            <div className="text-xl font-semibold">{formatTime(recordingTime)}</div>
                        </div>
                    )}

                    {audioUrl && !isLoading && !recording && (
                        <div className="w-full">
                            <p className="text-sm text-muted-foreground mb-2">Your recording:</p>
                            <audio controls src={audioUrl} className="w-full"></audio>
                        </div>
                    )}

                    {isLoading && (
                        <div className="flex flex-col items-center gap-2 py-4">
                            <Loader2 className="h-8 w-8 animate-spin text-primary" />
                            <p className="text-sm text-muted-foreground">Identifying audio...</p>
                        </div>
                    )}
                </div>

                {result && result.matches && (
                    <div className="mt-6">
                        <h3 className="font-medium mb-3">Matching Result:</h3>
                        <div className="border rounded-lg p-3">
                            <div className="flex justify-between items-center mb-2">
                                <h4 className="font-medium">{result.matches.song_name}</h4>
                                <Badge variant={result.matches.confidence * 100 > 80 ? "default" : "outline"}>
                                    {(result.matches.confidence * 100).toFixed(0)}%
                                </Badge>
                            </div>
                            <Progress value={result.matches.confidence * 100} className="h-2" />
                            <div className="mt-3 grid grid-cols-2 gap-2 text-sm">
                                <div>
                                    <span className="text-muted-foreground">Song ID:</span> {result.matches.song_id}
                                </div>
                                <div>
                                    <span className="text-muted-foreground">Match Count:</span> {result.matches.match_count}
                                </div>
                                <div className="col-span-2">
                                    <span className="text-muted-foreground">Offset:</span> {result.matches.offset}s
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {result && !result.matches && (
                    <Alert className="mt-4">
                        <AlertDescription>No matches found. Try recording a longer sample.</AlertDescription>
                    </Alert>
                )}
            </CardContent>
            <CardFooter className="flex justify-center text-sm text-muted-foreground">
                {recording ? "Click stop when you're done recording" : "Click the microphone to start recording"}
            </CardFooter>
        </Card>
    )
}

