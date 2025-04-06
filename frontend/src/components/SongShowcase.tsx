"use client"

import { useState, useMemo } from 'react';
import { Search, Music } from 'lucide-react';
import { type Song, songList } from '@/lib/songs-data';
import { CardContent, Card } from './ui/card';
import { Input } from './ui/input';
import { Tabs, TabsList, TabsTrigger } from './ui/tabs';

export default function SongsShowcase() {
    const [searchQuery, setSearchQuery] = useState('');
    const [view, setView] = useState<'grid' | 'list'>('grid');

    const filteredSongs = useMemo(() => {
        return songList.filter((song) => {
            const matchesSearchQuery =
                song.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                song.artist.toLowerCase().includes(searchQuery.toLowerCase()) ||
                song.album.toLowerCase().includes(searchQuery.toLowerCase());
            return matchesSearchQuery;
        })
    }, [searchQuery]);

    return (
        <div className='w-full space-y-6'>
            <div className='flex flex-col sm:flew-row gap-4 items-center justify-center'>
                <h2 className='text-2xl font-bold flex items-center gap-2'>
                    <Music className='w-5 h-5' />
                    Available Songs
                </h2>

                <div className="flex w-full sm:w-auto gap-2">
                    <div className="relative flex-1 sm:w-64">
                        <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
                        <Input
                            placeholder='Search for songs'
                            value={searchQuery}
                            className='pl-8'
                            onChange={(e) => setSearchQuery(e.target.value)}
                        />
                    </div>
                    <Tabs defaultValue="grid" className="hidden sm:block">
                        <TabsList>
                            <TabsTrigger value="grid" onClick={() => setView("grid")}>
                                Grid
                            </TabsTrigger>
                            <TabsTrigger value="list" onClick={() => setView("list")}>
                                List
                            </TabsTrigger>
                        </TabsList>
                    </Tabs>
                </div>
            </div>

            {filteredSongs.length === 0 ? (
                <div className="text-center py-12">
                    <p className="text-muted-foreground">No songs found matching your criteria.</p>
                </div>
            ) : view === "grid" ? (
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                    {filteredSongs.map((song) => (
                        <SongCard key={song.id} song={song} />
                    ))}
                </div>
            ) : (
                <div className="space-y-2">
                    {filteredSongs.map((song) => (
                        <SongListItem key={song.id} song={song} />
                    ))}
                </div>
            )}

        </div>
    )
}

function SongCard({ song }: { song: Song }) {
    return (
        <Card className="overflow-hidden transition-all hover:shadow-md hover:bg-muted/50">
            <CardContent className="p-4">
                <h3 className="font-medium truncate" title={song.title}>
                    {song.title}
                </h3>
                <p className="text-sm text-muted-foreground truncate" title={song.artist}>
                    {song.artist}
                </p>
                <p className="text-xs text-muted-foreground mt-1 truncate" title={song.album}>
                    {song.album}
                </p>
            </CardContent>
        </Card>
    )
}

function SongListItem({ song }: { song: Song }) {
    return (
        <div className="flex items-center p-3 rounded-lg border hover:bg-muted/50 transition-colors">
            <div className="min-w-0 flex-1">
                <h3 className="font-medium truncate" title={song.title}>
                    {song.title}
                </h3>
                <div className="flex flex-col sm:flex-row sm:items-center gap-0 sm:gap-2">
                    <p className="text-sm text-muted-foreground truncate" title={song.artist}>
                        {song.artist}
                    </p>
                    <span className="hidden sm:inline text-muted-foreground">•</span>
                    <p className="text-xs text-muted-foreground truncate" title={song.album}>
                        {song.album}
                    </p>
                </div>
            </div>
        </div>
    )
}