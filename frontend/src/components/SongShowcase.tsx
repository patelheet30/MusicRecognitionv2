"use client"

import { useState, useMemo } from 'react';
import { Search, Music, ExternalLink, ArrowLeft } from 'lucide-react';
import { type Song, songList } from '@/lib/songs-data';
import { CardContent, Card } from './ui/card';
import { Input } from './ui/input';
import { Tabs, TabsList, TabsTrigger } from './ui/tabs';
import { Button } from './ui/button';

type FilterView = 'song' | 'artist' | 'album';
type FilterSelection = { type: FilterView, value: string | null };

export default function SongsShowcase() {
    const [searchQuery, setSearchQuery] = useState('');
    const [view, setView] = useState<'grid' | 'list'>('grid');
    const [filterSelection, setFilterSelection] = useState<FilterSelection>({ 
        type: 'song', 
        value: null 
    });

    const uniqueArtists = useMemo(() => {
        const artists = new Set(songList.map(song => song.artist));
        return Array.from(artists).sort();
    }, []);

    const uniqueAlbums = useMemo(() => {
        const albums = new Set(songList.map(song => song.album));
        return Array.from(albums).sort();
    }, []);

    const filteredSongs = useMemo(() => {
        let result = songList;

        if (filterSelection.value) {
            if (filterSelection.type === 'artist') {
                result = result.filter(song => song.artist === filterSelection.value);
            } else if (filterSelection.type === 'album') {
                result = result.filter(song => song.album === filterSelection.value);
            }
        }

        if (searchQuery) {
            result = result.filter(song => {
                const matchesSearchQuery =
                    song.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                    song.artist.toLowerCase().includes(searchQuery.toLowerCase()) ||
                    song.album.toLowerCase().includes(searchQuery.toLowerCase());
                return matchesSearchQuery;
            });
        }

        return result;
    }, [searchQuery, filterSelection]);

    const handleFilterChange = (type: FilterView) => {
        setFilterSelection({ type, value: null });
        setSearchQuery('');
    };

    const handleBackToList = () => {
        setFilterSelection({ type: filterSelection.type, value: null });
    };

    const renderFilterContent = () => {
        if (filterSelection.value) {
            return (
                <>
                    <div className="flex items-center mb-4">
                        <Button 
                            variant="ghost" 
                            className="mr-2 p-1 h-8" 
                            onClick={handleBackToList}
                        >
                            <ArrowLeft className="h-4 w-4 mr-1" />
                            Back
                        </Button>
                        <h3 className="text-lg font-medium">
                            {filterSelection.type === 'artist' ? 'Artist: ' : 'Album: '}
                            <span className="font-bold">{filterSelection.value}</span>
                        </h3>
                    </div>
                    {renderSongs()}
                </>
            );
        }

        switch (filterSelection.type) {
            case 'artist':
                return (
                    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                        {uniqueArtists.map(artist => (
                            <Card 
                                key={artist} 
                                className="overflow-hidden transition-all hover:shadow-md hover:bg-muted/50 cursor-pointer"
                                onClick={() => setFilterSelection({ type: 'artist', value: artist })}
                            >
                                <CardContent className="p-4 flex items-center">
                                    <div className="flex-1">
                                        <h3 className="font-medium truncate" title={artist}>
                                            {artist}
                                        </h3>
                                        <p className="text-sm text-muted-foreground">
                                            {songList.filter(song => song.artist === artist).length} songs
                                        </p>
                                    </div>
                                </CardContent>
                            </Card>
                        ))}
                    </div>
                );
                
            case 'album':
                return (
                    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                        {uniqueAlbums.map(album => (
                            <Card 
                                key={album} 
                                className="overflow-hidden transition-all hover:shadow-md hover:bg-muted/50 cursor-pointer"
                                onClick={() => setFilterSelection({ type: 'album', value: album })}
                            >
                                <CardContent className="p-4 flex items-center">
                                    <div className="flex-1">
                                        <h3 className="font-medium truncate" title={album}>
                                            {album}
                                        </h3>
                                        <p className="text-sm text-muted-foreground">
                                            {songList.filter(song => song.album === album).length} songs
                                        </p>
                                    </div>
                                </CardContent>
                            </Card>
                        ))}
                    </div>
                );
                
            case 'song':
            default:
                return renderSongs();
        }
    };

    const renderSongs = () => {
        if (filteredSongs.length === 0) {
            return (
                <div className="text-center py-12">
                    <p className="text-muted-foreground">No songs found matching your criteria.</p>
                </div>
            );
        }

        return view === "grid" ? (
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
        );
    };

    return (
        <div className='w-full space-y-6'>
            <div className='flex flex-col sm:flex-row gap-4 items-center sm:justify-between'>
                <h2 className='text-2xl font-bold flex items-center gap-2'>
                    <Music className='w-5 h-5' />
                    Available Songs
                </h2>

                <div className="flex flex-wrap w-full sm:w-auto gap-2">
                    {/* Filter type selector */}
                    <Tabs 
                        value={filterSelection.type} 
                        className="mr-2"
                        onValueChange={(value) => handleFilterChange(value as FilterView)}
                    >
                        <TabsList>
                            <TabsTrigger value="song">Songs</TabsTrigger>
                            <TabsTrigger value="artist">Artists</TabsTrigger>
                            <TabsTrigger value="album">Albums</TabsTrigger>
                        </TabsList>
                    </Tabs>

                    {/* Search box */}
                    <div className="relative flex-1 sm:w-64">
                        <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
                        <Input
                            placeholder='Search...'
                            value={searchQuery}
                            className='pl-8'
                            onChange={(e) => setSearchQuery(e.target.value)}
                        />
                    </div>
                    
                    {/* Grid/List view selector */}
                    {(filterSelection.type === 'song' || filterSelection.value) && (
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
                    )}
                </div>
            </div>

            {renderFilterContent()}
        </div>
    );
}

function SongCard({ song }: { song: Song }) {
    const handleClick = () => {
        window.open(song.url, '_blank', 'noopener,noreferrer');
    };

    return (
        <Card 
            className="overflow-hidden transition-all hover:shadow-md hover:bg-muted/50 cursor-pointer"
            onClick={handleClick}
        >
            <CardContent className="p-4">
                <div className="flex justify-between items-start">
                    <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between">
                            <h3 className="font-medium truncate" title={song.title}>
                                {song.title}
                            </h3>
                            <ExternalLink className="h-4 w-4 text-muted-foreground ml-2" />
                        </div>
                        <p className="text-sm text-muted-foreground truncate" title={song.artist}>
                            {song.artist}
                        </p>
                        <p className="text-xs text-muted-foreground mt-1 truncate" title={song.album}>
                            {song.album}
                        </p>
                    </div>
                </div>
            </CardContent>
        </Card>
    );
}

function SongListItem({ song }: { song: Song }) {
    const handleClick = () => {
        window.open(song.url, '_blank', 'noopener,noreferrer');
    };

    return (
        <div 
            className="flex items-center p-3 rounded-lg border hover:bg-muted/50 transition-colors cursor-pointer"
            onClick={handleClick}
        >
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
            <ExternalLink className="h-4 w-4 text-muted-foreground flex-shrink-0 ml-2" />
        </div>
    );
}