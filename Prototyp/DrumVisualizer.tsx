import React, { useState, useEffect } from 'react';
import {Card, CardContent, CardHeader, CardTitle} from "@/components/ui/card.tsx";

interface DrumNote {
    name: string;
    midiNote: number;
    color: string;
    isPlaying: boolean;
}

const DRUM_TYPES: DrumNote[] = [
    { name: 'OPENHAT', midiNote: 46, color: 'bg-red-500', isPlaying: false },
    { name: 'HIHAT', midiNote: 42, color: 'bg-orange-500', isPlaying: false },
    { name: 'KICK', midiNote: 36, color: 'bg-yellow-500', isPlaying: false },
    { name: 'SNARE', midiNote: 38, color: 'bg-green-400', isPlaying: false },
    { name: 'COWBELL', midiNote: 56, color: 'bg-emerald-500', isPlaying: false },
    { name: 'CRASH', midiNote: 49, color: 'bg-blue-500', isPlaying: false },
    { name: 'HI TOM', midiNote: 50, color: 'bg-purple-500', isPlaying: false },
    { name: 'LOW TOM', midiNote: 45, color: 'bg-pink-500', isPlaying: false },
];

const DrumVisualizer: React.FC = () => {
    const [drums, setDrums] = useState<DrumNote[]>(DRUM_TYPES);

    useEffect(() => {
        const handleNoteEvent = (event: CustomEvent) => {
            const { note } = event.detail;

            setDrums(prevDrums =>
                prevDrums.map(drum => ({
                    ...drum,
                    isPlaying: drum.midiNote === note
                }))
            );

            setTimeout(() => {
                setDrums(prevDrums =>
                    prevDrums.map(drum => ({
                        ...drum,
                        isPlaying: false
                    }))
                );
            }, 100);
        };

        window.addEventListener('midiNote', handleNoteEvent as EventListener);
        return () => {
            window.removeEventListener('midiNote', handleNoteEvent as EventListener);
        };
    }, []);

    return (
        <Card className="w-full mx-auto bg-white">
            <CardHeader>
                <CardTitle>Drum Pattern Visualizer</CardTitle>
            </CardHeader>
            <CardContent>
            <div className="space-y-2">
                <div className="grid grid-cols-8 gap-2">
                    {drums.map((drum, index) => (
                        <div key={index} className="flex flex-col items-center space-y-2">
                            <div
                                className={`w-12 h-12 rounded transition-opacity duration-100
                                    ${drum.color} 
                                    ${drum.isPlaying ? 'opacity-100' : 'opacity-20'}`}
                            />
                            <div className="text-xs text-gray-600 font-medium text-center">
                                {drum.name}
                            </div>
                            <div className="text-xs text-gray-400">
                                {drum.midiNote}
                            </div>
                        </div>
                    ))}
                </div>
            </div>
            </CardContent>
        </Card>
    );
};

export default DrumVisualizer;