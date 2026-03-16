import { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Play, Pause, Volume2 } from 'lucide-react';
import { Slider } from '@/components/ui/slider';
import * as Tone from 'tone';
import { Midi } from '@tonejs/midi';

const EmotionPlayer = () => {
    const [isPlaying, setIsPlaying] = useState(false);
    const [volume, setVolume] = useState(0.8);
    const [instruments, setInstruments] = useState(null);
    const [currentTrack, setCurrentTrack] = useState(null);
    const [analysis, setAnalysis] = useState(null);

    const emotions = ['happy', 'sad', 'angry', 'disgust', 'fear', 'surprise', 'neutral'];
    const models = ['MusicVAE', 'DrumsRNN', 'MidiNET', 'VampNET', 'GrooveVAE'];
    const examples = Array.from({ length: 7 }, (_, i) => `example${i + 1}`);

    useEffect(() => {
        const kick = new Tone.MembraneSynth({
            pitchDecay: 0.008,
            octaves: 2,
            oscillator: { type: 'triangle' },
            envelope: {
                attack: 0.001,
                decay: 0.2,
                sustain: 0,
                release: 0.2
            }
        }).toDestination();

        const snare = new Tone.NoiseSynth({
            noise: { type: 'pink' },
            envelope: {
                attack: 0.001,
                decay: 0.15,
                sustain: 0,
                release: 0.05
            }
        }).toDestination();

        const hihat = new Tone.MetalSynth({
            frequency: 200,
            envelope: {
                attack: 0.001,
                decay: 0.1,
                sustain: 0,
                release: 0.01
            },
            harmonicity: 5.1,
            modulationIndex: 32,
            resonance: 4000,
            octaves: 1.5
        }).toDestination();

        [kick, snare, hihat].forEach(inst => {
            inst.volume.value = Tone.gainToDb(volume);
        });
        setInstruments({ kick, snare, hihat });

        return () => {
            kick.dispose();
            snare.dispose();
            hihat.dispose();
            Tone.Transport.cancel();
            Tone.Transport.stop();
        };
    }, [volume]);

    const analyzeMidi = (midi) => {
        const notes = midi.tracks.flatMap(track => track.notes);

        // Count hits per drum type
        const drumCounts = {
            kick: notes.filter(n => n.midi === 36).length,
            snare: notes.filter(n => n.midi === 38).length,
            hihat: notes.filter(n => n.midi === 42).length
        };

        // Analyze patterns (16th note grid)
        const gridSize = 16;
        const patterns = {
            kick: new Array(gridSize).fill(0),
            snare: new Array(gridSize).fill(0),
            hihat: new Array(gridSize).fill(0)
        };

        notes.forEach(note => {
            const position = Math.floor((note.time % 1) * gridSize);
            if (note.midi === 36) patterns.kick[position]++;
            if (note.midi === 38) patterns.snare[position]++;
            if (note.midi === 42) patterns.hihat[position]++;
        });

        // Calculate tempo and timing stats
        const tempoStats = {
            bpm: midi.header.tempos[0]?.bpm || 120,
            timeSignature: `${midi.header.timeSignatures[0]?.timeSignature[0]}/${midi.header.timeSignatures[0]?.timeSignature[1]}` || '4/4'
        };

        // Calculate density (notes per second)
        const density = notes.length / midi.duration;

        return {
            drumCounts,
            patterns,
            tempoStats,
            density: density.toFixed(2),
            totalNotes: notes.length,
            duration: midi.duration.toFixed(2)
        };
    };

    const togglePlay = async (source, identifier, isExample = false) => {
        if (!instruments) return;
        const trackId = isExample ? `example-${identifier}` : `${source}-${identifier}`;
        const path = isExample
            ? `/BachelorThesisListener/midiFiles/dataset/${identifier}.mid`
            : `/BachelorThesisListener/midiFiles/${source}/${identifier}.mid`;

        console.log('Attempting to play:', path);

        if (isPlaying) {
            Tone.Transport.stop();
            Tone.Transport.cancel();
            setIsPlaying(false);
            setCurrentTrack(null);
            if (currentTrack === trackId) return;
        }

        try {
            await Tone.start();
            const response = await fetch(path);
            if (!response.ok) {
                console.error(`Failed to fetch MIDI file at ${path}`);
                return;
            }

            const midiData = await response.arrayBuffer();
            const midi = new Midi(midiData);
            console.log('MIDI loaded, tracks:', midi.tracks.length);
            console.log('Notes:', midi.tracks.flatMap(track => track.notes).length);

            setAnalysis(analyzeMidi(midi));

            const notes = midi.tracks.flatMap(track => track.notes);
            console.log('Note details:', notes.map(n => ({
                midi: n.midi,
                time: n.time,
                duration: n.duration
            })));

            if (notes.length === 0) {
                console.log('No notes found in MIDI file');
                return;
            }

            Tone.Transport.position = 0;
            Tone.Transport.timeSignature = midi.header.timeSignatures[0]?.timeSignature[0] || 4;
            Tone.Transport.bpm.value = midi.header.tempos[0]?.bpm || 120;

            setIsPlaying(true);
            setCurrentTrack(trackId);

            Tone.Transport.cancel();
            Tone.Transport.stop();

            notes.forEach(note => {
                Tone.Transport.schedule(time => {
                    switch (note.midi) {
                        case 35:
                        case 36:
                            instruments.kick.triggerAttackRelease('C1', note.duration, time, note.velocity);
                            break;
                        case 38:
                            instruments.snare.triggerAttackRelease(note.duration, time, note.velocity);
                            break;
                        case 42:
                            instruments.hihat.triggerAttackRelease(note.duration, time, note.velocity);
                            break;
                    }
                }, note.time);
            });

            Tone.Transport.start();

            setTimeout(() => {
                Tone.Transport.stop();
                setIsPlaying(false);
                setCurrentTrack(null);
            }, midi.duration * 1000);
        } catch (error) {
            console.error('Error playing MIDI:', error, 'for path:', path);
            setAnalysis(null);
        }
    };

    const renderAnalysis = () => (
        <div className="fixed bottom-0 left-0 right-0 bg-white border-t shadow-lg p-4">
            <div className="max-w-7xl mx-auto">
                <div className="grid grid-cols-3 gap-3 text-sm">
                    <div>
                        <h3 className="font-semibold text-gray-600">Drum Counts</h3>
                        {analysis ? (
                            <>
                                <p>Kick: {analysis.drumCounts.kick}</p>
                                <p>Snare: {analysis.drumCounts.snare}</p>
                                <p>Hi-hat: {analysis.drumCounts.hihat}</p>
                            </>
                        ) : (
                            <p className="text-gray-400">No file playing</p>
                        )}
                    </div>
                    <div>
                        <h3 className="font-semibold text-gray-600">Timing</h3>
                        {analysis ? (
                            <>
                                <p>BPM: {analysis.tempoStats.bpm}</p>
                                <p>Time Signature: {analysis.tempoStats.timeSignature}</p>
                                <p>Duration: {analysis.duration}s</p>
                            </>
                        ) : (
                            <p className="text-gray-400">No file playing</p>
                        )}
                    </div>
                    <div>
                        <h3 className="font-semibold text-gray-600">Statistics</h3>
                        {analysis ? (
                            <>
                                <p>Notes/Second: {analysis.density}</p>
                                <p>Total Notes: {analysis.totalNotes}</p>
                            </>
                        ) : (
                            <p className="text-gray-400">No file playing</p>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );

    const renderPlaybackControls = (identifier, source, isExample = false) => (
        <div className="flex items-center gap-4">
            <button
                onClick={() => togglePlay(source, identifier, isExample)}
                className="p-2 rounded-full hover:bg-gray-200 bg-gray-100"
            >
                {currentTrack === (isExample ? `example-${identifier}` : `${source}-${identifier}`) && isPlaying ?
                    <Pause className="w-6 h-6" /> :
                    <Play className="w-6 h-6" />
                }
            </button>
            <div className="w-24">
                <Slider
                    defaultValue={[volume * 100]}
                    max={100}
                    step={1}
                    onValueChange={(value) => {
                        const newVolume = value[0] / 100;
                        setVolume(newVolume);
                        if (instruments) {
                            Object.values(instruments).forEach(inst => {
                                inst.volume.value = Tone.gainToDb(newVolume);
                            });
                        }
                    }}
                />
            </div>
            <Volume2 className="w-4 h-4" />
        </div>
    );

    return (
        <>
            <div className="p-6 pb-20 space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <Card className="w-full">
                        <CardHeader>
                            <CardTitle className="text-xl font-bold">Dataset Examples</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="space-y-4">
                                {examples.map(example => (
                                    <div key={example}
                                         className="flex items-center justify-between gap-4 p-2 bg-gray-100 rounded">
                                        <span className="capitalize">Example {example.slice(-1)}</span>
                                        {renderPlaybackControls(example, null, true)}
                                    </div>
                                ))}
                            </div>
                        </CardContent>
                    </Card>

                    {models.map(model => (
                        <Card key={model} className="w-full">
                            <CardHeader>
                                <CardTitle className="text-xl font-bold">{model}</CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="space-y-4">
                                    {emotions.map(emotion => (
                                        <div key={emotion}
                                             className="flex items-center justify-between gap-4 p-2 bg-gray-100 rounded">
                                            <span className="capitalize">{emotion}</span>
                                            {renderPlaybackControls(emotion, model)}
                                        </div>
                                    ))}
                                </div>
                            </CardContent>
                        </Card>
                    ))}
                </div>
            </div>
            {renderAnalysis()}
        </>
    );
};

export default EmotionPlayer;
