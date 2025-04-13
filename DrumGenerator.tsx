import React, { useState, useEffect } from 'react';
import * as mm from '@magenta/music';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card.tsx';
import { Button } from '@/components/ui/button.tsx';
import { Play, Square, Save } from 'lucide-react';

const EMOTION_CONFIGS = {
    happy: {
        temperature: 1.2,
        qpm: 130,
        baseVelocity: 100,
        density: 0.8,
        swing: 0.2,
        pattern: [
            [36, 42], [38, 42], [36, 42], [38, 42],
            [36, 42], [38, 42], [36, 42], [38, 42],
            [36, 42], [38, 42], [36, 42], [38, 42],
            [36, 42], [38, 42], [36, 42], [38, 42]
        ]
    },
    sad: {
        temperature: 0.8,
        qpm: 80,
        baseVelocity: 70,
        density: 0.5,
        swing: 0.1,
        pattern: [
            [36], [], [38], [],
            [36], [], [38], [],
            [36], [], [38], [],
            [36], [], [38], []
        ]
    },
    angry: {
        temperature: 1.4,
        qpm: 135,
        baseVelocity: 120,
        density: 1.0,
        swing: 0.05,
        pattern: [
            [36, 42, 49], [36, 38], [36, 42, 49], [38, 42],
            [36, 42, 49], [36, 38], [36, 42, 49], [38, 42],
            [36, 42, 49], [36, 38], [36, 42, 49], [38, 42],
            [36, 42, 49], [36, 38], [36, 42, 49], [38, 42]
        ]
    },
    disgusted: {
        temperature: 1.6,
        qpm: 90,
        baseVelocity: 90,
        density: 0.55,
        swing: 0.2,
        pattern: [
            [36, 49], [], [38, 46], [36], [42, 49],
            [36, 49], [], [38, 46], [36], [42, 49],
            [36, 49], [], [38, 46], [36], [42, 49],
            [36, 49], [], [38, 46], [36], [42, 49]
        ]
    },
    fearful: {
        temperature: 1.3,
        qpm: 100,
        baseVelocity: 75,
        density: 0.4,
        swing: 0.3,
        pattern: [
            [36, 42], [], [38, 46], [],
            [36, 42], [], [38, 46], [],
            [36, 42], [], [38, 46], [],
            [36, 42], [], [38, 46], []
        ]
    },
    surprised: {
        temperature: 1.5,
        qpm: 115,
        baseVelocity: 95,
        density: 0.7,
        swing: 0.25,
        pattern: [
            [36, 49], [42, 38], [36, 46], [42, 38],
            [36, 49], [42, 38], [36, 46], [42, 38],
            [36, 49], [42, 38], [36, 46], [42, 38],
            [36, 49], [42, 38], [36, 46], [42, 38]
        ]
    },
    neutral: {
        temperature: 1.0,
        qpm: 110,
        baseVelocity: 85,
        density: 0.6,
        swing: 0.15,
        pattern: [
            [36, 42], [], [38, 42], [],
            [36, 42], [], [38, 42], [],
            [36, 42], [], [38, 42], [],
            [36, 42], [], [38, 42], []
        ]
    }
};

const DrumGenerator: React.FC = () => {
    const [model, setModel] = useState<mm.MusicVAE | null>(null);
    const [isLoading, setIsLoading] = useState<boolean>(true);
    const [isPlaying, setIsPlaying] = useState<boolean>(false);
    const [player, setPlayer] = useState<mm.Player | null>(null);
    const [sequence, setSequence] = useState<mm.INoteSequence | null>(null);
    const [statusMessage, setStatusMessage] = useState<string>('Initializing...');
    const [currentEmotion, setCurrentEmotion] = useState<string>('');

    const [stats, setStats] = useState({
        avgVelocity: 0,
        noteDistribution: {},
        patternDensity: 0,
        swingFactor: 0
    });

    useEffect(() => {
        let mounted = true;

        async function initialize() {
            try {
                setStatusMessage('Initializing MusicVAE...');
                console.log('Starting initialization...');

                const grooveVAE = new mm.MusicVAE(
                    'https://storage.googleapis.com/magentadata/js/checkpoints/music_vae/drums_2bar_hikl_small'
                );

                await grooveVAE.initialize();
                console.log('Model initialized successfully');

                const drumPlayer = new mm.Player(false, {
                    run: (note: mm.NoteSequence.Note) => {
                        console.log('Playing note:', note);
                        window.dispatchEvent(new CustomEvent('midiNote', {
                            detail: {
                                note: note.pitch,
                                velocity: note.velocity
                            }
                        }));
                    },
                    stop: () => {
                        if (mounted) setIsPlaying(false);
                    }
                });

                if (mounted) {
                    setModel(grooveVAE);
                    setPlayer(drumPlayer);
                    createInitialSequence('neutral');
                    setIsLoading(false);
                    setStatusMessage('Ready to generate drums!');
                }
            } catch (error) {
                console.error('Initialization error:', error);
                setStatusMessage(`Error: ${error instanceof Error ? error.message : 'Failed to initialize'}`);
                setIsLoading(false);
            }
        }

        initialize();

        return () => {
            mounted = false;
            if (player) {
                player.stop();
            }
        };
    }, []);

    const createInitialSequence = (emotion: string) => {
        const config = EMOTION_CONFIGS[emotion as keyof typeof EMOTION_CONFIGS] || EMOTION_CONFIGS.neutral;

        const initialSequence: mm.INoteSequence = {
            notes: config.pattern.flatMap((pitches, i) =>
                pitches.map(pitch => ({
                    pitch,
                    quantizedStartStep: i,
                    quantizedEndStep: i + 1,
                    isDrum: true,
                    velocity: config.baseVelocity
                }))
            ),
            quantizationInfo: { stepsPerQuarter: 4 },
            tempos: [{ time: 0, qpm: config.qpm }],
            totalQuantizedSteps: 16
        };

        setSequence(initialSequence);
    };

    useEffect(() => {
        const handleFaceDetection = (event: CustomEvent) => {
            const expressions = event.detail.expressions;
            if (!expressions) return;

            console.log("expressions", expressions)

            const [dominantEmotion] = Object.entries(expressions)
                //@ts-ignore
                .reduce((a, b) => a[1] > b[1] ? a : b);

            console.log("dominantEmotion", dominantEmotion)

            setCurrentEmotion(dominantEmotion);
            const config = EMOTION_CONFIGS[dominantEmotion as keyof typeof EMOTION_CONFIGS] || EMOTION_CONFIGS.neutral;

            if (sequence) {
                const updatedSequence = {
                    ...sequence,
                    tempos: [{ time: 0, qpm: config.qpm }],
                    notes: sequence.notes.map(note => ({
                        ...note,
                        velocity: Math.floor(config.baseVelocity * (0.9 + Math.random() * 0.2))
                    }))
                };
                setSequence(updatedSequence);
            } else {
                createInitialSequence(dominantEmotion);
            }
        };

        window.addEventListener('faceDetection', handleFaceDetection as EventListener);
        return () => {
            window.removeEventListener('faceDetection', handleFaceDetection as EventListener);
        };
    }, [currentEmotion, sequence]);

    const generateSequence = async () => {
        if (!model || !sequence) return;

        try {
            setIsLoading(true);
            const config = EMOTION_CONFIGS[currentEmotion as keyof typeof EMOTION_CONFIGS] || EMOTION_CONFIGS.neutral;

            const sequences = await model.sample(4, 1.0);
            if (!sequences || sequences.length === 0) {
                throw new Error('Failed to generate sequences');
            }

            const selectedSequence = sequences[Math.floor(Math.random() * sequences.length)];

            const newNotes = selectedSequence.notes
                .filter(() => Math.random() < config.density)
                .map(note => ({
                    ...note,
                    isDrum: true,
                    velocity: Math.floor(config.baseVelocity * (0.9 + Math.random() * 0.2)),
                    quantizedStartStep: Math.min(15, Math.max(0,

                        Math.round(note.quantizedStartStep +
                            (note.quantizedStartStep % 2 === 1 ? config.swing : 0))
                    )),
                    quantizedEndStep: Math.min(16, Math.max(1,
                        Math.round(note.quantizedStartStep +
                            (note.quantizedStartStep % 2 === 1 ? config.swing : 0) + 1)
                    ))
                }));

            const patternNotes = config.pattern
                .flatMap((pitches, i) =>
                    pitches.filter(() => Math.random() < 0.4)
                        .map(pitch => ({
                            pitch,
                            quantizedStartStep: i,
                            quantizedEndStep: i + 1,
                            isDrum: true,
                            velocity: config.baseVelocity
                        }))
                );

            const newSequence: mm.INoteSequence = {
                notes: [...newNotes, ...patternNotes],
                quantizationInfo: { stepsPerQuarter: 4 },
                tempos: [{ time: 0, qpm: config.qpm }],
                totalQuantizedSteps: 16
            };

            setSequence(newSequence);
            setStatusMessage(`Generated new ${currentEmotion} pattern!`);
        } catch (error) {
            console.error('Generation error:', error);
            setStatusMessage(`Generation error: ${error instanceof Error ? error.message : 'Failed to generate'}`);
        } finally {
            setIsLoading(false);
        }
    };

    const togglePlay = () => {
        if (!player || !sequence) return;

        if (isPlaying) {
            player.stop();
            setIsPlaying(false);
        } else {
            setStatusMessage('Playing pattern...');
            player.start(sequence)
                .then(() => {
                    setStatusMessage('Ready to generate or play again!');
                    setIsPlaying(false);
                })
                .catch((error) => {
                    console.error('Playback error:', error);
                    setStatusMessage(`Playback error: ${error instanceof Error ? error.message : 'Failed to play'}`);
                    setIsPlaying(false);
                });
            setIsPlaying(true);
        }
    };

    const saveMidi = () => {
        if (!sequence) return;

        try {
            mm.sequenceProtoToMidi(sequence).then((midi) => {
                const file = new Blob([midi], { type: 'audio/midi' });
                const url = window.URL.createObjectURL(file);
                const a = document.createElement('a');
                a.href = url;
                a.download = `groove_${currentEmotion}_${new Date().getTime()}.midi`;
                a.click();
                window.URL.revokeObjectURL(url);
                setStatusMessage('MIDI file saved!');
            });
        } catch (error) {
            console.error('Save error:', error);
            setStatusMessage(`Save error: ${error instanceof Error ? error.message : 'Failed to save'}`);
        }
    };

    const calculateStats = (seq) => {
        if (!seq?.notes?.length) return;

        const avgVelocity = seq.notes.reduce((sum, note) => sum + note.velocity, 0) / seq.notes.length;

        const noteCount = seq.notes.reduce((acc, note) => {
            acc[note.pitch] = (acc[note.pitch] || 0) + 1;
            return acc;
        }, {});

        const density = seq.notes.length / seq.totalQuantizedSteps;

        const currentConfig = EMOTION_CONFIGS[currentEmotion as keyof typeof EMOTION_CONFIGS] || EMOTION_CONFIGS.neutral;
        const swing = currentConfig.swing || 0;

        setStats({
            avgVelocity: Math.round(avgVelocity),
            noteDistribution: noteCount,
            patternDensity: Math.round(density * 100),
            swingFactor: Math.round(swing * 100)
        });
    };

    useEffect(() => {
        if (sequence) {
            calculateStats(sequence);
        }
    }, [sequence, currentEmotion]);

    return (
        <Card className="w-full max-w-xl mx-auto h-96">
            <CardHeader className="h-20">
                <CardTitle>Drum Generator</CardTitle>
            </CardHeader>
            <CardContent className="h-[calc(100%-5rem)] flex flex-col">
                <div className="flex space-x-4 mb-4">
                    <Button
                        onClick={generateSequence}
                        disabled={isLoading || !model}
                        className="flex-1"
                    >
                        Generate
                    </Button>

                    <Button
                        onClick={togglePlay}
                        disabled={isLoading || !sequence}
                        className="flex-1"
                    >
                        {isPlaying ? <Square className="w-4 h-4 mr-2" /> : <Play className="w-4 h-4 mr-2" />}
                        {isPlaying ? 'Stop' : 'Play'}
                    </Button>

                    <Button
                        onClick={saveMidi}
                        disabled={!sequence}
                        variant="outline"
                        className="flex-1"
                    >
                        <Save className="w-4 h-4 mr-2" />
                        Save MIDI
                    </Button>
                </div>

                <div className="flex-1 flex flex-col items-center justify-center">
                    <div className="text-center text-sm mb-4">
                        {statusMessage}
                    </div>

                    {sequence && (
                        <div className="text-sm">
                            <div className="grid grid-cols-2 gap-x-12 gap-y-2">
                                <div className="flex justify-between space-x-4">
                                    <span className="text-gray-500">Emotion:</span>
                                    <span>{currentEmotion}</span>
                                </div>
                                <div className="flex justify-between space-x-4">
                                    <span className="text-gray-500">Avg Velocity:</span>
                                    <span>{stats.avgVelocity}</span>
                                </div>
                                <div className="flex justify-between space-x-4">
                                    <span className="text-gray-500">Tempo:</span>
                                    <span>{sequence.tempos?.[0].qpm} BPM</span>
                                </div>
                                <div className="flex justify-between space-x-4">
                                    <span className="text-gray-500">Density:</span>
                                    <span>{stats.patternDensity}%</span>
                                </div>
                                <div className="flex justify-between space-x-4">
                                    <span className="text-gray-500">Pattern Length:</span>
                                    <span>{sequence.notes?.length || 0} notes</span>
                                </div>
                                <div className="flex justify-between space-x-4">
                                    <span className="text-gray-500">Swing:</span>
                                    <span>{stats.swingFactor}%</span>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </CardContent>
        </Card>
);
};

export default DrumGenerator;