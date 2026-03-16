import React, { useEffect, useRef, useState } from 'react';
import { FaceMesh, Results } from '@mediapipe/face_mesh';
import { Camera } from '@mediapipe/camera_utils';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card.tsx';
import { Button } from "@/components/ui/button.tsx";

const FaceDetector: React.FC = () => {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const cameraRef = useRef<Camera | null>(null);
    const faceMeshRef = useRef<FaceMesh | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [isStarted, setIsStarted] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const emotions = ['happy', 'sad', 'angry', 'surprised', 'fearful', 'disgusted', 'neutral'];
    const [currentIndex, setCurrentIndex] = useState(0);

    useEffect(() => {
        if (!videoRef.current || !canvasRef.current) return;

        const faceMesh = new FaceMesh({
            locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
            }
        });

        faceMesh.setOptions({
            maxNumFaces: 1,
            refineLandmarks: true,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });

        faceMesh.onResults(onResults);
        faceMeshRef.current = faceMesh;

        const camera = new Camera(videoRef.current, {
            onFrame: async () => {
                if (videoRef.current && faceMeshRef.current) {
                    await faceMeshRef.current.send({ image: videoRef.current });
                }
            },
            width: 640,
            height: 480
        });

        cameraRef.current = camera;

        return () => {
            faceMesh.close();
            if (cameraRef.current) {
                cameraRef.current.stop();
            }
        };
    }, []);

    const onResults = (results: Results) => {
        if (!canvasRef.current) return;

        const ctx = canvasRef.current.getContext('2d');
        if (!ctx) return;

        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

        if (results.multiFaceLandmarks) {
            for (const landmarks of results.multiFaceLandmarks) {
                const emotions = detectEmotions(landmarks);
                window.dispatchEvent(new CustomEvent('faceDetection', {
                    detail: {
                        expressions: emotions
                    }
                }));

                const total = Object.values(emotions).reduce((a, b) => a + b, 0);
                const normalizedEmotions = Object.fromEntries(
                    Object.entries(emotions).map(([key, value]) => [key, value / total])
                );

                window.dispatchEvent(new CustomEvent('faceDetection', {
                    detail: {
                        expressions: normalizedEmotions
                    }
                }));

                ctx.fillStyle = '#00FF00';
                for (const landmark of landmarks) {
                    const x = landmark.x * canvasRef.current.width;
                    const y = landmark.y * canvasRef.current.height;
                    ctx.beginPath();
                    ctx.arc(x, y, 1, 0, 2 * Math.PI);
                    ctx.fill();
                }

                const dominantEmotion = Object.entries(normalizedEmotions)
                    .reduce((a, b) => a[1] > b[1] ? a : b);

                ctx.fillStyle = 'white';
                ctx.font = '16px Arial';
                ctx.fillText(
                    `${dominantEmotion[0]}: ${(dominantEmotion[1] * 100).toFixed(1)}%`,
                    10,
                    30
                );
            }
        }
    };

    const calculateMouthOpenness = (landmarks) => {
        const upperLipTop = landmarks[13];
        const upperLipBottom = landmarks[14];
        const lowerLipTop = landmarks[15];
        const lowerLipBottom = landmarks[17];

        const mouthGap = Math.abs(lowerLipTop.y - upperLipBottom.y);
        const mouthHeight = Math.abs(upperLipTop.y - lowerLipBottom.y);

        const mouthLeft = landmarks[61];
        const mouthRight = landmarks[291];
        const mouthWidth = Math.abs(mouthRight.x - mouthLeft.x);

        const openness = (mouthGap / mouthHeight + mouthWidth) / 2;
        return Math.min(openness, 1);
    };

    const calculateEyeOpenness = (landmarks) => {
        const leftUpperEye = landmarks[159];
        const leftLowerEye = landmarks[145];
        const rightUpperEye = landmarks[386];
        const rightLowerEye = landmarks[374];

        const leftEyeOpening = Math.abs(leftUpperEye.y - leftLowerEye.y);
        const rightEyeOpening = Math.abs(rightUpperEye.y - rightLowerEye.y);

        return (leftEyeOpening + rightEyeOpening) / 2;
    };

    const calculateBrowRaise = (landmarks) => {
        const leftBrow = landmarks[107];
        const rightBrow = landmarks[336];
        const leftEye = landmarks[159];
        const rightEye = landmarks[386];

        const nose = landmarks[1];
        const chin = landmarks[152];
        const faceHeight = Math.abs(nose.y - chin.y);

        const leftBrowRaise = Math.abs(leftBrow.y - leftEye.y);
        const rightBrowRaise = Math.abs(rightBrow.y - rightEye.y);

        return ((leftBrowRaise + rightBrowRaise) / 2) / faceHeight;
    };

    const startCamera = async () => {
        if (!cameraRef.current) return;

        setIsLoading(true);
        setError(null);

        try {
            await cameraRef.current.start();
            setIsStarted(true);
            console.log('Camera started successfully');
            await navigator.mediaDevices.getUserMedia({ video: true });
            window.dispatchEvent(new CustomEvent('cameraStatus', {
                detail: { isRunning: true }
            }));
        } catch (err) {
            console.error('Error starting camera:', err);
            setError('Failed to start camera');
            window.dispatchEvent(new CustomEvent('cameraStatus', {
                detail: { isRunning: false }
            }));
        } finally {
            setIsLoading(false);
        }
    };

    const stopCamera = () => {
        if (cameraRef.current) {
            cameraRef.current.stop();
            setIsStarted(false);
            window.dispatchEvent(new CustomEvent('cameraStatus', {
                detail: { isRunning: false }
            }));
        }
    };

    const detectEmotions = (landmarks) => {
        const mouthOpenness = calculateMouthOpenness(landmarks);
        const eyeOpenness = calculateEyeOpenness(landmarks);
        const browRaise = calculateBrowRaise(landmarks);

        const mouthCornerLeft = landmarks[61];
        const mouthCornerRight = landmarks[291];
        const mouthTop = landmarks[13];
        const mouthBottom = landmarks[14];
        const noseTop = landmarks[168];
        const noseBottom = landmarks[2];

        const mouthCurve = ((mouthCornerLeft.y + mouthCornerRight.y) / 2) - mouthTop.y;

        const mouthWidth = Math.abs(mouthCornerRight.x - mouthCornerLeft.x);
        const mouthHeight = Math.abs(mouthTop.y - mouthBottom.y);
        const mouthRatio = mouthWidth / mouthHeight;

        const noseWrinkle = Math.abs(noseTop.y - noseBottom.y);

        const mouthAsymmetry = Math.abs(mouthCornerLeft.y - mouthCornerRight.y);

        const baseEmotionValue = 0.05;

        console.log('Measurements:', {
            mouthOpenness,
            eyeOpenness,
            browRaise,
            mouthCurve,
            mouthRatio,
            noseWrinkle,
            mouthAsymmetry
        });

        const emotions = {
            // Happy: requires genuine smile configuration
            happy: (mouthCurve > 0.015 && mouthRatio > 1.4) ?
                Math.min(mouthCurve * 4 + mouthRatio * 0.2, 0.9) : baseEmotionValue,

            // Sad: requires clear downturn and appropriate brow position
            sad: (mouthCurve < -0.008 && browRaise < 0.07) ?
                Math.min(Math.abs(mouthCurve) * 3 + (0.08 - browRaise) * 2, 0.85) : baseEmotionValue,

            // Angry: requires clear brow lowering and tight mouth
            angry: (browRaise < 0.05 && mouthRatio < 1.3) ?
                Math.min((0.06 - browRaise) * 4 + mouthAsymmetry * 3, 0.85) : baseEmotionValue,

            // Surprised: requires ALL key features to be present
            surprised: (mouthOpenness > 0.15 && browRaise > 0.13 && eyeOpenness > 0.13) ?
                Math.min(mouthOpenness * 2 + browRaise * 1.5 + eyeOpenness * 1.5, 0.9) : baseEmotionValue,

            // Fearful: combination of wide eyes and slight mouth tension
            fearful: (eyeOpenness > 0.11 && browRaise > 0.09 && mouthRatio < 1.5) ?
                Math.min(eyeOpenness * 2.5 + browRaise * 1.5, 0.8) : baseEmotionValue,

            // Disgusted: nose wrinkle with mouth configuration
            disgusted: (noseWrinkle < 0.13 && mouthRatio < 1.4 && browRaise > 0.07) ?
                Math.min((0.14 - noseWrinkle) * 2.5 + (1.5 - mouthRatio), 0.8) : baseEmotionValue,

            // Neutral: moderate ranges for all features
            neutral: (
                browRaise >= 0.06 && browRaise <= 0.09 &&
                eyeOpenness >= 0.09 && eyeOpenness <= 0.12 &&
                mouthRatio >= 1.3 && mouthRatio <= 1.6 &&
                Math.abs(mouthCurve) < 0.012 &&
                mouthAsymmetry < 0.01
            ) ? 0.7 : baseEmotionValue
        };

        const maxEmotion = Object.entries(emotions).reduce((a, b) => a[1] > b[1] ? a : b);

        if (maxEmotion[0] !== 'neutral') {
            if (maxEmotion[1] > 0.5) {
                emotions.neutral = baseEmotionValue;
            }
            emotions[maxEmotion[0]] = Math.min(maxEmotion[1] * 1.2, 0.95);
        }

        const smoothingFactor = 0.3;
        const prevEmotions = window.prevEmotions || emotions;

        const smoothedEmotions = Object.fromEntries(
            Object.entries(emotions).map(([key, value]) => [
                key,
                value * smoothingFactor + (prevEmotions[key] || value) * (1 - smoothingFactor)
            ])
        );

        window.prevEmotions = smoothedEmotions;

        const total = Object.values(smoothedEmotions).reduce((a, b) => a + b, 0);
        const normalizedEmotions = Object.fromEntries(
            Object.entries(smoothedEmotions).map(([key, value]) => [
                key,
                value / total
            ])
        );

        return normalizedEmotions;
    };

    const cycleEmotion = () => {
        console.log("emotions", emotions)
        setCurrentIndex((prevIndex) => (prevIndex + 1) % emotions.length);
        window.dispatchEvent(new CustomEvent('faceDetection', {
            detail: {
                expressions: {
                    [emotions[currentIndex]]: 1.0
                }
            }
        }));
    };

    return (
        <Card className="w-full max-w-xl mx-auto h-96">
            <CardHeader className="h-20">
                <CardTitle>Face Expression Detector</CardTitle>
            </CardHeader>
            <CardContent className="h-[calc(100%-5rem)] flex flex-col">
                <div className="flex justify-center space-x-4 mb-4">
                    <Button
                        onClick={startCamera}
                        disabled={isLoading || isStarted}
                        className="w-32"
                    >
                        {isLoading ? 'Starting...' : 'Start Camera'}
                    </Button>
                    <Button
                        onClick={stopCamera}
                        disabled={!isStarted}
                        variant="destructive"
                        className="w-32"
                    >
                        Stop Camera
                    </Button>
                    <Button
                        onClick={cycleEmotion}
                        variant="secondary"
                        className="w-32"
                    >
                        Manual Emotion
                    </Button>
                </div>

                <div className="relative flex-1 bg-gray-100 rounded-lg overflow-hidden">
                    <video
                        ref={videoRef}
                        className="absolute inset-0 w-full h-full object-cover"
                        style={{ transform: 'scaleX(-1)' }}
                    />
                    <canvas
                        ref={canvasRef}
                        className="absolute inset-0 w-full h-full"
                        width={640}
                        height={480}
                        style={{ transform: 'scaleX(-1)' }}
                    />

                    {isLoading && (
                        <div className="absolute inset-0 flex items-center justify-center bg-black/50 text-white">
                            <span className="text-lg">Loading models...</span>
                        </div>
                    )}

                    {error && (
                        <div className="absolute inset-0 flex items-center justify-center bg-red-500/50 text-white">
                            <span className="text-lg bg-red-500 p-2 rounded">{error}</span>
                        </div>
                    )}

                    {!isStarted && !isLoading && !error && (
                        <div className="absolute inset-0 flex items-center justify-center bg-black/50 text-white">
                            <span className="text-lg">Click 'Start Camera' to begin</span>
                        </div>
                    )}
                </div>
            </CardContent>
        </Card>
    );
};

export default FaceDetector;