import React from 'react';
import FaceDetector from './components/FaceDetector.tsx';
import DrumGenerator from './components/DrumGenerator.tsx';
import DrumVisualizer from "@/components/DrumVisualizer.tsx";
import './App.css'

type StatusIndicatorProps = {
    isActive: boolean;
    label: string;
};

const StatusIndicator: React.FC<StatusIndicatorProps> = ({ isActive, label }) => (
    <div className="flex items-center gap-2">
        <div className={`w-2 h-2 rounded-full ${isActive ? 'bg-green-500' : 'bg-gray-400'}`}></div>
        <span className="text-sm text-gray-600">
            {label} {isActive ? 'running' : 'ready'}
        </span>
    </div>
);

const StatusIndicators = () => {
    const [isCameraActive, setIsCameraActive] = React.useState(false);
    const [isModelActive, setIsModelActive] = React.useState(false);

    React.useEffect(() => {
        const handleCameraStatus = (event) => {
            setIsCameraActive(event.detail.isRunning);
        };

        const handleModelStatus = (event) => {
            setIsModelActive(event.detail.isRunning);
        };

        window.addEventListener('cameraStatus', handleCameraStatus);
        window.addEventListener('modelStatus', handleModelStatus);

        setIsCameraActive(false);

        setIsModelActive(true);

        return () => {
            window.removeEventListener('cameraStatus', handleCameraStatus);
            window.removeEventListener('modelStatus', handleModelStatus);
        };
    }, []);

    return (
        <div className="flex gap-6">
            <StatusIndicator
                isActive={isModelActive}
                label="model"
            />
            <StatusIndicator
                isActive={isCameraActive}
                label="camera"
            />
        </div>
    );
};

const App: React.FC = () => {
    return (
        <div>
            <div className="w-full bg-gray-50 text-white py-6 px-4 mb-8">
                <div className="container mx-auto">
                    <h1 className="text-3xl text-black font-bold mb-2">Emotion-Based Drum Generator</h1>
                    <p className="text-gray-600">A Drum Generator driven by facial expressions</p>
                    <div className="mt-4">
                        <StatusIndicators />
                    </div>
                </div>
            </div>

            <div className="container mx-auto max-w-max p-4 space-y-6">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 max-w-screen-2xl">
                    <DrumGenerator />
                    <FaceDetector />
                </div>
                <div className="mx-auto">
                    <DrumVisualizer />
                </div>
            </div>
        </div>
    );
};

export default App;