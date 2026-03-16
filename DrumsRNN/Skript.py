### Erweitert und vereinfacht
### https://github.com/magenta/magenta/blob/main/magenta/models/drums_rnn
### durch folgende Funktionen:

import os
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("TensorFlow version:", tf.__version__)
import note_seq
from note_seq import midi_io
from note_seq import sequences_lib
from note_seq import drums_lib
from note_seq.protobuf import generator_pb2
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

class DrumsRNNTrainer:
	def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model_dir = os.path.join(output_dir, 'model')
        self.sequence_length = 32
        
        os.makedirs(self.model_dir, exist_ok=True)
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            tf.config.experimental.set_memory_growth(gpus[0], True)

    def prepare_dataset(self):
    	"""Prepare and analyze the MIDI dataset"""
        sequences = []
        
        for filename in os.listdir(self.input_dir):
            if filename.endswith('.mid'):
                midi_path = os.path.join(self.input_dir, filename)
                sequence = midi_io.midi_file_to_note_sequence(midi_path)
                
                drum_notes = [note for note in sequence.notes if note.is_drum]
                if drum_notes:
                    drums_sequence = note_seq.NoteSequence()
                    drums_sequence.notes.extend(drum_notes)
                    sequences.append(drums_sequence)
        
        return sequences, len(sequences)

    def _create_model(self):
    	"""Create and return the DrumsRNN model"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(128, return_sequences=True, 
                               input_shape=(self.sequence_length, 3)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(88, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def train_model(self, sequences, num_training_steps=1000):
    	"""Train the DrumsRNN model"""
        model = self._create_model()
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=3
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.model_dir, 'checkpoint'),
                save_best_only=True
            )
        ]
        
        X = []  
        y = []  
        
        for sequence in sequences:
            times = [note.start_time for note in sequence.notes]
            velocities = [note.velocity for note in sequence.notes]
            pitches = [note.pitch for note in sequence.notes]
            
            for i in range(len(pitches) - self.sequence_length):
                X.append([
                    pitches[i:i+self.sequence_length],
                    velocities[i:i+self.sequence_length],
                    times[i:i+self.sequence_length]
                ])
                y.append(pitches[i+1:i+self.sequence_length+1])
        
        X = np.array(X)
        y = np.array(y)
        
        history = model.fit(
            X, y,
            epochs=num_training_steps,
            batch_size=32,
            callbacks=callbacks,
            validation_split=0.2
        )
        
        model.save(os.path.join(self.model_dir, 'final_model'))
        
        return self.model_dir

    def _plot_training_loss(self, batch_losses, epoch_losses):
        """Plot and save training loss curves"""
        if not batch_losses or not epoch_losses:
            print("Warning: No loss data available for plotting")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

        ax1.plot(batch_losses)
        ax1.set_title('Training Loss (Batch Level)')
        ax1.set_xlabel('Batch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)

        ax2.plot(epoch_losses, 'r-', linewidth=2)
        ax2.set_title('Training Loss (Epoch Average)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Average Loss')
        ax2.grid(True)

        plt.tight_layout()

        filename = f'training_loss_{timestamp}.png'
        filepath = os.path.join(self.visualization_dir, filename)
        plt.savefig(filepath)
        print(f"\nLoss plot saved as: {filepath}")
        plt.close()

        csv_filename = f'training_loss_{timestamp}.csv'
        csv_path = os.path.join(self.visualization_dir, csv_filename)
        with open(csv_path, 'w') as f:
            f.write('batch_loss,epoch_loss\n')
            for i in range(len(batch_losses)):
                epoch_idx = i // 100 
                epoch_loss = epoch_losses[epoch_idx] if epoch_idx < len(epoch_losses) else ''
                f.write(f'{batch_losses[i]},{epoch_loss}\n')
        print(f"Loss values saved as: {csv_path}")

class EmotionMusicGenerator:
    def __init__(self, model_dir: str = './dataset/drums-rnn-output/model'):
        """Initialize the EmotionMusicGenerator with emotional mapping capabilities"""
        self.steps_per_quarter = 4  
        self.sequence_length = 32   
        self.feature_dim = 3

        self.drum_mapping = {
            36: 'Bass Drum',
            38: 'Snare Drum',
            42: 'Closed Hi-hat',
            46: 'Open Hi-hat',
            45: 'Low Tom',
            48: 'Mid Tom',
            50: 'High Tom',
            49: 'Crash Cymbal',
            51: 'Ride Cymbal'
        }

        self.emotion_music_map: Dict[str, Callable] = {
            'happy': lambda: self._generate_drums(
                primer_notes=[[36, 42], [38, 42], [36, 42], [38, 42]],  # Samba-style pattern
                temperature=1.2,
                qpm=130 
            ),
            'sad': lambda: self._generate_drums(
                primer_notes=[[36], [0], [38], [0]],  # Sparse baião rhythm
                temperature=0.8,
                qpm=80
            ),
            'angry': lambda: self._generate_drums(
                primer_notes=[[36, 42, 49], [36, 38], [36, 42, 49], [38, 42]],  # Intense maracatu pattern
                temperature=1.4,
                qpm=135
            ),
            'disgust': lambda: self._generate_drums(
                primer_notes=[[36, 49], [0], [38, 46], [36], [42, 49]],  # Dissonant, irregular pattern
                temperature=1.6, 
                qpm=90  
            ),
            'fear': lambda: self._generate_drums(
                primer_notes=[[36, 42], [0], [38, 46], [0]],  # Syncopated frevo rhythm
                temperature=1.3,
                qpm=100
            ),
            'surprise': lambda: self._generate_drums(
                primer_notes=[[36, 49], [42, 38], [36, 46], [42, 38]],  # Complex xote pattern
                temperature=1.5,
                qpm=115
            ),
            'neutral': lambda: self._generate_drums(
                primer_notes=[[36, 42], [0], [38, 42], [0]],  # Basic bossa nova pattern
                temperature=1.0,
                qpm=110
            )
        }

    def _create_midi_sequence(self, predicted_drums: np.ndarray, tempo: int) -> str:
        """
        Converts drum patterns directly to MIDI using midiutil
        
        Args:
            predicted_drums: Generated drum patterns
            tempo: Tempo in beats per minute
            
        Returns:
            Path to generated MIDI file
        """
        output_file = "output/generated_drums.mid"
        midi = midiutil.MIDIFile(1)
        track = 0
        channel = 9  
        time = 0
        midi.addTempo(track, time, tempo)

        for step, step_drums in enumerate(predicted_drums):
            for drum_idx, probability in enumerate(step_drums):
                if probability > 0.5:  
                    pitch = 35 + drum_idx
                    if pitch in self.drum_mapping:
                        velocity = min(127, max(1, int(probability * 127)))
                        midi.addNote(track, channel, pitch, time + step / 4, 0.25, velocity)

        os.makedirs("output", exist_ok=True)
        with open(output_file, "wb") as f:
            midi.writeFile(f)

        return output_file

    def _generate_drums(self, primer_notes: list, temperature: float = 1.0,
                     qpm: int = 120) -> np.ndarray:
        """
        Simplified drum generation
        
        Args:
            primer_notes: Initial drum pattern
            temperature: Controls randomness
            qpm: Tempo in quarters per minute
            
        Returns:
            Generated drum pattern as numpy array
        """
        try:
            actual_temperature = temperature * random.uniform(0.9, 1.1)

            lstm_input = self._sequence_to_input_tensors(primer_notes)

            sequences = []
            for _ in range(3): 
                outputs = self.inference(lstm_input=lstm_input)
                sequence = outputs['time_distributed'].numpy()[0]
                sequences.append(sequence)

            best_sequence = max(sequences,
                              key=lambda s: np.sum(s > 0.5) if np.sum(s > 0.5) < 100 else 0)

            return best_sequence

        except Exception as e:
            print(f"Generation error: {e}")
            raise

    def generate_and_visualize_all(self):
        """Generates visualizations for all emotional patterns"""
        sequences = {}
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)

        for emotion in self.emotion_music_map.keys():
            sequence = self.emotion_music_map[emotion]()
            sequences[emotion] = sequence
            midi_path = output_dir / f"{emotion}_music.mid"
            self._create_midi_sequence(sequence, 120)

        self._create_rhythm_heatmap(sequences)
        self._create_velocity_plot(sequences)
        self._create_note_distribution_plot(sequences)
        self._create_complexity_plot(sequences)

    def detect_emotion(self, manual_emotion: Optional[str] = None) -> Optional[str]:
        """Handles emotion selection"""
        if manual_emotion:
            manual_emotion = manual_emotion.lower()
            if manual_emotion not in self.emotion_music_map:
                raise ValueError(f"Unsupported emotion: {manual_emotion}")
            return manual_emotion
        return 'neutral'
