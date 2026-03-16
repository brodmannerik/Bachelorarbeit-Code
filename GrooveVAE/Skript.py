### Erweitert und vereinfacht
### https://github.com/magenta/magenta/blob/main/magenta/models/music_vae
### durch folgende Funktionen:

class GrooveVAETrainer:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model_dir = os.path.join(output_dir, 'model')
        self.visualization_dir = os.path.join(output_dir, 'visualizations')

        self.drum_mapping = {
            36: 0,  # Bass drum
            38: 1,  # Snare drum
            42: 2,  # Closed hi-hat
            46: 3,  # Open hi-hat
            45: 4,  # Low tom
            48: 5,  # Mid tom
            50: 6,  # High tom
            49: 7,  # Crash cymbal
            51: 8   # Ride cymbal
        }

    def _extract_groove_features(self, sequence):
        """Custom feature extraction specifically for drum patterns"""
        features = np.zeros((self.sequence_length, 9, 3))

        try:
            quantized = sequences_lib.quantize_note_sequence(
                sequence, steps_per_quarter=4
            )

            for note in quantized.notes:
                if note.is_drum and note.pitch in self.drum_mapping:
                    drum_idx = self.drum_mapping[note.pitch]
                    step = int(note.quantized_start_step % self.sequence_length)

                    if 0 <= step < self.sequence_length:
                        features[step, drum_idx, 0] = 1.0  
                        features[step, drum_idx, 1] = note.velocity / 127.0 
                        offset = (note.start_time - (note.quantized_start_step * 0.25)) / 0.25
                        features[step, drum_idx, 2] = np.clip(offset, -0.5, 0.5)

            return features
        except Exception as e:
            print(f"Error in _extract_groove_features: {str(e)}")
            return None

    def _build_encoder(self):
        """Custom encoder architecture"""
        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.sequence_length, 9, 3)),
            tf.keras.layers.Reshape((self.sequence_length, 27)), 
            
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(512, return_sequences=True)
            ),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(256, return_sequences=False)
            ),
            
            tf.keras.layers.Dense(self.z_size * 2)
        ])

    def groove_loss(y_true, y_pred):
        """Custom loss function specific to groove features"""
        hit_true, vel_true, off_true = tf.split(y_true, 3, axis=-1)
        hit_pred, vel_pred, off_pred = tf.split(y_pred, 3, axis=-1)

        hit_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(hit_true, hit_pred)
        )
        
        vel_loss = tf.reduce_mean(
            hit_true * tf.square(vel_true - vel_pred)
        )
        
        off_loss = tf.reduce_mean(
            hit_true * tf.square(off_true - off_pred)
        )

        return hit_loss + vel_loss + off_loss

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

    class GrooveVAE(tf.keras.Model):
        def __init__(self, batch_size=1):
        self._batch_size = batch_size
        self.sequence_length = 32
        self._session = tf.compat.v1.Session()

        with self._session.as_default():
            self._build_model()
            self._session.run(tf.compat.v1.global_variables_initializer())

        self.generated_sequences = {}  

    def _build_model(self):
        with tf.compat.v1.variable_scope('model'):
            self.z_input = tf.compat.v1.placeholder(
                tf.float32,
                shape=[self._batch_size, 256],
                name='z_input'
            )

            hidden = tf.compat.v1.layers.dense(
                self.z_input,
                1024,
                activation=tf.nn.tanh,
                name='hidden'
            )

            hidden = tf.compat.v1.layers.dense(
                hidden,
                512,
                activation=tf.nn.relu,
                name='hidden2'
            )

            hidden_seq = tf.tile(tf.expand_dims(hidden, 1), [1, self.sequence_length, 1])

            self.outputs = tf.compat.v1.layers.dense(
                hidden_seq,
                27,
                name='output'
            )

            self.probabilities = tf.nn.softmax(self.outputs)

    def analyze_drum_pattern(self, sequence: music_pb2.NoteSequence) -> Dict:
        """Analyze a drum sequence and return various metrics."""
        metrics = {
            'total_hits': len(sequence.notes),
            'time_series': [],
            'pitch_counts': {},
            'avg_velocity': 0,
            'hit_density': [],
            'pattern_complexity': 0
        }

        time_map = {}
        for note in sequence.notes:
            time = round(note.start_time, 3)  
            if time not in time_map:
                time_map[time] = 0
            time_map[time] += 1

            metrics['pitch_counts'][note.pitch] = metrics['pitch_counts'].get(note.pitch, 0) + 1
            metrics['avg_velocity'] += note.velocity

        if time_map:
            times = sorted(time_map.keys())
            metrics['time_points'] = times
            metrics['time_series'] = [time_map[t] for t in times]
        else:
            metrics['time_points'] = []
            metrics['time_series'] = []

        metrics['avg_velocity'] = metrics['avg_velocity'] / len(sequence.notes) if sequence.notes else 0

        quarter_notes = {}
        for time, count in time_map.items():
            quarter_note = int(time * 4)
            if quarter_note not in quarter_notes:
                quarter_notes[quarter_note] = 0
            quarter_notes[quarter_note] += count

        metrics['hit_density'] = list(quarter_notes.values())

        prev_pitch = None
        transitions = 0
        for note in sorted(sequence.notes, key=lambda x: x.start_time):
            if prev_pitch is not None and note.pitch != prev_pitch:
                transitions += 1
            prev_pitch = note.pitch
        metrics['pattern_complexity'] = transitions

        return metrics

    def plot_drum_analysis(self, sequence: music_pb2.NoteSequence, emotion: str, ax=None, skip_title=False):
        """Create a comprehensive visualization of the drum pattern."""
        metrics = self.analyze_drum_pattern(sequence)
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        if metrics['time_points'] and metrics['time_series']:
            ax.plot(metrics['time_points'], metrics['time_series'], label=emotion, alpha=0.7)

        ax.set_xlabel('Time (quarter notes)')
        ax.set_ylabel('Number of simultaneous hits')
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax.set_ylim(0, max(metrics['time_series']) + 1 if metrics['time_series'] else 4)

        return ax, metrics

    def compare_emotions(self, emotions):
        """Create comparison plots for different emotions."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10),
                                       gridspec_kw={'height_ratios': [2, 1]})

        all_metrics = {}

        total_hits = 0
        total_complexity = 0
        total_velocity = 0
        total_sequences = 0

        for emotion in emotions:
            if emotion in self.generated_sequences:
                _, metrics = self.plot_drum_analysis(
                    self.generated_sequences[emotion],
                    emotion,
                    ax=ax1,
                    skip_title=True  
                )
                all_metrics[emotion] = metrics
                total_hits += metrics['total_hits']
                total_complexity += metrics['pattern_complexity']
                total_velocity += metrics['avg_velocity'] * metrics['total_hits']  
                total_sequences += 1

        avg_velocity = total_velocity / total_hits if total_hits > 0 else 0

        title = f'Drum Pattern Analysis\nTotal hits: {total_hits}, Complexity: {total_complexity}, Avg Velocity: {avg_velocity:.1f}'
        ax1.set_title(title, pad=20)

        ax1.set_xlim(-0.5, self.sequence_length / 4 + 0.5)

        emotions_list = list(all_metrics.keys())
        complexity = [metrics['pattern_complexity'] for metrics in all_metrics.values()]
        density = [np.mean(metrics['hit_density']) if metrics['hit_density'] else 0
                   for metrics in all_metrics.values()]

        x = np.arange(len(emotions_list))
        width = 0.35

        ax2.bar(x, complexity, width, label='Pattern Complexity')
        ax2.bar(x + width, density, width, label='Average Hit Density')

        ax2.set_ylabel('Value')
        ax2.set_title('Pattern Characteristics by Emotion')
        ax2.set_xticks(x + width / 2)
        ax2.set_xticklabels(emotions_list)
        ax2.legend()

        plt.tight_layout()
        return fig, all_metrics

    def decode(self, z, temperature=1.0, emotion_pattern=None):
        """Decode latent vectors into drum sequences."""
        probs = self._session.run(
            self.probabilities,
            feed_dict={self.z_input: z}
        )

        sequences = []
        for batch_idx in range(self._batch_size):
            sequence = music_pb2.NoteSequence()
            sequence.tempos.add().qpm = 120.0

            time = 0.0
            for step_idx, step in enumerate(probs[batch_idx]):
                if temperature != 1.0:
                    step = np.power(step, 1.0 / temperature)
                    step /= np.sum(step)

                if emotion_pattern and step_idx % len(emotion_pattern) == 0:
                    pattern_step = emotion_pattern[step_idx // len(emotion_pattern) % len(emotion_pattern)]
                    if pattern_step != [0]: 
                        for pitch in pattern_step:
                            note = sequence.notes.add()
                            note.pitch = pitch
                            note.velocity = 80 + np.random.randint(-10, 10) 
                            note.start_time = time
                            note.end_time = time + 0.125
                            note.is_drum = True

                threshold = 0.3 / temperature  
                for i, prob in enumerate(step):
                    if prob > threshold and np.random.random() < temperature:  
                        note = sequence.notes.add()
                        note.pitch = 36 + i
                        note.velocity = int(min(127, prob * 100))
                        note.start_time = time + np.random.uniform(0, 0.02) 
                        note.end_time = note.start_time + 0.125
                        note.is_drum = True

                time += 0.125

            sequences.append(sequence)

        return sequences


def generate_emotion_patterns():
    """Create output directory"""
    os.makedirs('output', exist_ok=True)

    model = SimpleGrooveVAE(batch_size=1)

    emotions = {
        'happy': {
            'temperature': 1.2,
            'qpm': 130,
            'pattern': [[36, 42], [38, 42], [36, 42], [38, 42]]  # Samba-style pattern
        },
        'sad': {
            'temperature': 0.8,
            'qpm': 80,
            'pattern': [[36], [0], [38], [0]]  # Sparse baião rhythm
        },
        'angry': {
            'temperature': 1.4,
            'qpm': 135,
            'pattern': [[36, 42, 49], [36, 38], [36, 42, 49], [38, 42]]  # Intense maracatu
        },
        'disgust': {
            'temperature': 1.6,
            'qpm': 90,
            'pattern': [[36, 49], [0], [38, 46], [36], [42, 49]]  # Dissonant pattern
        },
        'fear': {
            'temperature': 1.3,
            'qpm': 100,
            'pattern': [[36, 42], [0], [38, 46], [0]]  # Syncopated frevo
        },
        'surprise': {
            'temperature': 1.5,
            'qpm': 115,
            'pattern': [[36, 49], [42, 38], [36, 46], [42, 38]]  # Complex xote
        },
        'neutral': {
            'temperature': 1.0,
            'qpm': 110,
            'pattern': [[36, 42], [0], [38, 42], [0]]  # Basic bossa nova
        }
    }

    for emotion, config in emotions.items():
        try:
            print(f"\nGenerating {emotion} pattern...")

            z = np.random.normal(0, 1.5, [1, 256]).astype(np.float32)  

            sequences = model.decode(
                z,
                temperature=config['temperature'],
                emotion_pattern=config['pattern']
            )

            if sequences and len(sequences) > 0:
                sequence = sequences[0]
                sequence.tempos[0].qpm = config['qpm']

                model.generated_sequences[emotion] = sequence

                midi_filename = f"output/groove_{emotion}.mid"
                note_seq.sequence_proto_to_midi_file(sequence, midi_filename)
                print(f"Generated and saved {emotion} pattern to {midi_filename}")

        except Exception as e:
            print(f"Error generating {emotion} pattern: {e}")
            continue

    try:
        fig, metrics = model.compare_emotions(emotions.keys())
        fig.savefig('output/emotion_analysis.png', bbox_inches='tight', dpi=300)
        plt.close(fig)

        print("\nDetailed Analysis:")
        for emotion, metric in metrics.items():
            print(f"\n{emotion.upper()}:")
            print(f"Total hits: {metric['total_hits']}")
            print(f"Pattern complexity: {metric['pattern_complexity']}")
            print(f"Average velocity: {metric['avg_velocity']:.1f}")

    except Exception as e:
        print(f"Error generating visualization: {e}")