### Erweitert und vereinfacht
### https://github.com/magenta/magenta/blob/main/magenta/models/music_vae
### durch folgende Funktionen:

class MusicVAETrainer:
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

    def plot_combined_emotion_drum_patterns(emotion_sequences, output_file='combined_emotion_plot.png'):
    	"""Plot all emotion drum patterns in a single figure"""
    	fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    	for emotion, sequence in emotion_sequences.items():
        	metrics = analyze_drum_pattern(sequence)
        	times = sorted(metrics['pitch_counts'].keys())
        	simultaneous_hits = [len(metrics['pitch_counts'][time]) for time in times]
        	axes[0].plot(times, simultaneous_hits, label=emotion, linewidth=2)
    
    	emotions = list(emotion_sequences.keys())
    	complexity_values = []
    	density_values = [] 
    	for emotion, sequence in emotion_sequences.items():
        	metrics = analyze_drum_pattern(sequence)
        	complexity_values.append(metrics['pattern_complexity'])
        	density_values.append(np.mean(metrics['hit_density']))
    
    	return fig, axes

	def plot_emotion_drum_pattern(sequence, emotion, output_dir='output'):
    	"""Creates visualization showing metrics for a specific emotion's drum pattern"""
    	metrics = analyze_drum_pattern(sequence)
    	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    	times = np.arange(len(metrics['time_series'])) / 4 
    	ax1.plot(times, metrics['time_series'], label=emotion, linewidth=2)

    	labels = ['Pattern Complexity', 'Average Hit Density'] 
    	values = [metrics['pattern_complexity'], np.mean(metrics['hit_density'])]
    	ax2.bar(labels, values)

	def analyze_drum_pattern(sequence):
    	"""Analyzes drum sequence and extracts metrics like"""
    	metrics = {
        	'total_hits': len(sequence.notes),
        	'time_series': [],
        	'pitch_counts': {},
        	'avg_velocity': 0,
        	'hit_density': [], 
        	'pattern_complexity': 0
    	}
    
    	velocity_sum = 0
    	for note in sequence.notes:
        	time = round(note.start_time, 2)
        	metrics['time_series'].append(time)
        	velocity_sum += note.velocity
        
        	if time not in metrics['pitch_counts']:
            	metrics['pitch_counts'][time] = []
        	metrics['pitch_counts'][time].append(note.pitch)

    	metrics['avg_velocity'] = velocity_sum / metrics['total_hits'] 
    
    	hit_density = []
    	for time, pitches in metrics['pitch_counts'].items():
        	hit_density.append(len(pitches))
        	metrics['pattern_complexity'] += len(pitches)

    	metrics['hit_density'] = hit_density
    	return metrics

	def save_sequence_to_midi(sequence, filename, output_dir='output'):
    	"""Saves a NoteSequence to a MIDI file in the specified output directory"""
    	output_dir = Path(output_dir)
    	output_dir.mkdir(exist_ok=True)
    	file_path = output_dir / f"{filename}.midi"
    	note_seq.midi_io.note_sequence_to_midi_file(sequence, str(file_path))

	def create_variable_mapping():
    	"""Creates mapping between checkpoint variables and model variables for loading pretrained models correctly"""
    	var_map = {
        	'bidirectional/forward_lstm/lstm_cell_1/kernel': 
            	'encoder/cell_0/bidirectional_rnn/fw/multi_rnn_cell/cell_0/lstm_cell/kernel',
        	'bidirectional/forward_lstm/lstm_cell_1/bias':
            	'encoder/cell_0/bidirectional_rnn/fw/multi_rnn_cell/cell_0/lstm_cell/bias',
        
        	'dense/kernel': 'encoder/mu/kernel',
        	'dense/bias': 'encoder/mu/bias',
        
        	'lstm_2/lstm_cell_6/kernel':
            	'decoder/multi_rnn_cell/cell_0/input_projection_wrapper/lstm_cell/kernel',
        	'lstm_2/lstm_cell_6/bias': 
            	'decoder/multi_rnn_cell/cell_0/input_projection_wrapper/lstm_cell/bias',
    	}
    	return var_map

class MusicVAE:
    """Handles generation of drum patterns with emotion conditioning"""
    def __init__(self, config, batch_size, checkpoint_dir_or_path):
        self._config = config
        self._batch_size = batch_size
        self.output_depth = 90
        self.z_size = 256

        with tf.compat.v1.Graph().as_default():
            self._session = tf.compat.v1.Session()
            
            self._z = tf.compat.v1.placeholder(
                tf.float32, 
                shape=[self._batch_size, self.z_size]
            )
            
            hidden = tf.keras.layers.Dense(512, activation='relu')(self._z)
            self._outputs = tf.keras.layers.Dense(self.output_depth * 32)(hidden)

    def decode(self, z, temperature=1.0, length=None):
        """
        Decodes latent vectors into drum sequences
        
        Args:
            z: Latent vectors to decode
            temperature: Controls randomness in generation
            length: Max sequence length
        """
        outputs = self._session.run(self._outputs, {self._z: z})
        outputs = outputs.reshape(self._batch_size, 32, -1)

        sequences = []
        for batch_idx in range(self._batch_size):
            sequence = note_seq.NoteSequence()
            sequence.tempos.add().qpm = 120.0

            time = 0.0
            for step in outputs[batch_idx]:
                if np.max(step) > 0.5:
                    note = sequence.notes.add()
                    note.pitch = 36
                    note.velocity = 80
                    note.start_time = time
                    note.end_time = time + 0.125
                    note.is_drum = True
                time += 0.125

            sequences.append(sequence)

        return sequences

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
            quarter_note = int(time * 4)  # Convert to quarter note
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