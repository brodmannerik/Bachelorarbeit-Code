### Erweitert und vereinfacht
### https://github.com/RichardYang40148/MidiNet/tree/master/v1
### durch folgende Funktionen:

class MidiNETTrainer:
	@tf.function
    def train_step(self, real_patterns):
        real_patterns = tf.cast(real_patterns, self.dtype)
        batch_size = tf.shape(real_patterns)[0]
        noise = tf.random.normal([batch_size, self.latent_dim], dtype=self.dtype)

        conditions = tf.map_fn(
            self.create_condition,
            real_patterns,
            fn_output_signature=self.dtype
        )

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_patterns = self.generator([noise, conditions], training=True)
            real_output = self.discriminator([real_patterns, conditions], training=True)
            fake_output = self.discriminator([generated_patterns, conditions], training=True)

            gen_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(
                    tf.ones_like(fake_output, dtype=self.dtype),
                    fake_output
                )
            )
            disc_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(
                    tf.ones_like(real_output, dtype=self.dtype),
                    real_output
                ) +
                tf.keras.losses.binary_crossentropy(
                    tf.zeros_like(fake_output, dtype=self.dtype),
                    fake_output
                )
            )

        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.g_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

        total_loss = gen_loss + disc_loss
        return total_loss, gen_loss, disc_loss

class MidiNET:
	def __init__(self, model_path: str = './saved_model/generator.h5'):
        """Emotion-specific parameters"""
        self.emotion_params = {
            'happy': {'temperature': 1.2, 'velocity': 100},  
            'sad': {'temperature': 0.8, 'velocity': 70},     
            'angry': {'temperature': 1.4, 'velocity': 120},  
            'disgust': {'temperature': 1.6, 'velocity': 90}, 
            'fear': {'temperature': 1.3, 'velocity': 85},    
            'surprise': {'temperature': 1.5, 'velocity': 95},
            'neutral': {'temperature': 1.0, 'velocity': 80}  
        }

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

    def generate_emotion_pattern(self, emotion: str) -> pretty_midi.PrettyMIDI:
        """Generates patterns based on emotions"""
        params = self.emotion_params[emotion]
        temperature = params['temperature']
        velocity = params['velocity']

        if emotion == 'sad':
            threshold = 0.3  
        elif emotion in ['angry', 'surprise']:
            threshold = 0.4 
        else:
            threshold = 0.35  

    def analyze_pattern(self, pattern: pretty_midi.PrettyMIDI) -> Dict:
        """Provides detailed metrics about generated patterns"""
        metrics = {
            'total_hits': len(pattern.instruments[0].notes),
            'pattern_complexity': 0,
            'avg_velocity': 0,
            'pitch_counts': defaultdict(int),
            'time_series': defaultdict(list)
        }
        
        if metrics['total_hits'] > 1:
            note_times = [note.start for note in pattern.instruments[0].notes]
            time_diffs = np.diff(sorted(note_times))
            metrics['pattern_complexity'] = int(np.std(time_diffs) * 1000)

    def compare_emotions(self, emotions: List[str] = None) -> Tuple[plt.Figure, Dict]:
        """Compares patterns across different emotions"""
        if emotions is None:
            emotions = ['happy', 'sad', 'angry', 'disgust', 'fear', 'surprise', 'neutral']

        metrics = {}
        patterns = {}

        for emotion in emotions:
            pattern = self.generate_emotion_pattern(emotion)
            metrics[emotion] = self.analyze_pattern(pattern)
            patterns[emotion] = pattern

    def rhythm_consistency_loss(self, patterns):
        """loss function"""
        shifted = tf.roll(patterns, shift=1, axis=1)
        correlation = tf.reduce_mean(tf.abs(patterns - shifted))
        return correlation

    def plot_training_loss(self, batch_losses, epoch_losses):
        """visualization"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Creates both batch-level and epoch-level visualizations
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        
        # Plot batch-level losses
        ax1.plot(batch_losses)
        ax1.set_title('Training Loss (Batch Level)')
        
        # Plot epoch-level losses
        ax2.plot(epoch_losses, 'r-', linewidth=2)
        ax2.set_title('Training Loss (Epoch Average)')
