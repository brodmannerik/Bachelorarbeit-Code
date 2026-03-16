### Erweitert und vereinfacht
### https://github.com/hugofloresgarcia/vampnet
### durch folgende Funktionen:

class VAMPnetConfig:
    """Configuration dataclass for VAMPnet parameters"""
    input_shape: Tuple[int, ...]
    latent_dim: int = 16
    hidden_dims: List[int] = None
    learning_rate: float = 1e-4
    beta: float = 0.01
    gamma: float = 1.0
    n_components: int = 32

class EncoderNetwork(Model):
    """Neural network for encoding inputs into latent space"""
    def __init__(self, latent_dim, hidden_dims):
        # Conv layers for downsampling
        self.conv1 = layers.Conv2D(32, 3, strides=2, padding='same')
        self.conv2 = layers.Conv2D(64, 3, strides=2, padding='same')
        # Dense layers for latent projection        
        self.z_mean = layers.Dense(latent_dim)
        self.z_log_var = layers.Dense(latent_dim)

class DecoderNetwork(Model):
    """Neural network for decoding latent vectors back to input space"""
    def __init__(self, input_shape, hidden_dims):
        self.dense_layers = []
        self.conv1 = layers.Conv2DTranspose(64, 3, strides=2, padding='same')
        self.conv2 = layers.Conv2DTranspose(32, 3, strides=2, padding='same')

class VAMPnetLoss(tf.keras.losses.Loss):
    """Custom loss function combining reconstruction and VAMP terms"""
    def __init__(self, beta: float = 0.01, gamma: float = 1.0):
        self.beta = beta
        self.gamma = gamma

class VAMPnet(tf.keras.Model): 
    """Main VAMPnet model combining encoder, decoder and loss"""
    def __init__(self, config):
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        
        # Methods for training and inference
         @tf.function
    def train_step(self, x):
        """Fixed training step with correct tensor dimensions"""
        with tf.GradientTape() as tape:
            # Forward passes
            z_mean, z_log_var, z = self.encoder(x)
            x_decoded = self.decoder(z)

            # Fixed reconstruction loss calculation
            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(
                    x,
                    x_decoded,
                    from_logits=False
                )
            ) * 0.01

            # Rest remains the same
            kl_loss = tf.clip_by_value(
                -0.5 * tf.reduce_mean(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
                ),
                0.0, 10.0
            )

            vamp_loss = self._compute_vamp_loss(z) * 0.1

            total_loss = reconstruction_loss + self.config.beta * kl_loss - self.config.gamma * vamp_loss

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(total_loss, variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return total_loss, reconstruction_loss, kl_loss, vamp_loss
            
        def encode(self, x):
        """Forward pass through encoder"""
            mean, log_var, z = self.encoder(x)
            return mean, log_var
            
        def decode(self, z):
        """Forward pass through decoder"""
            return self.decoder(z)

class AudioVAMPnet(VAMPnet):
    """VAMPnet extension specifically for audio processing"""
    def __init__(self, config):
        self.audio_processor = self.build_audio_processor()
        
    def build_audio_processor(self):
        return Model(inputs, x, name='audio_processor')
	