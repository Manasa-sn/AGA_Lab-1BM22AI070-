import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

# Load Fashion MNIST and normalize
(x_train, _), (x_test, _) = fashion_mnist.load_data()
x_train, x_test = x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0
x_train = np.expand_dims(x_train, -1)  # Add channel dimension
x_test = np.expand_dims(x_test, -1)    # Add channel dimension

# VAE Model
class VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2), padding='same'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2), padding='same'),
            layers.Flatten(),
            layers.Dense(latent_dim + latent_dim)
        ])
        self.decoder = models.Sequential([
            layers.Dense(7 * 7 * 64, activation='relu', input_dim=latent_dim),
            layers.Reshape((7, 7, 64)),
            layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same'),
            layers.UpSampling2D((2, 2)),
            layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same'),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
        ])

    def call(self, x):
        z = self.encoder(x)
        mu, log_var = z[:, :self.latent_dim], z[:, self.latent_dim:]
        z = mu + tf.exp(0.5 * log_var) * tf.random.normal(tf.shape(mu))
        return self.decoder(z), mu, log_var

# Calculate loss as a percentage of max possible loss (for binary cross-entropy)
def compute_loss_percentage(loss, image_shape):
    max_possible_loss_per_pixel = np.log(2)  # log(2) is the maximum binary cross-entropy loss per pixel
    total_pixels = np.prod(image_shape)  # Total number of pixels per image (28x28)
    max_possible_loss = max_possible_loss_per_pixel * total_pixels
    percentage_loss = (loss / max_possible_loss) * 100
    return percentage_loss

# VAE Loss Function
def vae_loss(x, x_decoded_mean, mu, log_var):
    reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(x, x_decoded_mean), axis=(1, 2)))
    kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=1))
    return reconstruction_loss + kl_loss

# Custom training loop for VAE (as it's easier to handle multiple outputs)
vae = VAE(latent_dim=64)

@tf.function
def compute_loss(model, x):
    # Forward pass
    decoded, mu, log_var = model(x)
    # Compute the loss
    loss = vae_loss(x, decoded, mu, log_var)
    return loss

# Train VAE
optimizer = tf.keras.optimizers.Adam()
for epoch in range(50):
    for batch in range(0, len(x_train), 256):
        x_batch = x_train[batch:batch+256]

        with tf.GradientTape() as tape:
            loss = compute_loss(vae, x_batch)

        grads = tape.gradient(loss, vae.trainable_variables)
        optimizer.apply_gradients(zip(grads, vae.trainable_variables))
    percentage_loss = compute_loss_percentage(loss.numpy(), x_train.shape[1:])
    print(f"Epoch {epoch + 1}, Loss: {loss.numpy()}, Percentage Loss: {percentage_loss:.2f}%")

# Generate and visualize new images from VAE
def generate_images(vae, num_images=10):
    random_latent_vectors = tf.random.normal(shape=(num_images, 64))
    return vae.decoder(random_latent_vectors).numpy()

generated_images = generate_images(vae)
plt.figure(figsize=(12, 12))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(generated_images[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.show()
