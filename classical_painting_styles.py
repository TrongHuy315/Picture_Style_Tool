import tensorflow as tf
from PIL import Image
import numpy as np

def load_and_process_image(image_path, target_size=(256, 256)):
    if isinstance(image_path, str):
        image = Image.open(image_path)
    elif isinstance(image_path, Image.Image):
        image = image_path
    else:
        raise ValueError("Input should be a file path or a PIL Image object.")

    # Convert to RGB mode to ensure 3 channels
    image = image.convert('RGB')
    
    # Resize image
    image = image.resize(target_size)
    
    # Convert to numpy array and normalize
    image = np.array(image, dtype=np.float32) / 255.0

    # Add batch dimension explicitly
    image = np.expand_dims(image, axis=0)
    
    return tf.convert_to_tensor(image, dtype=tf.float32)

def deprocess_image(processed_image):
    """Deprocess the tensor back to an image."""
    img = processed_image.numpy()[0]  # Remove batch dimension
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(img)

# # Chuyển đổi ảnh PIL hoặc TensorFlow Tensor thành dạng phù hợp cho ESRGAN
# def preprocess_for_esrgan(image):
#     if isinstance(image, Image.Image):  # Nếu là PIL Image
#         image = np.array(image, dtype=np.float32) / 255.0  # Chuẩn hóa về [0, 1]
#         image = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
#     elif isinstance(image, tf.Tensor):  # Nếu là Tensor
#         if image.dtype != tf.float32:
#             image = tf.image.convert_image_dtype(image, dtype=tf.float32)  # Chuẩn hóa về [0, 1]
#         if len(image.shape) == 3:  # Nếu thiếu chiều batch
#             image = tf.expand_dims(image, axis=0)
#     return image

# # Chuyển đổi kết quả ESRGAN thành ảnh PIL
# def postprocess_from_esrgan(esrgan_tensor):
#     # Xoá chiều batch
#     esrgan_tensor = tf.squeeze(esrgan_tensor, axis=0)
#     # Chuyển giá trị từ [0, 1] về [0, 255]
#     esrgan_image = tf.clip_by_value(esrgan_tensor * 255.0, 0, 255)
#     esrgan_image = tf.cast(esrgan_image, dtype=tf.uint8).numpy()
#     return Image.fromarray(esrgan_image)

# def compute_style_transfer(content_image, style_image, iterations=100, content_weight=1e4, style_weight=1e-2):
#     """Perform style transfer using VGG19."""
#     # Ensure inputs are proper 4D tensors
#     if len(content_image.shape) != 4:
#         raise ValueError(f"Content image must have 4 dimensions, got shape {content_image.shape}")
#     if len(style_image.shape) != 4:
#         raise ValueError(f"Style image must have 4 dimensions, got shape {style_image.shape}")

#     # Print shapes for debugging
#     print("Content image shape:", content_image.shape)
#     print("Style image shape:", style_image.shape)

#     # Load VGG19 model
#     vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
#     vgg.trainable = False
    
#     # Create model with specific layers
#     model = tf.keras.Model([vgg.input], [
#         vgg.get_layer(name).output for name in ["block5_conv2", "block1_conv1"]
#     ])

#     def gram_matrix(features):
#         """Compute Gram matrix for style representation."""
#         # Ensure input has correct shape
#         if len(features.shape) != 4:
#             raise ValueError(f"Features must have 4 dimensions, got shape {features.shape}")
            
#         batch_size = tf.shape(features)[0]
#         height = tf.shape(features)[1]
#         width = tf.shape(features)[2]
#         channels = tf.shape(features)[3]
        
#         # Reshape features to 2D matrix
#         features = tf.reshape(features, (batch_size, height * width, channels))
        
#         # Compute gram matrix
#         gram = tf.matmul(features, features, transpose_a=True)
        
#         # Normalize
#         denominator = tf.cast(height * width * channels, tf.float32)
#         return gram / denominator

#     def compute_loss(model, content, style, generated):
#         """Calculate combined content and style loss."""
#         content_features, style_features = model(content), model(style)
#         generated_features = model(generated)

#         content_loss = tf.reduce_mean(tf.square(content_features[0] - generated_features[0]))
        
#         style_loss = 0
#         for sf, gf in zip(style_features, generated_features):
#             style_loss += tf.reduce_mean(tf.square(gram_matrix(sf) - gram_matrix(gf)))

#         total_loss = content_weight * content_loss + style_weight * style_loss
#         return total_loss

#     # Initialize generated image
#     generated_image = tf.Variable(content_image)
    
#     # Create optimizer
#     optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)

#     # Create progress bar
#     progress_bar = st.progress(0)
#     status_text = st.empty()

#     for i in range(iterations):
#         with tf.GradientTape() as tape:
#             loss = compute_loss(model, content_image, style_image, generated_image)
        
#         print(f"Iteration {i+1}, Loss: {loss.numpy()}")
        
#         grads = tape.gradient(loss, generated_image)
#         optimizer.apply_gradients([(grads, generated_image)])
#         generated_image.assign(tf.clip_by_value(generated_image, 0.0, 1.0))

#         # Update progress
#         progress = (i + 1) / iterations
#         progress_bar.progress(progress)
#         status_text.text(f"Processing... Iteration {i + 1}/{iterations}")

#     status_text.text("Style transfer complete!")
#     return generated_image
