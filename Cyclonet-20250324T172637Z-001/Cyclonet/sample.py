import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf

print(tf.__version__)
model = tf.keras.models.load_model('./Model.h5')

# Load and preprocess the input image
def preprocess_image(image_path, target_size):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Example: Single input image test
image_path = "test_image.png"
target_size = (512, 512)
input_image = preprocess_image(image_path, target_size=target_size)

# Test image with the model
prediction = model.predict(input_image, verbose=1).round(2)
cyclone_effect_percentage = prediction[0][0]

# Determine cyclone intensity
if cyclone_effect_percentage < 50:  # Adjust these thresholds as needed
    intensity = "Low"
elif 50 <= cyclone_effect_percentage < 100:
    intensity = "Medium"
else:
    intensity = "High"

# Visualization
plt.figure(figsize=(10, 5))  # Increase figure width to make space for the textbox

# Display the image
plt.subplot(1, 2, 1)  # Create a subplot for the image
plt.imshow(input_image[0])
plt.title(f"Cyclone Effect: {cyclone_effect_percentage:.2f} knots")
plt.axis("off")

# Display the textbox
plt.subplot(1, 2, 2)  # Create a subplot for the text
plt.text(0.1, 0.5, f"Intensity: {intensity}", fontsize=12, verticalalignment='center')
plt.axis("off") #turn off axis for the text subplot

plt.show()