import tensorflow as tf
import cv2
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("dogs-vs-cats.h5")

'''
# Read the image to be tested
image = cv2.imread("photos\dog2.jpg")

# Resize the image to the input shape of the model
image = cv2.resize(image, (150, 150))

# Convert the image to a numpy array and add an additional dimension
image = np.expand_dims(image, axis=0)

# Normalize the image
image = image / 255.0

# Make a prediction
predictions = model.predict(image)

# Get the class with the highest probability
class_index = np.argmax(predictions[0])

# Define the classes
classes = ["cat", "dog"]

# Print the predicted class
print("The model predicts that the image is a:", classes[class_index])
'''

# Cat recognition form Cameraś
cap = cv2.VideoCapture(0)

while 1:  

    # reads frames from a camera  
    ret, image = cap.read()  
    image2 = image
    
    # Resize the image to the input shape of the model
    image = cv2.resize(image, (150, 150))

    # Convert the image to a numpy array and add an additional dimension
    image = np.expand_dims(image, axis=0)

    # Normalize the image
    image = image / 255.0

    # Make a prediction
    predictions = model.predict(image)

    # Get the class with the highest probability
    class_index = np.argmax(predictions[0])

    # Define the classes
    classes = ["cat", "dog"]

    # Print the predicted class
    print("The model predicts that the image is a:", classes[class_index])
  
  
    # Display an image in a window  
    cv2.imshow('img',image2)  
  
    # Wait for Esc key to stop  
    k = cv2.waitKey(30) & 0xff
    if k == 27:  
        break
  
# Close the window  
cap.release()  
# De-allocate any associated memory usage  
cv2.destroyAllWindows()
