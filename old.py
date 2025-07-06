import cv2
import numpy as np
import pickle
import pandas as pd
from keras.models import Sequential
from keras.layers import Activation, Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import load_model
from keras import losses
from gtts import gTTS
import os
import platform

# Path to Haar cascade for hand detection
haar_cascade = 'hand_detection_cascade.xml'
image_no = 0

# Speak the given text
def speak(ip_text):
    tts = gTTS(text=ip_text, lang='en')
    tts.save("pcvoice.mp3")
    if platform.system() == "Windows":
        os.system("start pcvoice.mp3")
    else:
        os.system("mpg321 pcvoice.mp3")

# Convert label index to corresponding alphabet
def convert_label(x):
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return labels[x] if 0 <= x < len(labels) else "?"

def main():
    # Load training and testing datasets
    train_data = pd.read_csv('sign_mnist_train.csv', sep=',', header=None, low_memory=False)
    test_data = pd.read_csv('sign_mnist_test.csv', sep=',', header=None, low_memory=False)

    train_data = train_data[1:]
    test_data = test_data[1:]

    # Extract labels and pixel data
    observed_train_values = train_data.iloc[:, 0].astype(int).tolist()
    only_train_pixels = train_data.iloc[:, 1:].astype(int).values.tolist()

    observed_test_values = test_data.iloc[:, 0].astype(int).tolist()
    only_test_pixels = test_data.iloc[:, 1:].astype(int).values.tolist()

    print("Train Shape: ", len(only_train_pixels), "x", len(only_train_pixels[0]))
    print("Train Labels Shape: ", len(observed_train_values))
    print("Test Shape: ", len(only_test_pixels), "x", len(only_test_pixels[0]))
    print("Test Labels Shape: ", len(observed_test_values))

    try:
        model = load_model('cnn_model.h5', custom_objects={'loss_categorical_crossentropy': losses.categorical_crossentropy})
    except Exception as e:
        print("Model loading failed:", e)
        return

    x, y, w, h = 300, 100, 300, 300
    captureFlag = False

    # Start webcam
    live_stream = cv2.VideoCapture(0)
    global image_no

    while True:
        keypress = cv2.waitKey(1)
        success, img = live_stream.read()
        if not success:
            continue

        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_to_use = img
        hand_cascade = cv2.CascadeClassifier(haar_cascade)
        hands = hand_cascade.detectMultiScale(img_to_use, 1.3, 5)

        for (x, y, w, h) in hands:
            cv2.rectangle(img_to_use, (x, y), (x + w, y + h), (0, 0, 255), 2)

        if keypress == ord('c'):
            captureFlag = True

        if captureFlag:
            captureFlag = False
            roi = gray[y:y+h, x:x+w]
            resized_image = cv2.resize(roi, (28, 28))
            resized_image = resized_image.astype("float32") / 255.0
            resized_image = np.reshape(resized_image, [1, 28, 28, 1])
            
            prediction = model.predict(resized_image)
            predicted_index = np.argmax(prediction)
            predicted_letter = convert_label(predicted_index)

            print("Predicted:", predicted_letter)
            speak(predicted_letter)

            # Save the image
            cv2.imwrite(f"./saved/image{image_no}.jpg", resized_image[0, :, :, 0] * 255)
            image_no += 1

        if keypress == 27:
            break

        cv2.imshow("Hand Detection", img_to_use)

    live_stream.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
