import sys
import os
import tensorflow as tf
from gtts import gTTS
from playsound import playsound

# Disable TensorFlow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Use TensorFlow 1.x compatibility mode for deprecated functions
tf.compat.v1.disable_eager_execution()

# Language used by Google Text to Speech
language = 'en'

# Get the image path from command line argument
if len(sys.argv) < 2:
    print("Usage: python classify.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

# Read the image data
image_data = tf.io.gfile.GFile(image_path, 'rb').read()

# Load label lines
label_lines = [line.strip() for line in tf.io.gfile.GFile("training_set_labels.txt")]

# Load trained model's graph
with tf.io.gfile.GFile("trained_model_graph.pb", 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

# Start a new TensorFlow session
with tf.compat.v1.Session() as sess:
    # Feed image into graph and get predictions
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

    # Sort predictions by score (descending)
    sorted_predictions = predictions[0].argsort()[::-1]

    # Get top prediction
    top_index = sorted_predictions[0]
    predicted_label = label_lines[top_index].upper()
    predicted_score = predictions[0][top_index]

    print(f"\nPredicted Letter: {predicted_label}\tScore: {predicted_score:.5f}\n")

    # Speak the prediction
    prediction_text = f"The predicted letter is {predicted_label}"
    tts = gTTS(text=prediction_text, lang=language, slow=False)
    tts.save("prediction.mp3")
    playsound("prediction.mp3")

    # Optional: show all predictions
    # for i in sorted_predictions:
    #     print(f"{label_lines[i]} (score = {predictions[0][i]:.5f})")
