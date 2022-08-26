import imageio
import tensorflow as tf
from flask import Flask, render_template, request, session
import os
import config
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils


app = Flask(__name__, static_url_path="/static", template_folder='templates',
            static_folder='static')


UPLOAD_FOLDER = os.path.join('static', 'uploads')
OUTPUT_FOLDER = os.path.join ('static', 'outputs')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.secret_key = config.secret_key



def load_image_into_numpy_array(path):
    return np.array(Image.open(path))


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/', methods=['GET', 'POST'])
def upload_image():

    uploaded_img = request.files['uploaded-file']
    img_filename = secure_filename(uploaded_img.filename)
    # Upload file to database (defined uploaded folder in static path)
    uploaded_img.save('static/uploads/' + img_filename)
    # Storing uploaded file path in flask session
    session['uploaded_img_file_path'] = os.path.join(
        app.config['UPLOAD_FOLDER'], img_filename)

    print(session['uploaded_img_file_path'])

    return render_template('display_image.html')


@app.route('/show_image', methods=['GET', 'POST'])
def display_image():
    image_path = session.get('uploaded_img_file_path', None)
    return render_template('show_image.html', user_image=image_path)


@app.route('/detect_image', methods=['GET', 'POST'])
def detect():
    
    detect_fn = tf.saved_model.load(config.path_to_saved_model)
    category_index = label_map_util.create_category_index_from_labelmap(
    config.path_to_labels, use_display_name=True)

    image_path = session.get('uploaded_img_file_path', None)
    image_np = load_image_into_numpy_array(image_path)

    input_tensor = tf.convert_to_tensor(image_np)

    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(
        np.int64)

    image_np_with_detections = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=10,
        min_score_thresh=.5,
        agnostic_mode=False
        )
    
    plt.figure()
    session['output_img_file_path'] = os.path.join(
        app.config['OUTPUT_FOLDER'], config.output_file)
    
    output_path = session.get('output_img_file_path', None)

    return render_template('detect_image.html', name = output_path)


if __name__ == '__main__':
    app.run(port=8300, debug=True)
