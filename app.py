from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import base64
import io
import numpy as np
import matplotlib.pyplot as plt
from kmeans_v2 import CustomKMeans as ckm
from sklearn.metrics import pairwise_distances
import matplotlib.backends.backend_agg as agg

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.abspath('uploads')

@app.route('/', methods=['GET', 'POST'])
def index():
    original_image = None
    segmented_image = None

    if request.method == 'POST':
        image = request.files['image']
        num_clusters = int(request.form['clusters'])

        if image:
            filename = secure_filename(image.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(image_path)

            original_image, segmented_image = segment_image(image_path, num_clusters)

    return render_template('index.html', original_image=original_image, segmented_image=segmented_image)

def segment_image(selected_image_path, num_clusters):
    image = plt.imread(selected_image_path) 

    n_pixels = image.shape[0] * image.shape[1]
    image_pixels = image.reshape(n_pixels, -1)

    kmeans_custom = ckm(n_clusters=num_clusters)
    kmeans_custom.fit(image_pixels)
    cluster_labels = kmeans_custom.predict(image_pixels)
    segmented_img = kmeans_custom.centroids[cluster_labels].reshape(image.shape)

    original_image_b64 = image_to_base64(image)
    segmented_image_b64 = image_to_base64(segmented_img)

    return f"data:image/png;base64,{original_image_b64}", f"data:image/png;base64,{segmented_image_b64}"

def image_to_base64(image):
    img_buffer = io.BytesIO()
    plt.figure()
    plt.axis('off')
    plt.imshow(image)
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
    plt.close()
    return img_base64

if __name__ == '__main__':
    app.run(debug=True)
