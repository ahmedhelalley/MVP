from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import uuid
import numpy as np
import cv2
from scipy.fftpack import dct, idct
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient


connection_string = "DefaultEndpointsProtocol=https;AccountName=wwwsa;AccountKey=VJScHCYBJs5rJMQ8mvdG5KJwqkvk/PER72+uV90iYNcYeSexM8oAYubIPY8GtAzpPGkjoJuv2R1I+AStlNh36g==;EndpointSuffix=core.windows.net"
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_name = "pixelpress-compressed-images"
container_client = blob_service_client.get_container_client(container_name)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['COMPRESSED_FOLDER'] = 'compressed/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp'}

# Ensure upload and compressed directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['COMPRESSED_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def apply_dct(image):
    return dct(dct(image.T, norm='ortho').T, norm='ortho')

def apply_idct(dct_image):
    return idct(idct(dct_image.T, norm='ortho').T, norm='ortho')

def quantize(dct_image, quantization_matrix):
    return np.round(dct_image / quantization_matrix)

def dequantize(quantized_image, quantization_matrix):
    return quantized_image * quantization_matrix

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'})
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'})
    
    if image_file and allowed_file(image_file.filename):
        filename = secure_filename(image_file.filename)
        unique_id = str(uuid.uuid4())
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_id + "_" + filename)
        image_file.save(image_path)

        compression_level = request.form.get('compression_level', 'medium')
        
        compressed_image_path = compress_image(image_path, compression_level)
        return jsonify({'status': 'success', 'image_id': unique_id, 'compressed_image': compressed_image_path})
    
    return jsonify({'status': 'error', 'message': 'File not allowed'})

def compress_image(image_path, compression_level):
    quantization_matrices = {
        'low': np.ones((8, 8)),
        'medium': np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ]),
        'high': np.array([
            [4, 3, 2, 4, 6, 10, 13, 15],
            [3, 3, 3, 5, 7, 14, 15, 14],
            [4, 3, 4, 6, 10, 14, 17, 14],
            [4, 5, 6, 8, 13, 22, 20, 16],
            [5, 6, 10, 14, 17, 27, 26, 19],
            [6, 9, 14, 16, 20, 26, 28, 23],
            [12, 16, 20, 22, 26, 30, 30, 25],
            [18, 23, 24, 25, 28, 25, 26, 25]
        ])
    }

    quantization_matrix = quantization_matrices.get(compression_level, quantization_matrices['medium'])

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    
    height, width = image.shape

    compressed_image = np.zeros_like(image, dtype=np.float32)
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = image[i:i+8, j:j+8]
            dct_block = apply_dct(block - 128)
            quantized_block = quantize(dct_block, quantization_matrix)
            dequantized_block = dequantize(quantized_block, quantization_matrix)
            decompressed_block = apply_idct(dequantized_block) + 128
            compressed_image[i:i+8, j:j+8] = decompressed_block

    compressed_image = np.clip(compressed_image, 0, 255).astype(np.uint8)
    compressed_image_path = os.path.join(app.config['COMPRESSED_FOLDER'], os.path.basename(image_path))
    cv2.imwrite(compressed_image_path, compressed_image)
    
    #return compressed_image_path

    blob_client = container_client.get_blob_client(os.path.basename(compressed_image_path))
    with open(compressed_image_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)

@app.route('/download/<filename>')
def download_image(filename):
    blob_client = container_client.get_blob_client(filename)
    download_stream = blob_client.download_blob()
    return Response(download_stream.readall(), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)