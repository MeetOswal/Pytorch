from flask import Flask, request, jsonify

from torch_utils import transform_image, get_prediction

app = Flask(__name__)

ALLOWED_EXTENTION = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENTION

@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({'error': 'No File'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'File not supported'})

        try:
            image_bytes = file.read()
            tensor = transform_image(image_bytes)
            prediction = get_prediction(tensor)
            data = {'prediction': prediction.item(), 'class_name': str(prediction.item())} # tensor has only one element
            return jsonify(data)
        except:
           return jsonify({'error': 'error during prediction'})

if __name__ == '__main__':
    app.run(debug= True, port= 5000)

