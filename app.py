# import cv2
#
# # Read the input image
# img = cv2.imread('test2.jpg')
#
# # Convert into grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # Load the cascade
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
#
# # Detect faces
# faces = face_cascade.detectMultiScale(gray, 1.1, 4)
#
# # Draw rectangle around the faces and crop the faces
# for (x, y, w, h) in faces:
#     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
#     faces = img[y:y + h, x:x + w]
#     # cv2.imshow("face",faces)
#     cv2.imwrite('face.jpg', faces)
#
# # Display the output
# cv2.imwrite('detcted.jpg', img)
# # cv2.imshow('img', img)
# cv2.waitKey()

from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
from align import detect_image
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_PATH'] = 20000000

@app.route('/')
def hello():
    return 'The server is up and running 1.2'


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save('uploads/'+secure_filename(f.filename))
        zoom = request.form.get('zoom')
        aspect =request.form.get('aspect')
        # img = cv2.imread(secure_filename(f.filename))
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # for (x, y, w, h) in faces:
        #     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        #     faces = img[y:y + h, x:x + w]
        #     # cv2.imshow("face",faces)
        #     cv2.imwrite('face.jpg', faces)
        # cv2.imwrite('detcted.jpg', img)
        detect_image('uploads/'+secure_filename(f.filename), zoom, aspect)
        # cv2.imshow('img', img)
        return send_file('uploads/output.JPG', mimetype='image/jpeg')
        # return 'file uploaded successfully'

if __name__ == '__main__':
    app.run(debug=True)
