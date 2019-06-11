import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from imageio import imread
from scipy.spatial import distance
from keras.models import load_model
import pandas as pd
from tqdm import tqdm
import dlib
from model import create_model
from align import AlignDlib
import glob
import imutils
import time as time_
import flask
import io
from flask_cors import CORS, cross_origin
import PIL
from PIL import Image
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils

def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    # image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image
# PRE-PROCESSING
def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

def align_face(face):
    (h,w,c) = face.shape
    bb = dlib.rectangle(0, 0, w, h)
    return alignment.align(96, face, bb,landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
  
def load_and_align_images(filepaths):
    aligned_images = []
    for filepath in filepaths:
        img = cv2.imread(filepath)
        aligned = align_face(img)
        aligned = (aligned / 255.).astype(np.float32)
        aligned = np.expand_dims(aligned, axis=0)
        aligned_images.append(aligned)
            
    return np.array(aligned_images)
    
def calc_embs(filepaths, batch_size=64):
    pd = []
    for start in tqdm(range(0, len(filepaths), batch_size)):
        aligned_images = load_and_align_images(filepaths[start:start+batch_size])
        pd.append(nn4_small2.predict_on_batch(np.squeeze(aligned_images)))
    #embs = l2_normalize(np.concatenate(pd))
    embs = np.array(pd)

    return np.array(embs)
    
def align_faces(faces):
    aligned_images = []
    for face in faces:
        aligned = align_face(face)
        aligned = (aligned / 255.).astype(np.float32)
        aligned = np.expand_dims(aligned, axis=0)
        aligned_images.append(aligned)
        
    return aligned_images

def calc_emb_test(faces):
    pd = []
    aligned_faces = align_faces(faces)
    if(len(faces)==1):
        pd.append(nn4_small2.predict_on_batch(aligned_faces))
    elif(len(faces)>1):
        pd.append(nn4_small2.predict_on_batch(np.squeeze(aligned_faces)))
    #embs = l2_normalize(np.concatenate(pd))
    embs = np.array(pd)
    return np.array(embs)


def detect_face(test_image):
    show_image = test_image.copy()

    hogFaceDetector = dlib.get_frontal_face_detector()
    faceRects = hogFaceDetector(test_image, 0)
    
    faces = []
    
    for faceRect in faceRects:
        x1 = faceRect.left()
        y1 = faceRect.top()
        x2 = faceRect.right()
        y2 = faceRect.bottom()
        face = test_image[y1:y2,x1:x2]
        
        faces.append(face)

    # print("len(faces) = {0}".format(len(faces)))
    if(len(faces)==0):
        print("no face detected!")
        return
    else:    
        test_embs = calc_emb_test(faces)

    test_embs = np.concatenate(test_embs)
        
    people = []
    for i in range(test_embs.shape[0]):
        distances = []
        for j in range(len(train_paths)):
            distances.append(np.min([distance.euclidean(test_embs[i].reshape(-1), train_embs[k].reshape(-1)) for k in label2idx[j]]))
            #for k in label2idx[j]:
                #print(distance.euclidean(test_embs[i].reshape(-1), train_embs[k].reshape(-1)))
        print(distances)
        if np.min(distances)>threshold:
            people.append("unknown")
        else:
            res = np.argsort(distances)[:1]
            people.append(res)

    names = []
    title = ""
    for p in people:
        if p == "unknown":
            name = "unknown"
        else:
            name = df_train[(df_train['label']==p[0])].name.iloc[0]
        names.append(name)
        title = title + name + " "
        
    for i,faceRect in enumerate(faceRects):
        x1 = faceRect.left()
        y1 = faceRect.top()
        x2 = faceRect.right()
        y2 = faceRect.bottom()
        cv2.rectangle(show_image,(x1,y1),(x2,y2),(255,0,0),3)
        cv2.putText(show_image,names[i],(x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,0),3,cv2.LINE_AA)
        
    show_image = imutils.resize(show_image,width = 720)   
    # cv2.imshow("result",show_image)

# initialize our Flask application and Redis server
app = flask.Flask(__name__)
CORS(app)
# model = None
# INITIALIZE MODELS
nn4_small2 = create_model()

# nn4_small2.summary()

nn4_small2.load_weights('weights/nn4.small2.v1.h5')

alignment = AlignDlib('shape_predictor_68_face_landmarks.dat')

#LOAD TRAINING INFORMATION
train_paths = glob.glob("image1/*")
print(train_paths)

nb_classes = len(train_paths)

df_train = pd.DataFrame(columns=['image', 'label', 'name'])

for i,train_path in enumerate(train_paths):
    name = train_path.split("\\")[-1]
    images = glob.glob(train_path + "/*")
    for image in images:
        df_train.loc[len(df_train)]=[image,i,name]
        
# TRAINING
label2idx = []

for i in tqdm(range(len(train_paths))):
    label2idx.append(np.asarray(df_train[df_train.label == i].index))

train_embs = calc_embs(df_train.image)
np.save("train_embs.npy", train_embs)

train_embs = np.concatenate(train_embs)
threshold = 1

@app.route("/")
def homepage():
    return "Welcome to the PyImageSearch Keras REST API!"

@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        print("How are you")
        print(flask.request.files)
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            # print(image)
            image = Image.open(io.BytesIO(image))
            image = prepare_image(image, (100, 100))
            # cv2.imshow("image", image)
            # preprocess the image and prepare it for classification
            detect_face(image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    return flask.jsonify("a: 1")
    # return the data dictionary as a JSON response
    # return flask.jsonify(data)

# for debugging purposes, it's helpful to start the Flask testing
# server (don't use this for production
if __name__ == "__main__":
    print("* Starting web service...")
    app.run(debug = False, threaded = False)

# TEST
# test_paths = glob.glob("test_image/*.jpg")
# for path in test_paths:
#     test_image = cv2.imread(path)
#     detect_face(test_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

