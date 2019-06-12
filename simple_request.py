# USAGE
# python simple_request.py

# import the necessary packages
import requests
import cv2

# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://localhost:5000/predict"
IMAGE_PATH = "test_image/do_muoi.jpg"

image1 = cv2.imread(IMAGE_PATH)

# load the input image and construct the payload for the request
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

# submit the request
r = requests.post(KERAS_REST_API_URL, files=payload).json()

# ensure the request was sucessful
if r["success"]:
	print(r["predictions"])
# otherwise, the request failed-
else:
 	print("Request failed")