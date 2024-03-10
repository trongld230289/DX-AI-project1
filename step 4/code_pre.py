import requests
from urllib.parse import urlparse
from io import BytesIO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import os, time, uuid

from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials

# helper function
def show_image_in_cell(img_url):
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content))
    plt.figure(figsize=(20,10))
    plt.imshow(img)
    plt.show()

TRAINING_ENDPOINT = "https://dxcustomvision.cognitiveservices.azure.com/"
training_key = "eb69e1f90f34440c944b60f8969e105a"
training_resource_id = '/subscriptions/d9a31020-17ef-44e7-9836-f10843b2cf0a/resourceGroups/AChau-Storage-Rg/providers/Microsoft.CognitiveServices/accounts/dxcustomvision'

PREDICTION_ENDPOINT = 'https://dxcustomvision-prediction.cognitiveservices.azure.com/'
prediction_key = "db277faf9f0f433ab9e5a0c8af4f7818"
prediction_resource_id = "/subscriptions/d9a31020-17ef-44e7-9836-f10843b2cf0a/resourceGroups/AChau-Storage-Rg/providers/Microsoft.CognitiveServices/accounts/dxcustomvision-Prediction"

# Instantiate and authenticate the training client with endpoint and key
training_credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(TRAINING_ENDPOINT, training_credentials)
trainer.api_version

prediction_credentials = ApiKeyCredentials(
    in_headers={"Prediction-key": prediction_key}
)
predictor = CustomVisionPredictionClient(PREDICTION_ENDPOINT, prediction_credentials)

project_id = "0bbedf09-7377-40f3-9230-033ac72d04b8"
publish_iteration_name = "Iteration2"

path = "source/"

for file in os.listdir(path):
    with open(
        f"{path}/{file}", "rb"
    ) as image_contents:
        results = predictor.detect_image(
            project_id, publish_iteration_name, image_contents.read()
        )
    for prediction in results.predictions:
        print(
            f"{prediction.tag_name} in {file} with probability is {round(prediction.probability * 100, 2)}%"
        )