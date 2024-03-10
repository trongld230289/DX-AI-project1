import io
import datetime
import pandas as pd
from PIL import Image
import requests
import io
import glob, os, sys, time, uuid

from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import requests

from urllib.parse import urlparse
from io import BytesIO
from PIL import Image, ImageDraw

from video_indexer import VideoIndexer
from azure.cognitiveservices.vision.face import FaceClient
from azure.cognitiveservices.vision.face.models import TrainingStatusType
from msrest.authentication import CognitiveServicesCredentials

# get from https://www.videoindexer.ai/
CONFIG = {
    'SUBSCRIPTION_KEY': '',
    'LOCATION': 'trial',
    'ACCOUNT_ID': ''
}

video_analysis = VideoIndexer(
    vi_subscription_key=CONFIG['SUBSCRIPTION_KEY'],
    vi_location=CONFIG['LOCATION'],
    vi_account_id=CONFIG['ACCOUNT_ID']
)

video_analysis.check_access_token()

uploaded_video_id = video_analysis.upload_to_video_indexer(
    input_filename="video/avkash-boarding-pass.mp4",
    video_name="avkash-boarding-pass-" + str(uuid.uuid4()),
    video_language="English",
)

# Wait for the video to be processed
while True:
    info = video_analysis.get_video_info(uploaded_video_id, video_language="English")
    if info['state'] == "Processed":
        break
    time.sleep(10)  # wait for 10 seconds before checking again

info = video_analysis.get_video_info(uploaded_video_id, video_language="English")

print("Video id:", uploaded_video_id)

images = []
for thumbnail in info["videos"][0]["insights"]["faces"][0]["thumbnails"]:
    file_name = thumbnail["fileName"]
    thumbnail_id = thumbnail["id"]
    img_code = video_analysis.get_thumbnail_from_video_indexer(
        uploaded_video_id, thumbnail_id
    )
    img_stream = io.BytesIO(img_code)
    img = Image.open(img_stream)
    images.append(img)

print("Extracted faces from video:")
for i, img in enumerate(images):
    img.save("extract-faces/face" + str(i + 1) + ".jpg")

sentiments = info["summarizedInsights"]["sentiments"]
print("Sentiments:")
for sentiment in sentiments:
    print("-" + sentiment["sentimentKey"])

emotions = info["summarizedInsights"]["emotions"]
print("Emotions:")
for emotion in emotions:
    print("-" + emotion["type"])


# Get the access token
token_url = f"https://api.videoindexer.ai/auth/central-us/Accounts/{CONFIG['ACCOUNT_ID']}/AccessToken?allowEdit=true"
token_response = requests.get(token_url, headers={'Ocp-Apim-Subscription-Key': CONFIG['SUBSCRIPTION_KEY']})
access_token = token_response.text.strip('"')

# Delete the video
delete_url = f"https://api.videoindexer.ai/central-us/Accounts/{CONFIG['ACCOUNT_ID']}/Videos/{uploaded_video_id}?accessToken={access_token}"
delete_response = requests.delete(delete_url)

if delete_response.status_code == 200:
    print("Video deleted successfully.")
else:
    print("Failed to delete video.")