from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
import os

face_endpoint = "https://dx-face.cognitiveservices.azure.com/"
face_key = ""

face_client = FaceClient(face_endpoint, CognitiveServicesCredentials(face_key))
print("Face client created")
print("Detecting faces in images from ids folder...")
for file in os.listdir("ids"):
    if file.endswith(".png"):
        with open(f"ids/{id}", "rb") as id_file:
            print(f"Detecting faces in {id}...")
            faces = face_client.face.detect_with_stream(id_file)
            for face in faces:
                print(f"Found Face ID {face.face_id} in {id}")