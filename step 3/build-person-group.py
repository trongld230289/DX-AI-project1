from video_indexer import VideoIndexer
import glob, os, sys, time, uuid
from azure.cognitiveservices.vision.face import FaceClient
from azure.cognitiveservices.vision.face.models import TrainingStatusType
from msrest.authentication import CognitiveServicesCredentials
import os

face_endpoint = "https://dx-face.cognitiveservices.azure.com/"
face_key = ""

face_client = FaceClient(face_endpoint, CognitiveServicesCredentials(face_key))
face_images = [file for file in glob.glob("./ids/*.png")]

# For reference, see:
# https://github.com/Azure-Samples/cognitive-services-quickstart-code/blob/master/python/Face/DetectIdentifyFace.py

def build_person_group(client, face_images):
    person_group_id = str(uuid.uuid4())
    client.person_group.create(person_group_id=person_group_id, name=person_group_id)

    person = client.person_group_person.create(
        person_group_id=person_group_id, name=person_group_id
    )

    for face_image in face_images:
        with open(face_image, "rb") as w:
            client.person_group_person.add_face_from_stream(
                person_group_id, person.person_id, w
            )

    client.person_group.train(person_group_id)

    while True:
        training_status = client.person_group.get_training_status(person_group_id)
        print("Training status: {}.".format(training_status.status))
        if training_status.status is TrainingStatusType.succeeded:
            break
        elif training_status.status is TrainingStatusType.failed:
            client.person_group.delete(person_group_id=person_group_id)
            sys.exit("Training the person group is failed.")
        time.sleep(10)


def detect_faces(client, face_images):
    face_ids = {}
    for face_image in face_images:
        image = open(face_image, "rb")
        time.sleep(10)

        faces = client.face.detect_with_stream(image)

        for face in faces:
            print(
                "Face ID",
                face.face_id,
                "found in image",
                os.path.splitext(image.name)[0] + ".png",
            )
            face_ids[image.name] = face.face_id

    return face_ids


print("Building person group...")
build_person_group(face_client, face_images)
print("Person group built")

print("Detecting faces in images from ids folder...")
ids = detect_faces(face_client, face_images)
print("Face IDs:", ids)