import glob
import sys
import time
import uuid
from azure.ai.formrecognizer import FormRecognizerClient
from azure.ai.formrecognizer import FormTrainingClient
from azure.core.credentials import AzureKeyCredential
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType
from urllib.parse import urlparse
from io import BytesIO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import os
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials
import pandas as pd
from datetime import datetime

TRAINING_ENDPOINT = "https://dxcustomvision.cognitiveservices.azure.com/"
training_key = "eb69e1f90f34440c944b60f8969e105a"
training_resource_id = '/subscriptions/d9a31020-17ef-44e7-9836-f10843b2cf0a/resourceGroups/AChau-Storage-Rg/providers/Microsoft.CognitiveServices/accounts/dxcustomvision'

PREDICTION_ENDPOINT = 'https://dxcustomvision-prediction.cognitiveservices.azure.com/'
prediction_key = "db277faf9f0f433ab9e5a0c8af4f7818"
prediction_resource_id = "/subscriptions/d9a31020-17ef-44e7-9836-f10843b2cf0a/resourceGroups/AChau-Storage-Rg/providers/Microsoft.CognitiveServices/accounts/dxcustomvision-Prediction"

subscription_key = "9d497cb35fb24b8294cfee8deb6b0c15"
location = "trial"
account_id = "88c51449-daf3-4a29-aa56-b5dc3b61f538"

face_endpoint = "https://dx-face.cognitiveservices.azure.com/"
face_key = "33c804e1892847999a03459cdad34679"

form_endpoint = "https://dx-doc-intelligence.cognitiveservices.azure.com/"
form_key = "6f40a3949f2848539a4082c9d6d23f28"

manifest = pd.read_csv("flight_manifest_trongld1.csv")

form_training_client = FormTrainingClient(form_endpoint, AzureKeyCredential(form_key))
form_recognizer_client = FormRecognizerClient(form_endpoint, AzureKeyCredential(form_key))

face_client = FaceClient(face_endpoint, face_key)

# training_process = form_training_client.begin_training(trainingDataUrl, use_training_labels=True)
# custom_model = training_process.result()

def build_person_group(client, person_group_id, face_images):
    print("Create and build a person group...")
    # Create empty Person Group. Person Group ID must be lower case, alphanumeric, and/or with '-', '_'.
    print("Person group ID:", person_group_id)
    client.person_group.create(person_group_id=person_group_id, name=person_group_id)

    person = client.person_group_person.create(
        person_group_id=person_group_id, name=person_group_id
    )

    for image_p in face_images:
        with open(image_p, "rb") as w:
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
            sys.exit("Training the person group has failed.")
        time.sleep(5)

person_group_id = str(uuid.uuid4())
face_client = FaceClient(face_endpoint, CognitiveServicesCredentials(face_key))
face_images = [file for file in glob.glob("./ids/*.png")]

build_person_group(face_client, person_group_id, face_images)

def extract_id_data(id_path):
    with open(id_path, "rb") as f:
        poller = form_recognizer_client.begin_recognize_identity_documents(identity_document=f)

    id_documents = poller.result()[0].to_dict()

    id_dict = {}

    id_dict['first_name'] = id_documents['fields']['FirstName']['value'].strip().lower()
    id_dict['last_name'] = id_documents['fields']['LastName']['value'].strip().lower()
    id_dict['dob'] = id_documents['fields']['DateOfBirth']['value']
    # print(id_dict)
    return id_dict

def extract_boarding_pass(model_id, file_name):
    with open(file_name, "rb") as f:
        poller = form_recognizer_client.begin_recognize_custom_forms(
            model_id=model_id,
            form=f,
            include_field_elements=True,
        )
    boarding_pass = poller.result()[0].to_dict()

    boarding_data_dict = {}
    boarding_data_dict['name'] = boarding_pass['fields']['Passenger Name']['value']
    boarding_data_dict['flight_num'] = boarding_pass['fields']['Flight No']['value']
    boarding_data_dict['seat'] = boarding_pass['fields']['Seat']['value']
    boarding_data_dict['from'] = boarding_pass['fields']['From']['value']
    boarding_data_dict['to'] = boarding_pass['fields']['To']['value']
    boarding_data_dict['date'] = datetime.strptime(boarding_pass['fields']['Date']['value'], "%B %d, %Y")
    boarding_data_dict['time'] = datetime.strptime(boarding_pass['fields']['Boarding Time']['value'][:-3].strip(), "%I:%M %p")
    # print(boarding_data_dict)
    return boarding_data_dict

# print("Extracting data from ID and boarding pass...")
# id_data = []
id_path = "ids/"
# for file in os.listdir(id_path):
#     if file.endswith(".png"):
#         current_data = extract_id_data(id_path + file)
#         print(current_data)
#         id_data.append(current_data)

# print("Extracting data from boarding pass...")
model_id = "e43d6db0-d749-496c-a14b-8eac90e05ccb"
# boarding_data = []
boarding_path = "boarding-pass/"
# for file in os.listdir(boarding_path):
#     if file.endswith(".pdf"):
#         current_data = extract_boarding_pass(model_id, boarding_path + file)
#         print(current_data)
#         boarding_data.append(current_data)

# with open('boarding_data', 'wb') as fp:
#     pickle.dump(boarding_data, fp)


def validate_passenger(manifest_df, id_path, boarding_path, person_name):
    print("VALIDATING PASSENGER:", person_name)
    # Set initial validation results
    name_validation = False
    dob_validation = False
    boarding_pass_validation = False
    person_id_validation = False
    luggage_validation = is_validate_baggege("bagge/", person_name)
    print("luggage_validation", luggage_validation)
    # Get data
    id_data = extract_id_data(id_path)
    boarding_data = extract_boarding_pass(model_id, boarding_path)
    

    # print('First Name', manifest_df['First Name'].str.lower())
    # print("id_data['first_name']", id_data['first_name'])
    # print("ahihi",manifest_df.index[(manifest_df['First Name'].str.lower() == id_data['first_name'])])
    manifest_index = manifest_df.index[(manifest_df['First Name'].str.lower() == id_data['first_name']) & (manifest_df['Last Name'].str.lower() == id_data['last_name'])]
    # print("manifest_index: ", manifest_index)

    # 3-Way Person Name Validation
    # print("id_data['first_name']", id_data['first_name'])
    # print("id_data['last_name']", id_data['last_name'])
    # print("boarding_data['name']", boarding_data['name'])

    boarding_data['name'] = boarding_data['name'].strip().lower()
    if (id_data['first_name'] in boarding_data['name'] and id_data['last_name'] in boarding_data['name']):
        name_validation = True
    print("name_validation", name_validation)
    # DoB Validation
    if len(manifest_index > 0):
        'success'
        date_of_birth_id = id_data['dob']
        date_of_birth_manifest = datetime.strptime(manifest_df.iloc[manifest_index]['DateofBirth'][manifest_index[0]], "%m/%d/%Y")
        if (date_of_birth_id.year == date_of_birth_manifest.year and \
            date_of_birth_id.month == date_of_birth_manifest.month and \
                date_of_birth_id.day == date_of_birth_manifest.day):

                manifest_df.loc[manifest_index, 'DoBValidation'] = True
                dob_validation = True
    print("dob_validation", dob_validation)
    # Boarding Pass Validation: flight number, seat number, class, origin, destination, flight date, and flight time
    manifest_flight_num = ""
    if len(manifest_index > 0):
        manifest_flight_num = manifest_df.iloc[manifest_index]['Flight No.'][manifest_index[0]]
        manifest_seat_num = manifest_df.iloc[manifest_index]['SeatNo'][manifest_index[0]].strip().lower()
        manifest_origin = manifest_df.iloc[manifest_index]['Origin'][manifest_index[0]].strip().lower()
        manifest_destination = manifest_df.iloc[manifest_index]['Destination'][manifest_index[0]].strip().lower()
        manifest_flight_date = datetime.strptime(manifest_df.iloc[manifest_index]['Date'][manifest_index[0]], "%d-%b-%y")
        manifest_flight_time = datetime.strptime(manifest_df.iloc[manifest_index]['Time'][manifest_index[0]].strip(), "%H:%M")

    # print("manifest_flight_num", manifest_flight_num)
    # print("manifest_seat_num", manifest_seat_num)
    # print("manifest_origin", manifest_origin)
    # print("manifest_destination", manifest_destination)
    # print("manifest_flight_date", manifest_flight_date)
    # print("manifest_flight_time", manifest_flight_time)
    # # print boarding data
    # print("boarding_data['flight_num']", boarding_data['flight_num'])
    # print("boarding_data['seat']", boarding_data['seat'].strip().lower())
    # print("boarding_data['from']", boarding_data['from'].strip().lower())
    # print("boarding_data['to']", boarding_data['to'].strip().lower())
    # print("boarding_data['date']", boarding_data['date'])
    # print("boarding_data['time']", boarding_data['time'])

    # print(str(boarding_data['flight_num']) == str(manifest_flight_num))
    # print(boarding_data['seat'].strip().lower() == manifest_seat_num.strip().lower())
    # print(boarding_data['from'].strip().lower() == manifest_origin.strip().lower())
    # print(boarding_data['to'].strip().lower() == manifest_destination.strip().lower())
    # print(boarding_data['date'] == manifest_flight_date)
    # print(boarding_data['time'] == manifest_flight_time)

    if str(boarding_data['flight_num']) == str(manifest_flight_num) and \
        boarding_data['seat'].strip().lower() == manifest_seat_num.strip().lower() and \
            boarding_data['from'].strip().lower() == manifest_origin.strip().lower() and \
                boarding_data['to'].strip().lower() == manifest_destination.strip().lower() and \
                    boarding_data['date'] == manifest_flight_date and \
                        boarding_data['time'] == manifest_flight_time:

        boarding_pass_validation = True
    print("boarding_pass_validation", boarding_pass_validation)
    
    with open(id, "rb") as id_file:
        face_client.detect_with_stream(id_file)
        face_ids = [face.face_id for face in face_images]

    identity_result = face_client.identity(face_ids, person_group_id)
    for person in identity_result:
        if len(person.candidates) > 0 and person.candidates[0].confidence >= 0.65:
            person_id_validation = False
    
    print("person_id_validation", person_id_validation)     

    manifest.loc[manifest_index, 'NameValidation'] = name_validation
    manifest.loc[manifest_index, 'DoBValidation'] = dob_validation
    manifest.loc[manifest_index, 'PersonValidation'] = person_id_validation
    manifest.loc[manifest_index, 'BoardingPassValidation'] = boarding_pass_validation
    manifest.loc[manifest_index, 'LuggageValidation'] = luggage_validation

    # Save manifest
    manifest.to_csv('flight_manifest_trongld1.csv')
    # combine date from date and time
    flight_time = datetime.combine(boarding_data['date'], boarding_data['time'].time())
    # all validation passed
    if (name_validation and dob_validation and boarding_pass_validation and person_id_validation and luggage_validation):
        print(f"""
        Dear {boarding_data['name']},
        You are welcome to flight # {boarding_data['flight_num']} leaving at {flight_time} from
        {boarding_data['flight_num']} to {boarding_data['to']}.
        Your seat number is {boarding_data['seat']}, and it is confirmed.
        We did not find a prohibited item (lighter) in your carry-on baggage, thanks for following the procedure.
        Your identity is verified so please board the plane.
        """)
    
    elif (name_validation and dob_validation and boarding_pass_validation and person_id_validation and not luggage_validation):
        print(f"""
        Dear {boarding_data['name']},
        You are welcome to flight # {boarding_data['flight_num']} leaving at {flight_time} from
        {boarding_data['flight_num']} to {boarding_data['to']}.
        Your seat number is {boarding_data['seat']}, and it is confirmed.
        We have found a prohibited item in your carry-on baggage, and it is flagged for removal. 

        Your identity is verified. However, your baggage verification failed, so please see a customer service representative.
        """)
    
    else:
        print("""
        Dear Sir/Madam,
        Some of the information on your ID card does not match the flight manifest data, so you cannot board the plane.
        Please see a customer service representative.
        """)
    
    return manifest


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

def is_validate_baggege(path, file_name):
    with open(
        f"{path}/{file_name}.jpg", "rb"
    ) as image_contents:
        results = predictor.detect_image(
            project_id, publish_iteration_name, image_contents.read()
        )

    for prediction in results.predictions:
        # if prediction.probability > 0.5 then add to invalid_baggege and break
        if prediction.probability > 0.9:
           return False
        break
    return True

passengers = ['avkash-chauhan', 'james-jackson', 'james-webb', 'libby-herold', 'radha-s-kumar', 'sameer-kumar']
# passengers = ['james-jackson']

for p in passengers:
    validate_passenger(manifest, id_path + "ca-dl-" + p + ".png", boarding_path + "boarding-" + p + ".pdf", p)