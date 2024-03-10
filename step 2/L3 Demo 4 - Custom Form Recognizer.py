#!/usr/bin/env python
# coding: utf-8

# ### Please install the required Python modules/SDKs

# In[1]:

#PRACTICE ON LAB then copy the code here

from sysconfig import get_python_version


get_python_version().system(' activate ai-azure-c1')

import sys

sys.path.append("/opt/conda/envs/ai-azure-c1/lib/python3.8/site-packages")


# ## Importing Azure Form Recognizer Python modules

# In[2]:


import os
from azure.core.exceptions import ResourceNotFoundError
from azure.ai.formrecognizer import FormRecognizerClient
from azure.ai.formrecognizer import FormTrainingClient
from azure.core.credentials import AzureKeyCredential


# In[3]:


AZURE_FORM_RECOGNIZER_ENDPOINT = "https://dx-doc-intelligence.cognitiveservices.azure.com/"
AZURE_FORM_RECOGNIZER_KEY = ""


# In[4]:


endpoint = AZURE_FORM_RECOGNIZER_ENDPOINT
key = AZURE_FORM_RECOGNIZER_KEY


# In[5]:


form_training_client = FormTrainingClient(endpoint=endpoint, credential=AzureKeyCredential(key))


# In[6]:


saved_model_list = form_training_client.list_custom_models()


# ## Training Source Data URL
# 
# To generate the training data URL, you can 
# 1. Download the Cognito Corporation training documents to your local system: https://github.com/udacity/cd0461-building-computer-vision-solutions-with-azure-exercises/tree/main/resources/cognito-corp-docs.
# 2. Upload the training documents to a blob container at Azure Blob Storage. Training documents are named `Cognito-corporation-u*.pdf`. 
# 3. Generate a SAS URL of the training data container. 
# 4. Once the model is trained, you will use the `Cognito-corporation-test01.png` file located in this GitHub directory to perform prediction: https://raw.githubusercontent.com/udacity/cd0461-building-computer-vision-solutions-with-azure-exercises/main/resources/Cognito-corporation-test01.png

# In[7]:


trainingDataUrl = "https://achauuatstorage.blob.core.windows.net/boarding?sp=rwdl&st=2024-03-07T14:40:13Z&se=2025-03-07T22:40:13Z&spr=https&sv=2022-11-02&sr=c&sig=CrFqiTTUFE%2FaplNZ9zSp%2BiLNn4lMSPwoTdo6B9cmb0Y%3D"


# ## Performing Unlabeled Traning
# ### use_training_labels=False

# In[8]:


training_process = form_training_client.begin_training(trainingDataUrl, use_training_labels=False)
custom_model = training_process.result()


# ## Getting Model Info

# In[9]:


custom_model


# In[10]:


custom_model.model_id


# In[11]:


custom_model.status


# In[12]:


custom_model.training_started_on


# In[13]:


custom_model.training_completed_on


# In[14]:


custom_model.training_documents


# In[15]:


for doc in custom_model.training_documents:
    print("Document name: {}".format(doc.name))
    print("Document status: {}".format(doc.status))
    print("Document page count: {}".format(doc.page_count))
    print("Document errors: {}".format(doc.errors))


# In[16]:


custom_model.properties


# In[17]:


custom_model.submodels


# In[18]:


for submodel in custom_model.submodels:
    print(
        "The submodel with form type '{}' has recognized the following fields: {}".format(
            submodel.form_type,
            ", ".join(
                [
                    field.label if field.label else name
                    for name, field in submodel.fields.items()
                ]
            ),
        )
    )


# In[19]:


custom_model.model_id


# In[20]:


custom_model_info = form_training_client.get_custom_model(model_id=custom_model.model_id)
print("Model ID: {}".format(custom_model_info.model_id))
print("Status: {}".format(custom_model_info.status))
print("Training started on: {}".format(custom_model_info.training_started_on))
print("Training completed on: {}".format(custom_model_info.training_completed_on))


# ## Using an image document as test document URL (Not using PDF here)
# 
# * Here, you will use the `Cognito-corporation-test01.png` file located in this GitHub directory to perform prediction: https://raw.githubusercontent.com/udacity/cd0461-building-computer-vision-solutions-with-azure-exercises/main/resources/Cognito-corporation-test01.png
# * Note: If you want to use a PDF document for the test, please save and upload PDF to Azure Blob Storage and use the SAS URL of this PDF document as the target URL.
# * Using a PDF document from the GitHub URL will give you an error.
# * You will see a screenshot of how to do this on the exercise solution page later in this lesson. 

# In[21]:


new_test_url = "https://achauuatstorage.blob.core.windows.net/boarding/boarding-avkash.pdf"


# In[22]:


new_test_url


# In[23]:


form_recognizer_client = FormRecognizerClient(endpoint=endpoint, credential=AzureKeyCredential(key))


# In[24]:


custom_model.model_id


# In[25]:


custom_model_info.model_id


# In[26]:


custom_test_action = form_recognizer_client.begin_recognize_custom_forms_from_url(model_id=custom_model_info.model_id, form_url=new_test_url)


# In[29]:


custom_test_action.status()


# In[30]:


custom_test_action_result = custom_test_action.result()


# In[31]:


for recognized_content in custom_test_action_result:
    print("Form type: {}".format(recognized_content.form_type))
    for name, field in recognized_content.fields.items():
        print("Field '{}' has label '{}' with value '{}' and a confidence score of {}".format(
            name,
            field.label_data.text if field.label_data else name,
            field.value,
            field.confidence
        ))


# ### As you can see above, the confidence is very low with string fields, so we will want to add training labels to improve the confidence scores.
# 
# # ===========PAUSE HERE==============
# 
# ## At this point, you should go to the Form Recognizer portal and label your training documents manually there.
# ## Please read the following instructions:
# 1. If you haven't labeled the training documents from the portal demo, you should now visit the Form Recognizer portal and create a new project (https://fott-2-1.azurewebsites.net/projects/) using the same blob container where you have stored the Cognito Corp training documents.
# 2. When you read the training files in the blob container from the Form Recognizer portal, a master `project_name.fott` file will be auto-generated in your blob container. When you add tags, a `fields.json` file is auto-generated in your blob container.
# 3. When you run layout on a training document, an `ocr.json` file gets auto-generated in your blob container. When you label the fields at the Form Recognizer portal, a `labels.json` file is auto-generated in your blob container. These files are essential for a labeled training to work. **If you don't have those documents,  you will get the error: <br>"Can't find any OCR files for training." or "Can't find any label files for training."**
# 5. Label **at least 5** (if not all) of the training documents at the Form Recognizer portal. This will auto-generate the `labels.json` documents in the blob container. If you saved label documents from the previous demo, you can also upload your own `labels.json` documents into the blob container so that you don't have to label the training documents again. 
# 6. Please go back to the portal demo pages if you need help with these steps. 

# ### use_training_labels=True

# In[32]:


labeled_training_process = form_training_client.begin_training(trainingDataUrl, use_training_labels=True)
labeled_custom_model = labeled_training_process.result()


# In[33]:


labeled_custom_model.model_id


# In[34]:


labeled_custom_model.status


# In[35]:


labeled_custom_model.training_documents


# In[36]:


for doc in labeled_custom_model.training_documents:
    print("Document name: {}".format(doc.name))
    print("Document status: {}".format(doc.status))
    print("Document page count: {}".format(doc.page_count))
    print("Document errors: {}".format(doc.errors))


# In[37]:


labeled_custom_model.model_id


# In[38]:


labeled_custom_test_action = form_recognizer_client.begin_recognize_custom_forms_from_url(model_id=labeled_custom_model.model_id, form_url=new_test_url)


# In[39]:


labeled_custom_test_action.status()


# In[40]:


labeled_custom_test_action_result = labeled_custom_test_action.result()


# In[41]:


for recognized_content in labeled_custom_test_action_result:
    print("Form type: {}".format(recognized_content.form_type))
    for name, field in recognized_content.fields.items():
        print("Field '{}' has label '{}' with value '{}' and a confidence score of {}".format(
            name,
            field.label_data.text if field.label_data else name,
            field.value,
            field.confidence
        ))


# ## As you can see above, the confidence for string fields is very high, so a labeled training is better.

# ## Listing Models

# In[ ]:


saved_model_list = form_training_client.list_custom_models()


# In[ ]:


for model in saved_model_list:
    print(model.model_id)


# # The videos on composed model can be found on the next page

# ## Creating Composed Model
# 
# ### All models in composed models list must be created from the labeled training process.

# In[ ]:


## Cognito corporation model with labeled training (First)
labeled_custom_model.model_id


# In[ ]:


## Creating another model with labeled training
labeled_2_training_process = form_training_client.begin_training(trainingDataUrl, use_training_labels=True)
labeled_2_custom_model = labeled_2_training_process.result()


# In[ ]:


## Cognito corporation model with labeled training (Second)
labeled_2_custom_model.model_id


# In[ ]:


cognito_corporation_model_list = [labeled_custom_model.model_id, labeled_2_custom_model.model_id]


# In[ ]:


composed_process = form_training_client.begin_create_composed_model(
            cognito_corporation_model_list, model_name="Cognito Corporation Model")
composed_process_model = composed_process.result()


# In[ ]:


composed_process_model.model_id


# In[ ]:


composed_model_info = form_training_client.get_custom_model(model_id=composed_process_model.model_id)
print("Model ID: {}".format(composed_model_info.model_id))
print("Status: {}".format(composed_model_info.status))
print("Training started on: {}".format(composed_model_info.training_started_on))
print("Training completed on: {}".format(composed_model_info.training_completed_on))


# In[ ]:


# Is this composed model
composed_model_info.properties


# ### Using composed model to extract text

# In[ ]:


composed_model_testing = form_recognizer_client.begin_recognize_custom_forms_from_url(model_id=composed_process_model.model_id, form_url=new_test_url)


# In[ ]:


composed_model_testing.status()


# In[ ]:


composed_model_testing_result = composed_model_testing.result()


# In[ ]:


for recognized_content in composed_model_testing_result:
    print("Form type: {}".format(recognized_content.form_type))
    for name, field in recognized_content.fields.items():
        print("Field '{}' has label '{}' with value '{}' and a confidence score of {}".format(
            name,
            field.label_data.text if field.label_data else name,
            field.value,
            field.confidence
        ))


# ## Resources 
# - https://docs.microsoft.com/en-us/samples/azure/azure-sdk-for-python/formrecognizer-samples/
