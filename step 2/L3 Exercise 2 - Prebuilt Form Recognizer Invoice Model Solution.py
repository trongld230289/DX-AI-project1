#!/usr/bin/env python
# coding: utf-8

# ### Please install the required Python modules/SDKs

# In[16]:
#PRACTICE ON LAB then copy the code here

from sysconfig import get_python_version


get_python_version().system(' activate ai-azure-c1')

import sys

sys.path.append("/opt/conda/envs/ai-azure-c1/lib/python3.8/site-packages")


# ## Importing Azure Form Recognizer python modules

# In[17]:


from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import FormRecognizerClient


# In[18]:


AZURE_FORM_RECOGNIZER_ENDPOINT = "https://dx-doc-intelligence.cognitiveservices.azure.com/"
AZURE_FORM_RECOGNIZER_KEY = ""


# In[19]:


endpoint = AZURE_FORM_RECOGNIZER_ENDPOINT
key = AZURE_FORM_RECOGNIZER_KEY


# In[20]:


form_recognizer_client = FormRecognizerClient(endpoint=endpoint, credential=AzureKeyCredential(key))


# ## Source Document

# In[21]:


content_url =  "https://achauuatstorage.blob.core.windows.net/id-card/ca-dl-avkash-chauhan.png"


# In[22]:


invoices_from_url = form_recognizer_client.begin_recognize_identity_documents_from_url(content_url)


# ### Optional Step: Use the following if your source document is located the local disk
# form_recognizer_client.begin_recognize_invoices()

# In[23]:


collected_invoices = invoices_from_url.result()


# In[24]:


collected_invoices


# In[25]:


def get_id_card_details(identity_card):
    first_name = identity_card.fields.get("FirstName")
    if first_name:
        print("First Name: {} has confidence: {}".format(first_name.value, first_name.confidence))
    last_name = identity_card.fields.get("LastName")
    if last_name:
        print("Last Name: {} has confidence: {}".format(last_name.value, last_name.confidence))
    document_number = identity_card.fields.get("DocumentNumber")
    if document_number:
        print("Document Number: {} has confidence: {}".format(document_number.value, document_number.confidence))
    dob = identity_card.fields.get("DateOfBirth")
    if dob:
        print("Date of Birth: {} has confidence: {}".format(dob.value, dob.confidence))
    doe = identity_card.fields.get("DateOfExpiration")
    if doe:
        print("Date of Expiration: {} has confidence: {}".format(doe.value, doe.confidence))
    sex = identity_card.fields.get("Sex")
    if sex:
        print("Sex: {} has confidence: {}".format(sex.value, sex.confidence))
    address = identity_card.fields.get("Address")
    if address:
        print("Address: {} has confidence: {}".format(address.value, address.confidence))
    country_region = identity_card.fields.get("CountryRegion")
    if country_region:
        print("Country/Region: {} has confidence: {}".format(country_region.value, country_region.confidence))
    region = identity_card.fields.get("Region")
    if region:
        print("Region: {} has confidence: {}".format(region.value, region.confidence))


# In[34]:


print(content_url)
get_id_card_details(collected_invoices[0])




