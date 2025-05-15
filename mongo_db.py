"""
MongoDB bson Utility Module

This module provides functionality for connecting to Mongo-DB, retrieving documents,
and converting them to json format compatible with the existing pipeline. It includes
functions for establishing database connections, yielding job documents from collections,
and transforming Mongo-DB bson documents into standardized json structures.

The module supports various job board formats with special handling for portal-specific
fields (indeed or stepstone) and ObjectId conversion. It also includes a test function to verify connection
and document conversion functionality.
"""

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from bson.json_util import dumps, loads
from dotenv import load_dotenv
import json
import os

load_dotenv(override=True)  # Force reload and override existing env vars
uri = os.environ.get("MONGODB_URI")  # Use os.environ.get instead of os.getenv
#uri = "mongodb+srv://Tom:pwc123@scrapedata.4qouh.mongodb.net/?retryWrites=true&w=majority&appName=ScrapeData"

# Add fallback and debug output
if not uri or len(uri) < 20:  # Basic validation
    print(f"Warning: Mongo-DB URI appears incomplete: {uri}")
    # Fallback to hardcoded URI only during development
    uri = "your_connection_string"  # Only use during development

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Add database and collection access
db = client['Test_Docker']
#jobs_collection = db['stepstone_marketing_manageraccounting_manager']  

def get_mongo_jobs(database_name=None):
    """
    Generator to yield jobs from Mongo-DB with collection name, converted to compatible json format
    
    :param collection_name: Name of the Mongo-DB collection to query, defaults to None
    :type collection_name: str, optional
    :yield: Tuple containing the converted job document and collection name
    :rtype: tuple(dict, str)
    """

    if database_name:
        db = client[database_name]
    else:
        db = client['Test_Docker']  # Default database

    # Get all collection names in the database
    collection_names = db.list_collection_names()
    
    for collection_name in collection_names:
        collection = db[collection_name]
        for doc in collection.find({}):
            # Convert Mongo-DB document to compatible json structure
            compatible_job = convert_mongo_to_compatible_json(doc)
            yield compatible_job, collection_name


def convert_mongo_to_compatible_json(mongo_doc):
    """
    Convert Mongo-DB document to a json structure compatible with the existing pipeline
    
    :param mongo_doc: Mongo-DB document to convert
    :type mongo_doc: dict
    :return: Compatible json structure for the pipeline
    :rtype: dict
    """
    # Create a compatible structure by directly mapping fields
    compatible_job = {
        "url": mongo_doc.get("url", "") or mongo_doc.get("URL", ""),
        "paragraphs": mongo_doc.get("paragraphs", []),
        "lists": mongo_doc.get("lists", {})
    }
    
    # Preserve Indeed-specific fields
    if "jobLocationText" in mongo_doc:
        compatible_job["jobLocationText"] = mongo_doc["jobLocationText"]
    
    if "benefits" in mongo_doc:
        compatible_job["benefits"] = mongo_doc["benefits"]
    
    if "paragraphs" in mongo_doc:
        compatible_job["paragraphs"] = mongo_doc["paragraphs"]

    if "datePosted" in mongo_doc:
        compatible_job["datePosted"] = mongo_doc["datePosted"]

    if "Company Name" in mongo_doc:
        compatible_job["Company Name"] = mongo_doc["Company Name"]
    
    # Add title if available
    if "Job Title" in mongo_doc and mongo_doc["Job Title"]:
        compatible_job["Job Title"] = mongo_doc["Job Title"]
    elif "jobId" in mongo_doc:
        compatible_job["title"] = f"Job ID: {mongo_doc['jobId']}"
    elif "Job #" in mongo_doc:
        compatible_job["title"] = f"Job #: {mongo_doc['Job #']}"
    
    # Convert ObjectId to string if needed
    if "_id" in mongo_doc and hasattr(mongo_doc["_id"], "__str__"):
        compatible_job["_id"] = str(mongo_doc["_id"])
    
    return compatible_job


def test_conversion():
    """
    Test the Mongo-DB document conversion and actual connection.
    Prints out the first job in the database converted into json.
    """
    # Get one document
    doc = next(get_mongo_jobs())[0]
    
    # Print the converted document
    print("Converted document structure:")
    print(json.dumps(doc, indent=2))
    
    # Check if the structure is compatible
    required_fields = ["url", "paragraphs", "lists"]
    missing_fields = [field for field in required_fields if field not in doc]
    
    if missing_fields:
        print(f"Warning: Missing required fields: {missing_fields}")
    else:
        print("Conversion successful! Document has all required fields.")

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to Mongo-DB!")
except Exception as e:
    print(e)

# Run the test if this file is executed directly
if __name__ == "__main__":
    test_conversion()
