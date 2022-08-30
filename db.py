import os
from dotenv import load_dotenv
load_dotenv()

from datetime import datetime
START_TIME = datetime.utcnow()

import pymongo

CONNECTION_STR = os.getenv("MONGODB_URI")

client = pymongo.MongoClient(CONNECTION_STR)
print("Connected to database!")
db = client.serverlogs
logs_collection = db.logs

def output_to_db(pedestrian_count, cameras):
    END_TIME = datetime.utcnow()
    
    entries = []
    for index in range(len(pedestrian_count)):
        count = pedestrian_count[index]
        camera = cameras[index]
        entries.append({
            "url" : camera["url"],
            "count" : count,
        })

    submit_dict = {
        "counts" : entries,
        "now" : str(END_TIME),
        "start" : str(START_TIME)
    }

    logs_collection.insert_one(submit_dict)
