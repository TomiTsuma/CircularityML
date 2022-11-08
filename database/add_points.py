from bson.objectid import ObjectId

from database.connection import get_database
dbname = get_database()
collection_name = dbname["users"]

def addPoints(user_id, points):
    document = collection_name.find_one({ "_id": ObjectId(user_id) })
    print(document)
    usr_points = document['points']
    usr_points = points + usr_points

    collection_name.update_one({ "_id": ObjectId(user_id) }, { "$set": { "points": usr_points } })

