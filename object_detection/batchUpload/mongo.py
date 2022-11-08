#/usr/bin/python
# from email_api import SendMail
import mariadb,pymongo,os,sys,pprint,time
# declare the location of the remote database

target_server = "mongodb+srv://codem:1100111@virdismart.nx3p6.mongodb.net/Virdsmart?retryWrites=true&w=majority"

local_username = "user1"
local_password = "virdis"
local_dbname = "Virdis"
host = "127.0.0.1"
port = 3306

mailing_content = "Logs: "


try:
    conn = mariadb.connect(
        user = local_username,
        password = local_password,
        host = host,
        port = port,
        database = local_dbname
    )
except mariadb.Error as e:
    mailing_content+= "Error connecting to local database\n Exiting."
    sys.exit(1)

# else:
curr = conn.cursor()


# connect online
client = pymongo.MongoClient(target_server)

db = client['Virdismart']

collection_user = db['users']
collection_deposit = db['Deposit']

def parseData(dataBlock,binKeys):
    zipped = dict()
    bunch = list()

    for singleton in dataBlock:
        bunch.append(dict(zip(binKeys,singleton)))

    return bunch


def updateLocalDB():
    need_update = list(collection_user.find({"downloaded":False},{"qrId","utype","CardId"}))
    
    try:
        collection_user.update_many({"downloaded":False}, {"$set":{"downloaded":True}})
    except Exception as e:
        print(e)
        mailing_content = "Error updating remote user table"

    for singleRecord in need_update:
        try:
            query = "INSERT INTO userC (qrId,utype,cardId) VALUES (?,?,?)"
            curr.execute(query,(singleRecord['qrId'],singleRecord['utype'],singleRecord['CardId']))
            conn.commit()
        except Exception as e:
            mailing_content = ", Could not insert record "+ str(singleRecord)
            print(e)
    print("Done")


def SyncData(collection_name , dataset):
    pprint.pprint(dataset)

    try:
        
        collection_name.insert_many(dataset)

    except Exception as e:
        print("Error updating the online store")
        mailing_content=",\nLogs: "+ "table "+ collection_name + " threw an error"
    else:
        query = "UPDATE Deposit set uploaded=? where uploaded=?"
        curr.execute(
            query,(1,0)
            )
        conn.commit()
        print("Sync successfull")
    finally:
        print("exiting")

def uploadData():
    updateLocalDB()
    
    query = "SELECT binId,userId,massThrown,sizeThrown,fillA,fillB,utype,log,uploaded FROM Deposit WHERE uploaded=?"

    try:
        curr.execute(
        query,(0,)
    )
    except Exception as e:
        mailing_content ="\n "+ "Error executint query " + query

    else:
        pending_deposit_upload_list = curr.fetchall()

        if len(pending_deposit_upload_list)>0:
            columns = ("binId","userId","massThrown","sizeThrown","fillA","fillB","utype","log","uploaded")
            parsed = parseData(pending_deposit_upload_list,columns)
            SyncData(collection_deposit,parsed)

    #SendMail.send_email("cirenenum@gmail.com","gpJ513672","cirenenum@gmail.com","Smart Bin Logs",mailing_content)
    print("Database updated online")
