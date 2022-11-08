#/usr/bin/bash

ping google.com -c 3;
if [ $? -eq 0 ]
then
    #pip install dnspython
    sudo python /home/smart_bin/object_detection/updated.py;
fi
