""" recognize faces in loaded images using the profiles images ONE FACE PER IMAGE """

import cv2
import numpy as np
import face_recognition
import os

# profile images variables
path_profilesimg = 'app/Images/ProfilesImg'
images = []
classnames = []

# loaded images variables
path_loadedimg = 'app/Images/LoadedImg'
loadedimages = []

# function to find the encodings
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# function to find the matches between the loaded images and the profiles images
def findMatches(encodeListKnown, encodedListLoaded):
    matches = []
    for encodeLoaded in encodedListLoaded:
        match = face_recognition.compare_faces(encodeListKnown, encodeLoaded)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeLoaded)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if match[matchIndex]:
            name = classnames[matchIndex].upper()
            # print(name)
            matches.append(name)
        else:
            matches.append('Unknown')
    return matches

# Get all the profiles images
profilelist = os.listdir(path_profilesimg)

for cl in profilelist:
    if not cl.startswith('.'):
        curImg = cv2.imread(f'{path_profilesimg}/{cl}')
        images.append(curImg)
        classnames.append(os.path.splitext(cl)[0])

encodeListKnown = findEncodings(images)

# Get all the loaded images
loadedlist = os.listdir(path_loadedimg)

for cl in loadedlist:
    if not cl.startswith('.'):
        curImg = cv2.imread(f'{path_loadedimg}/{cl}')
        loadedimages.append(curImg)

encodedListLoaded = findEncodings(loadedimages)

matches = findMatches(encodeListKnown, encodedListLoaded)

# print the results
print(matches)
