from user.models import user_profile
from django.shortcuts import render, redirect
import cv2
from django.core.mail import EmailMultiAlternatives
from PIL import Image
import numpy as np
from django.views.decorators.csrf import csrf_exempt
import time
from datetime import datetime
import pythoncom
from win32com.client import constants, Dispatch
from translate import Translator
import random
from gtts import gTTS
from playsound import playsound
import joblib

# from sklearn.externals import joblib
# filename = "Z:\BTP\knk\static\gest2aud\HOG_full_newaug.sav"
import keras

import pickle
import h5py

from keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf
import os
import json
from django.http import HttpResponse
from keras.models import load_model
import skimage
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import numpy as np
import cv2
import keras
import tensorflow as tf
from string import ascii_uppercase
from gingerit.gingerit import GingerIt


parser = GingerIt()


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
    except RuntimeError as e:
        print(e)


model = keras.models.load_model(
    "C:/project/test/signlanguage/Indian-Sign-Language-Gesture-Recognition-master/gest2aud/model-all1-alpha.h5")


alpha_dict = {}
j = 0
for i in ascii_uppercase:
    alpha_dict[j] = i
    j = j + 1
    if j == 14:
        alpha_dict[j] = "None"
        j = j + 1


def test_image(image):
    pred = model.predict(image)
    letter = alpha_dict[np.argmax(pred)]
    if letter == "None":
        return " "
    return letter



word1 = ""

def convert(gestures):
    word = ""
    print("Convert multiple is called")

    for image in gestures:
        print("image is called")
        print(
            "---------------------------------------------------------------------Next gesture-----------------------------------------------------------")
        temp_word = test_image(image)
        print("word: " + word)
        word += temp_word
    
    word = word.lower()
    print(word)
    final_word = parser.parse(word)['result']
    return final_word



def convert_single(gestures):
    global word1
    print("Cnvert single is called")

    for image in gestures:
        print("image is called")
        print(
            "---------------------------------------------------------------------Next gesture-----------------------------------------------------------")
        temp_word = test_image(image)
        print("word: " + word1)
        if temp_word == " ":
            speaker = Dispatch("SAPI.SpVoice")  # Create SAPI SpVoice Object
            speaker.Speak(word1)  # Process TTS
            del speaker
            word1 = ""
        word1 += temp_word

    


@csrf_exempt
def take_snaps(request):
    if request.user.is_authenticated:
        cam = cv2.VideoCapture(0)
        # cv2.namedWindow("Record Hand Gestures")
        img_counter = 0
        gestures = []  # list to store images
        x1 = datetime.now()
        initial = 0
        while True:
            x2 = datetime.now()
            ret, frame = cam.read()
            frame = cv2.flip(frame, 1)
            cv2.rectangle(frame, (319, 9), (620 + 1, 309),
                          (0, 255, 0), 1)  # changed
            # cv2.imshow("Record Hand Gestures", frame)
            frame_crop = frame[10:300, 320:620]
                    #
            gray = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY)
            gaussblur = cv2.GaussianBlur(gray, (5, 5), 2)
            smallthres = cv2.adaptiveThreshold(gaussblur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                       cv2.THRESH_BINARY_INV, 9, 2.8)
            ret1, temp_image = cv2.threshold(
                        smallthres, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            temp_image = cv2.resize(temp_image, (128, 128))

            temp_image = np.reshape(
                        temp_image, (1, temp_image.shape[0], temp_image.shape[1], 1))
            pred = model.predict(temp_image)
            cv2.putText(frame,alpha_dict[np.argmax(pred)], (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            cv2.imshow("Frame", frame)
            

            if not ret:
                break

            if (x2 - x1).seconds >= 8:

                x1 = x2

                initial += 1
                # print(str(initial)+"  xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                if initial > 1:
                    frame_crop = frame[10:300, 320:620]
                    #
                    gray = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY)
                    gaussblur = cv2.GaussianBlur(gray, (5, 5), 2)
                    smallthres = cv2.adaptiveThreshold(gaussblur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                       cv2.THRESH_BINARY_INV, 9, 2.8)
                    ret1, final_image = cv2.threshold(
                        smallthres, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    cv2.imshow("BW", final_image)
                    final_image = cv2.resize(final_image, (128, 128))

                    final_image = np.reshape(
                        final_image, (1, final_image.shape[0], final_image.shape[1], 1))
                    #
                    gestures.append(final_image)
                    len(gestures)
                    cv2.putText(frame,"snapped", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 5, (0, 255, 0), 2)
                    cv2.imshow("Frame", frame)
                    print("snapped " + str(img_counter))
                    img_counter += 1
                    # convert_single([final_image])
                    

            k = cv2.waitKey(1)

            if k == 27:
                print("Escape hit, closing...")
                break

        cam.release()
        cv2.destroyAllWindows()

        print(img_counter)
        print("Number of images cptured -> ", len(gestures))
        max_word = convert(gestures)
        print(max_word)

        # Code to convert text into speech
        speaker = Dispatch("SAPI.SpVoice")  # Create SAPI SpVoice Object
        speaker.Speak(max_word)  # Process TTS
        del speaker

        data = {}
        data['max_word'] = max_word
        json_data = json.dumps(data)

        return HttpResponse(json_data, content_type="application/json")
    else:
        return redirect('../login')


def gest_keyboard(request):
    if request.user.is_authenticated:
        context = {}
        if request.method == "POST":
            print(request.POST['gest_text'])
            gest_text = request.POST['gest_text']
            pythoncom.CoInitialize()

            speaker = Dispatch("SAPI.SpVoice")  # Create SAPI SpVoice Object
            speaker.Speak(gest_text)  # Process TTS
            del speaker

            context = {'gest_text': gest_text}
            print("ddd")
        return render(request, 'gest2aud/gest_keyboard.html', context)
    else:
        return redirect('../login')


def emergency(request):
    if (request.method == "POST"):
        print(request.POST)
        # print(request.user)
        print(user_profile.objects.get(user=request.user))
        usr = user_profile.objects.get(user=request.user)
        mail_text = []
        # print(request.POST['csrfmiddlewaretoken'])
        for i in request.POST:
            if (i != "csrfmiddlewaretoken"):
                mail_text.append(request.POST[i])
        print(mail_text)

        EMAIL = []
        EMAIL.append(usr.Email1)
        EMAIL.append(usr.Email2)
        EMAIL.append(usr.Email3)
        EMAIL.append(usr.Email4)
        EMAIL.append(usr.Email5)
        print(EMAIL, "-------------------------------------------------")
        for i in EMAIL:
            subject, from_email, to = "Emergency Message", "tempuxyz@gmail.com", i
            text_content = "This is an emergnecy message from your deaf friend"
            text_content += '\n'
            for i in mail_text:
                text_content += i
                text_content += '\n'

            msg = EmailMultiAlternatives(
                subject, text_content, from_email, [to])
            msg.send()

    return render(request, 'gest2aud/Emergency.html', {})

@csrf_exempt
def language_convert(request):
    if request.user.is_authenticated:
        print("were hereeeeeeeeee babayyyy")
        body_unicode = request.body.decode('utf-8') 	
        body1 = json.loads(body_unicode)
        print(body1)
        
        print(body1["lngCode"])
        translator= Translator(to_lang=body1["lngCode"])
        translation = translator.translate(body1["text"])
        print(translation)

        myobj = gTTS(text=translation, lang=body1["lngCode"], slow=False)

        # Saving the converted audio in a mp3 file named
        # welcome
        myobj.save("C:/project/test/signlanguage/Indian-Sign-Language-Gesture-Recognition-master/gest2aud/welcome.mp3")
        time.sleep(2)
        # Playing the converted file
        playsound("C:/project/test/signlanguage/Indian-Sign-Language-Gesture-Recognition-master/gest2aud/welcome.mp3")
        data = {}
        data['max_word'] = translation
        json_data = json.dumps(data)
        os.remove("C:/project/test/signlanguage/Indian-Sign-Language-Gesture-Recognition-master/gest2aud/welcome.mp3")
        return HttpResponse(json_data, content_type="application/json")
    else:
         return redirect('../login')


