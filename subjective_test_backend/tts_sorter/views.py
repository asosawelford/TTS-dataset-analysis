from django.shortcuts import render
import json
from django.http import HttpResponse
import csv
import pickle
from config import base_dir
from utilities import shuffle_filepaths
import random
import os


# Create your views here.

def say_hello(request): 
    return HttpResponse('hello world')

def receive_form(request):
    """
    Receives form data from frontend and appends it to results.csv.
    """
    body_unicode = request.body.decode('utf-8')
    body = json.loads(body_unicode)

    # parse form data
    age = body['age']
    sex = body['sex']
    if body['country'] != '' :
        country = body['country']
    else:
        country = 'NA'
    if body['province'] != '' :
        province = body['province']
    else:
        province = 'NA'
    familiaridad = body['familiaridad']
    auriculares = body['auris']
    userID = body.get('userID', '')

    print(userID, age, sex, country, province, familiaridad, auriculares)
    json.dumps({})
    
        # Open the CSV file in append mode
    with open('participants.csv', 'a', newline='') as file:
        writer = csv.writer(file)

        # Write the variables to the CSV file
        writer.writerow([userID, age, sex, country, province, familiaridad, auriculares])
    
    
    return HttpResponse( json.dumps({}), status=200)

def load_audios(request):
    new_paths = {}

   #iterate over each category of stimulus
    for letter in "ABCDE":
        #open the pickle file with the filepaths
        path = os.path.join(base_dir, letter + '.pickle')
        with open(path, 'rb') as handle:
            filepaths = pickle.load(handle)
        try:
            new_paths[letter] = filepaths.pop() # Take the last element from the list
            # save filepaths as pickle file
            path = os.path.join(base_dir, letter + '.pickle')
            with open(path, 'wb') as handle:
                pickle.dump(filepaths, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except IndexError:
            shuffle_filepaths(letter=letter)
            path = os.path.join(base_dir, letter + '.pickle')
            with open(path, 'rb') as handle:
                filepaths = pickle.load(handle)
            new_paths[letter] = filepaths.pop() # Take the last element from the list
            # save filepaths as pickle file
            path = os.path.join(base_dir, letter + '.pickle')
            with open(path, 'wb') as handle:
                pickle.dump(filepaths, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(new_paths)
    values = list(new_paths.values())
    random.shuffle(values)
    shuffled_dict = {k: v for k, v in zip(new_paths.keys(), values)}
    print(shuffled_dict)

    return HttpResponse(json.dumps(shuffled_dict), status=200)
    
def receive_rate(request):
    """
    Receives stimulus rate data from frontend and appends it to results.csv.
    """
    body_unicode = request.body.decode('utf-8')
    body = json.loads(body_unicode)
    rate1fp = body.get('rate1fp', '')[30:]
    rate2fp = body.get('rate2fp', '')[30:]
    rate3fp = body.get('rate3fp', '')[30:]
    rate4fp = body.get('rate4fp', '')[30:]
    rate5fp = body.get('rate5fp', '')[30:]
    rate1 = body.get('rate1', '')
    rate2 = body.get('rate2', '')
    rate3 = body.get('rate3', '')
    rate4 = body.get('rate4', '')
    rate5 = body.get('rate5', '')
    userID = body.get('userID', '')
    print(rate1)

    json.dumps({})
    
        # Open the CSV file in append mode
    with open('results.csv', 'a', newline='') as file:
        writer = csv.writer(file)

        # Write the variables to the CSV file
        writer.writerow([userID, rate1fp, rate1])
        writer.writerow([userID, rate2fp, rate2])
        writer.writerow([userID, rate3fp, rate3])
        writer.writerow([userID, rate4fp, rate4])
        writer.writerow([userID, rate5fp, rate5])
    
    return HttpResponse( json.dumps({}), status=200)

def receive_email(request):
    """
    Receives stimulus rate data from frontend and appends it to results.csv.
    """
    body_unicode = request.body.decode('utf-8')
    body = json.loads(body_unicode)
    userID = body.get('userID', '')
    email = body.get('email', '')

    with open('emails.csv', 'a', newline='') as file:
        writer = csv.writer(file)

        # Write the variables to the CSV file
        writer.writerow([userID, email])
        
    return HttpResponse( json.dumps({}), status=200)

