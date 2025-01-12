# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 01:46:41 2018

@author: vivek
"""

import requests
import numpy as np
import time

pointraw = """25.2407032  83.0566977
25.3385583  82.97931
25.3612349  82.9966254
25.3291433  83.0008222
25.3244814  83.0069087
25.3241179  83.0010828
25.3303828  82.9920889
25.3385593  82.9802643
25.329282   82.974565
25.2568781  83.0053126
25.3264241  82.9914717
25.3195106  82.9844653
25.2626048  83.0040131
25.3166017  82.9611156
25.3276834  82.9521587
25.2969241  82.9838664
25.3066491  82.9412402
25.3053663  82.9234884
25.3220669  83.0005952
25.3061911  83.008097
25.3217636  82.9868155
25.3293404  82.9963367
25.3031856  82.9774482
25.301926   82.9661924
25.2907646  82.9594976
25.2877488  82.9398779
25.3000968  82.9987992
25.3021377  83.0024443
25.2980559  83.0051999
25.289192   83.004366
25.2836766  82.9725997
25.2935442  82.9826277
25.3177379  82.9733744
25.2834972  82.9732165
25.2551347  82.9710908
25.2686311  82.9704446
25.2819536  82.9827647
25.2940581  82.9752527
25.3032182  82.9535653
25.2860584  82.966456
25.2841929  82.966357
25.2611941  82.9496352
25.2930101  82.9897254
25.2603877  83.0170152
25.2752078  83.0090936
25.2811799  82.9954713
25.2930155  82.9965752
25.2932362  82.9897193
25.2888028  82.9968273
25.2835944  82.989237
25.2758349  82.9997182
25.3466467  83.0056218
25.3240558  83.0257186
25.3126682  83.0259879
25.3349108  82.9905595
25.3578064  82.9927483
25.3624631  82.9948911
25.3571538  82.9988397
25.3725336  83.0159708"""

GOOGLE_MAPS_API = "https://maps.googleapis.com/maps/api/distancematrix/json?"
DUMMY_RESPONSE = {'destination_addresses': ['Hamidpur, Uttar Pradesh, India',
  'Kashipuram Rd, Kashipuram Colony, Bhagwanpur, Varanasi, Uttar Pradesh 221011, India',
  'B12/973, Kakarmata, Varanasi, Uttar Pradesh 221010, India',
  'Sa3/186 AD, Daulatpur Rd, Vivekanand Nagar Colony, Sudama Nagar Colony, Akatha, Varanasi, Uttar Pradesh 221007, India',
  'J12/121, Dhupchandi, Jaitpura, Varanasi, Uttar Pradesh 221001, India',
  'K66/1-A, Kabirchaura Naibasti, Kabir Nagar Churaha, Kotwali, Varanasi, Uttar Pradesh 221001, India',
  'C 7/39 C, Ram Katora Rd, Rampuri Colony, Jaitpura, Varanasi, Uttar Pradesh 221001, India',
  'C28/96, Raja Bazar Rd, Maldahiya, Chetganj, Varanasi, Uttar Pradesh 221001, India',
  'Stranger Rd, Varanasi Cantt, Varanasi, Uttar Pradesh 221002, India',
  'Building number 64, Shop No.2A, 64, The Mall Rd, Varuna Bridge, Varanasi Cantt, Varanasi, Uttar Pradesh 221002, India'],
 'origin_addresses': ['Hamidpur, Uttar Pradesh, India'],
 'rows': [{'elements': [{'distance': {'text': '1 m', 'value': 0},
     'duration': {'text': '1 min', 'value': 0},
     'status': 'OK'},
    {'distance': {'text': '7.1 km', 'value': 7058},
     'duration': {'text': '22 mins', 'value': 1291},
     'status': 'OK'},
    {'distance': {'text': '12.6 km', 'value': 12561},
     'duration': {'text': '43 mins', 'value': 2590},
     'status': 'OK'},
    {'distance': {'text': '19.7 km', 'value': 19725},
     'duration': {'text': '55 mins', 'value': 3322},
     'status': 'OK'},
    {'distance': {'text': '17.2 km', 'value': 17227},
     'duration': {'text': '44 mins', 'value': 2661},
     'status': 'OK'},
    {'distance': {'text': '16.6 km', 'value': 16639},
     'duration': {'text': '46 mins', 'value': 2783},
     'status': 'OK'},
    {'distance': {'text': '17.9 km', 'value': 17893},
     'duration': {'text': '48 mins', 'value': 2856},
     'status': 'OK'},
    {'distance': {'text': '17.5 km', 'value': 17512},
     'duration': {'text': '44 mins', 'value': 2650},
     'status': 'OK'},
    {'distance': {'text': '19.0 km', 'value': 19028},
     'duration': {'text': '51 mins', 'value': 3031},
     'status': 'OK'},
    {'distance': {'text': '19.2 km', 'value': 19192},
     'duration': {'text': '51 mins', 'value': 3086},
     'status': 'OK'},
    {'distance': {'text': '1 m', 'value': 0},
     'duration': {'text': '1 min', 'value': 0},
     'status': 'OK'},
    {'distance': {'text': '7.1 km', 'value': 7058},
     'duration': {'text': '22 mins', 'value': 1291},
     'status': 'OK'},
    {'distance': {'text': '12.6 km', 'value': 12561},
     'duration': {'text': '43 mins', 'value': 2590},
     'status': 'OK'},
    {'distance': {'text': '19.7 km', 'value': 19725},
     'duration': {'text': '55 mins', 'value': 3322},
     'status': 'OK'},
    {'distance': {'text': '17.2 km', 'value': 17227},
     'duration': {'text': '44 mins', 'value': 2661},
     'status': 'OK'},
    {'distance': {'text': '16.6 km', 'value': 16639},
     'duration': {'text': '46 mins', 'value': 2783},
     'status': 'OK'},
    {'distance': {'text': '17.9 km', 'value': 17893},
     'duration': {'text': '48 mins', 'value': 2856},
     'status': 'OK'},
    {'distance': {'text': '17.5 km', 'value': 17512},
     'duration': {'text': '44 mins', 'value': 2650},
     'status': 'OK'},
    {'distance': {'text': '19.0 km', 'value': 19028},
     'duration': {'text': '51 mins', 'value': 3031},
     'status': 'OK'},
    {'distance': {'text': '19.2 km', 'value': 19192},
     'duration': {'text': '51 mins', 'value': 3086},
     'status': 'OK'}
     ]}],
'status': 'OK'}

points = [ x.split() for x in pointraw.split('\n')]

def getquery(points, x , start, end):
    origins = ','.join(points[x])
    destinations = []
    for j in range(start, end):
        destinations.append(','.join(points[j]))
    query = "origins=%s&destinations=%s" % (origins, '|'.join(destinations))
    #queryfile = open("C:\\Users\\vivek\\Milk Production\\Queries\\googlequery%s_%s_%s.txt" % (x, start, end), 'w')
    #queryfile.write(query)
    return query

def call_google(query):
    #return DUMMY_RESPONSE
    req = requests.get(GOOGLE_MAPS_API + query)
    return req.json()

def get_distance_and_time(response, expected_count):
    assert len(response['rows']) == 1
    assert len(response['rows'][0]['elements']) == expected_count
    elements = response['rows'][0]['elements']
    distance = []
    duration = []
    for element in elements:
        distance.append(element['distance']['value'])
        duration.append(element['duration']['value'])
    
    return distance, duration
        
partitions = [0, 20, 40, 59]
def throttled_query():
    dist_matrix_file_name = "C:\\Users\\vivek\\Milk Production\\Queries\\Dist_Matrix.csv"
    time_matrix_file_name = "C:\\Users\\vivek\\Milk Production\\Queries\\Time_Matrix.csv"
    distance_rows = []
    time_rows = []
    for i in range(len(points)):
        distance_rows.append([])
        time_rows.append([])
        for j in range(len(partitions)-1):
            query = getquery(points, i, partitions[j], partitions[j+1])
            print("Query for origin %s and destinations %s to %s is %s" % (i, partitions[j], partitions[j+1], query))
            response = call_google(query)
            distance, duration = get_distance_and_time(response, partitions[j+1] - partitions[j])
            distance_rows[i] += (distance)
            time_rows[i] += (duration)
            time.sleep(0.25)
    np.savetxt(dist_matrix_file_name, distance_rows, delimiter=',', fmt='%.4d')
    np.savetxt(time_matrix_file_name, time_rows, delimiter=',', fmt='%.4d')

throttled_query()
        
