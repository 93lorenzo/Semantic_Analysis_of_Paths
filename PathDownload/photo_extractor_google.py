import urllib, os
import json
import random
import time
import sys
import urllib2
import timeit
start = timeit.default_timer()
import webbrowser
from bs4 import BeautifulSoup
import re
from selenium import webdriver
import time

# very important selenium it's needed to start automatically the browser
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
# very important 

import math
from geopy.distance import vincenty
import time
from time import gmtime, strftime


try:
    from urllib import pathname2url         # Python 2.x
except:
    from urllib.request import pathname2url # Python 3.x
import re

# to have a unique name for the folder
ACTUAL_TIME = strftime("%Y-%m-%d %H:%M:%S", gmtime()).replace(":", "-")

# WE GIVE THE POSITION AND WE GOT BACK A PHOTO #KEY = "AIzaSyBRdGrISJfuDRdlO814z8F5EvQPy3kyTeI"
KEY = "AIzaSyCzLibAXVbbejhLT1deKX7-ldxhuU8C9W4"
#Google Maps Directions API #PATH_KEY = "AIzaSyAuzYCbZNgxmGytPBvkru4uutcMoPfPNR0" #KEY FOR DIRECTIONS
PATH_KEY = "AIzaSyBRdGrISJfuDRdlO814z8F5EvQPy3kyTeI"
COUNTER = 0
decode_string_1 ='''<html><head>
        <script type="text/javascript" src="http://maps.google.com/maps/api/js?libraries=geometry&amp;sensor=false&key=AIzaSyDJCqCfipFTeEVfMZV9FmW-i1rpPznZp6I"></script>
        <script>
            function decode(){
                var array, text, len, i;
                array = ['''
# in this way with beautiful soup we can take the values inside the id="text"
decode_string_2 = '''];            
                len = array.length;
                document.write("<body><p id='text'>");

                for (i = 0; i < len; i++) {
                        var decodedPath = google.maps.geometry.encoding.decodePath(array[i]);
                        document.write(decodedPath);
                }
            }decode();
            document.write("</p></body>");

            
        </script><head><body></body>'''

myloc = "pictures_prova/using_javascript/" #replace with your own location
if not os.path.exists(myloc):
    os.makedirs(myloc)

key = "&key=" + KEY #got banned after ~100 requests with no key

# VERY IMPORTANT
# the exe of gecko must be in the same directory
# VERY IMPORTANT
def position_extraction(html_file):
    #gecko = os.path.normpath(os.path.join(os.path.dirname(__file__), 'geckodriver'))
    #binary = FirefoxBinary(r'C:\Program Files (x86)\Mozilla Firefox\firefox.exe')
    #driver = webdriver.Firefox(firefox_binary=binary, executable_path=gecko+'.exe')
    # LINUX ***
    # VERY IMPORTANT UPDATE FIREFOX AND INSTALL FROM THE SITE GECKO PUTTING IT IN THE BIN FOLDER
    # *** LINUX
    driver = webdriver.Firefox()
    # take the response from the page and then open it with the driver
    file_path = 'file://'+os.path.abspath(html_file)
    print file_path
    response = driver.get('file://'+os.path.abspath(html_file))

    # with 1 second should be fine
    time.sleep(3)
    htmlSource = driver.page_source
    response = htmlSource
    soup = BeautifulSoup(response,"html.parser")
    #print soup
    for node in  soup.findAll(id="text"):
        html_string = ''.join(node.findAll(text=True))

    #to close the window
    stringExtract2 = re.findall(r'\(([^()]+)\)', html_string)
    #for s in stringExtract2:
    #    s = re.sub('[ ]','',s)
    driver.close()
    return stringExtract2



# ******** getStreet(Add, location) *********
# giving as Add the string of lat,lng and as location the folder where
# the images will be stored, we will obtain one image for each
# location ("lat,lng") passed
# ******** ************************ *********
def getStreet(Add,location,waypoint="_"):
  Add = str(Add)
  # ****
  #print " +++"+Add+"*** "+str(location)
  #base = "https://maps.googleapis.com/maps/api/streetview?size=1200x800&location="
  base = "https://maps.googleapis.com/maps/api/streetview?size=600x400&location="
  #base = "https://maps.googleapis.com/maps/api/streetview?size=128x64&location="

  # BEGIN TO CHECK IF THERE IS THE IMAGE
  base_status = "https://maps.googleapis.com/maps/api/streetview/metadata?size=600x300&location="
  MyUrl_status = base_status+Add+key
  response = urllib.urlopen(MyUrl_status)
  data = json.loads(response.read())
  if(str(data["status"]) == "ZERO_RESULTS"):
    return
  # END TO CHECK IF THERE IS THE IMAGE

  MyUrl = base + Add + key
  # ****
  #print MyUrl
  file_name = Add + ".jpg"


  location =  location + str(ACTUAL_TIME) +"prova" +str(waypoint)+"/"
  # create the directory if it does not exist
  dir = location
  if not os.path.exists(dir):
      os.makedirs(dir)

  global COUNTER
  new_name = location + str(COUNTER) + "_" + file_name
  COUNTER = COUNTER+1

  #print "***"+str(new_name)+"***"

  # SLEEP SLEEP SLEEP
  time.sleep(0.4)
  try :
    print MyUrl
    urllib.urlretrieve(MyUrl, new_name)
  except IOError :
      print "--- IO ERROR URL RETRIEVE ---"
"""
Y = zoom
a = not yet figured out what it is
t = tilt angle down or up
h = angle of the rotation
fov->zoom heading->angle rotation  pitch->tilt
"""

# ******** getPath(start, end) *********
# giving as start the starting position as a string of lat,lng
# and as end the a string of lat,lng, we will obtain a set of
# location ("lat,lng") each step of the walking path of google maps
# to have more frame per location you have to put the rotating variables true
# ******** ************************ *********
def getPath(start, end, travel_mode = "walking", rotating_pitch=False, rotating_heading=False, waypoint=None ):
    if waypoint == None :
        path_url = "https://maps.googleapis.com/maps/api/directions/json?origin="+ start +"&destination="+ end +"&mode="+travel_mode+"&key="+PATH_KEY
    else :
        path_url = "https://maps.googleapis.com/maps/api/directions/json?origin=" + start + "&destination=" + end + "&waypoints=" + str(waypoint)+ "&mode=" + travel_mode + "&key=" + PATH_KEY

    print "PATH URL IS "+str(path_url)
    try:
        response = urllib.urlopen(path_url)
        data = json.loads( response.read() )
        # print json.dumps(data, sort_keys=True,indent=4, separators=(',', ': '))
            # data["routes"][0]["legs"][0]["steps"] -> those are the steps in order to reach the place,
        # they have start_location and end_location, the end of the i-th is the start of the i+1-th
        #print json.dumps(data,indent=4, sort_keys=True)
        file_json = open( "json_file.txt", "w")
        file_json.write( str(  json.dumps(data,indent=4, sort_keys=True) )  )
        file_json.close()

        # we just want to name it polyline for now
        #file_polyline = open( "polyline_"+str(start)+"_to_"+str(end)+".txt", "w")
        file_polyline = open( "polyline.txt", "w")
        # html file to write and execute to retrieve points
        file_decode = open( "decode.html", "w")
        file_decode.write(decode_string_1)
        try:
            for x in data["routes"][0]["legs"][0]["steps"]:
                #print "POLYLINE POINTS"
                points = x["polyline"]["points"]
                if "\u" in str(points):
                    points = points.replace("\u" , "\\"+"\u")
                    #continue

                #print points
                file_polyline.write(points+"\n")
                # there is one , more than expected
                # I had to insert a space otherwise can happen some strange char like \' and it ruins everything
                file_decode.write( "'"+ points +" '," )
                print points

        except IndexError:
            print "--- INDEX ERROR IN GETPATH --- \n --- CLICK IN THE LINK TO SEE THE ERROR --- \n"+str(path_url)
        file_decode.write(decode_string_2)
        file_polyline.close()
        file_decode.close()
        # HERE file correctly written

        filename = "decode.html"
        # in filename we have the correct file
        # we should extract from filename the lat and lng
        lat_lng_list =  position_extraction(filename)
        for s  in lat_lng_list :
            ss = re.sub('[ ]', '', s)
            if rotating_pitch:
                for x in range(7):
                    getStreet(Add= str(ss) + "&pitch=" + str(x*45), location=myloc)
            elif rotating_heading:
                #for x in range(2):
                #fov=120 to obtain less zoom possible image
                getStreet(Add= str(ss) + "&fov=120"+"&heading=90" , location=myloc, waypoint=waypoint)
                getStreet(Add= str(ss) + "&fov=120"+"&heading=270" , location=myloc, waypoint=waypoint)
            else:
                getStreet(Add=str(ss), location=myloc)


    except Exception as e:
        print( str(e) )
        #raise  # reraises the exception    print "--- RESPONSE ERROR ---"



#getPath(start=start_path,end=end_path,rotating_heading=True)
# good one
#getPath(start=start_path,end=end_path)

def obtaining_path( start_path, end_path , waypoint=None) :
    lat_1 = float( start_path[0] )
    lat_2 = float( end_path[0] )

    lng_1 = float( start_path[1] )
    lng_2 = float( end_path[1] )

    if waypoint == None :
        getPath(start=str(lat_1) + "," + str(lng_1), end=str(lat_2) + "," + str(lng_2), rotating_heading=True)
    else :
        lng_3 = float( waypoint[1] )
        lat_3 = float( waypoint[0])
        getPath(start=str(lat_1) + "," + str(lng_1), end=str(lat_2) + "," + str(lng_2), waypoint=str(lat_3) + "," + str(lng_3),rotating_heading=True)


#to check if all the element in a list are different
def allUnique(x):
    seen = list()
    return not any(i in seen or seen.append(i) for i in x)


#start_path = "41.8993391,12.5144523"
#end = "41.9013312,12.517322"
#end_path = "41.908248,12.5150477"
pos_list = []


# where we built places.txt https://www.doogal.co.uk/RandomAddresses.php

with open("places.txt", "r") as file:
    # for each line in the file I populate my list
    for line in file.readlines():
        lat = float( line.split(",")[0] )
        lng = float( (line.split(",")[1]).split(",")[0] )
        # fill the list
        pos_list.append([ lat , lng ])
        # send to position to have a path and then images

# random election of a pair of element till the end of the list
while( len(pos_list) > 0 ):
    if len(pos_list) == 2:
        elem_1 = pos_list[1]
        elem_2 = pos_list[0]
        del pos_list[1]
        del pos_list[0]

    else :
        rand_num_1 = random.randint(1, len(pos_list)) -1
        elem_1 = pos_list[rand_num_1]
        del pos_list[rand_num_1]
        print "len is "+str(len(pos_list))
        # ******
        # if the number of elements is odd here we will have len = 0 so just one position
        if (len(pos_list) < 1):
            rand_num_2 = 0
        else:
            rand_num_2 = random.randint(1, len(pos_list)) -1

        #print rand_num_1
        #print rand_num_2

        elem_2 = pos_list[rand_num_2]
        del pos_list[rand_num_2]

    #angle = atan(y2-y1/x2-x1)
    angle = math.atan( float( elem_2[1]-elem_1[1] )/float( elem_2[0]-elem_1[0] ) )
    #print angle
    if angle+90 > 360 :
        perp_angle = angle-90
    else :
        perp_angle = angle + 90

    waypoints = False
    if waypoints == True :
        # we want waypoints ?
        mid_lat = float(elem_1[0] + elem_2[0]) / float(2)
        mid_lng = float(elem_1[1] + elem_2[1]) / float(2)
        DIST = 0.0002

        # 1 waypoint
        m_lat = mid_lat + math.sin(perp_angle)*+DIST
        m_lng = mid_lng + math.cos(perp_angle)*+DIST
        wayp = [m_lat, m_lng]
        print str(vincenty( [mid_lat,mid_lng], wayp).meters)+" meters from the midpoint"
        #obtaining_path(start_path=elem_1, end_path=elem_2,waypoint=wayp)

        # 2 waypoint
        DIST = -DIST
        m_lat = mid_lat + math.sin(perp_angle)*+DIST
        m_lng = mid_lng + math.cos(perp_angle)*+DIST
        wayp = [m_lat, m_lng]
        print wayp
        print str(vincenty( [mid_lat,mid_lng], wayp).meters)+" meters from the midpoint"

        #obtaining_path(start_path=elem_1, end_path=elem_2,waypoint=wayp)

        # 3 waypoint
        DIST = 0
        m_lat = mid_lat + math.sin(perp_angle)*+DIST
        m_lng = mid_lng + math.cos(perp_angle)*+DIST
        wayp = [m_lat, m_lng]
        print wayp
        print str(vincenty( [mid_lat,mid_lng], wayp).meters)+" meters from the midpoint"

        print str(vincenty([0.0, 0.002], [0,0]).meters) + " meters from the midpoint"

        #obtaining_path(start_path=elem_1, end_path=elem_2,waypoint=wayp)


    # ***
    # here the function to obtain the path
    obtaining_path(start_path=elem_1,end_path=elem_2)
    # ***

#Your statements here
stop = timeit.default_timer()
print "TIME :"
print stop - start
