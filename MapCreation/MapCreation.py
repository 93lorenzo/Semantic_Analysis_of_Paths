import glob
import torch.nn as nn
from torchvision import models, transforms
import torch
import os
import torch.nn as nn
import numpy as np
from keras.preprocessing import image
from PIL import Image


path="path/"
#path="test/"
N_CLASS=5
batch_size=256
lr=0.001
percentage=0.9
directory_checkpoint = "checkpoint/"

# find the correct checkpoint name as we named in the training
checkpoint_filename = directory_checkpoint + 'model_best_batch' + str(batch_size) + '_lr' + str(lr) + '_' + str(
    percentage) + '.pth.tar'

class ResNet101(nn.Module):
    def __init__(self):
        super(ResNet101, self).__init__()

        num_classes = N_CLASS
        expansion = 4
        self.core_cnn = models.resnet101(pretrained=True)
        self.fc = nn.Linear(512 * expansion, num_classes)

        return

    def forward(self, x):
        x = self.core_cnn.conv1(x)
        x = self.core_cnn.bn1(x)
        x = self.core_cnn.relu(x)
        x = self.core_cnn.maxpool(x)

        x = self.core_cnn.layer1(x)
        x = self.core_cnn.layer2(x)
        x = self.core_cnn.layer3(x)
        x = self.core_cnn.layer4(x)

        x_p = self.core_cnn.avgpool(x)
        x_p = x_p.view(x_p.size(0), -1)
        x = self.fc(x_p)
        #print(x)
        return x, x_p



FirstPartHTML = """
<!DOCTYPE html>
<html>
  <head>
    <title>Simple Map</title>
    <meta name="viewport" content="initial-scale=1.0">
    <meta charset="utf-8">
    <style>
      /* Always set the map height explicitly to define the size of the div
       * element that contains the map. */
      #map {
        height: 99%;
      }
      /* Optional: Makes the sample page fill the window. */
      html, body {
        height: 99%;
        margin: 0;
        padding: 0;
      }
    </style>
  </head>


  <body>
    <div id="map"></div>
    <script type="text/javascript" src="http://maps.google.com/maps/api/js?libraries=geometry&amp;sensor=false&key=AIzaSyDJCqCfipFTeEVfMZV9FmW-i1rpPznZp6I"></script>
    <script>

      var map;
      var KEY = "AIzaSyCzLibAXVbbejhLT1deKX7-ldxhuU8C9W4";

      // *************************
      // params: localization of the marker, the map, the title and the image/icon path
      // *************************
      function markerCreator(loc,map,icon){

        var m =  new google.maps.Marker({
          position: loc,
          map: map,
          icon:  {
            url: icon,
            scaledSize: new google.maps.Size(50, 50)
          }
        }
        );


        return m;
      }


      // *************************
      // Main
      // *************************
      function initMap() {

        map = new google.maps.Map(document.getElementById('map'), {  });

        var directionsDisplay;
        directionsDisplay = new google.maps.DirectionsRenderer();
        var directionsService = new google.maps.DirectionsService();
        directionsDisplay.setMap(map);

"""

ThirdPartHTML = """
// route request
        var request = {
            origin: loc_A,
            destination: loc_B,
            travelMode: google.maps.TravelMode.WALKING
        };

        // get route from positions
        directionsService.route(request, function (response, status) {
            if (status == google.maps.DirectionsStatus.OK) {
                directionsDisplay.setDirections(response);
                // simplest case no waypoints so only one route
                var route = response.routes[0];
                console.log(route);
                var legs = route.legs[0];
                var steps = legs.steps;
                var points;
                var decodedPath;

                // for the current route
                for (var i = 0; i < steps.length; i++) {
                  // the variable points contains all the points of each path segment encoded
                  points = steps[i].polyline.points;
                  // decodedPath will contain all the position of each segment of the entire path
                  decodedPath = google.maps.geometry.encoding.decodePath(points);
                  // loop to iterate for every position
                  for (var j = 0; j < decodedPath.length; j++){
                    var startString = 1;
                    var endString = String(decodedPath[j]).indexOf(",");
                    var lat = String(decodedPath[j]).substring(startString,endString)

                    var startString = endString+2;
                    var endString = String(decodedPath[j]).length-1;
                    var lng = String(decodedPath[j]).substring(startString,endString)

                  }

                }

            } else {
                alert("Directions Request from " + start.toUrlValue(6) + " to " + end.toUrlValue(6) + " failed: " + status);
            }

        });



      }



    </script>
    <script src="https://maps.googleapis.com/maps/api/js?key=INSERT YOUR KEY &callback=initMap"
    async defer></script>
  </body>

</html>"""


def model_loading():
    print("model loading")

    file_name_correct = os.path.dirname(os.path.realpath(__file__)) + "/" + checkpoint_filename

    # use the cprrect model
    model = ResNet101()
    checkpoint = torch.load(file_name_correct, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    return model
# 0 - mountains
# 1 - green roads
# 2 - urban
# 3 - forest
# 4 - water

def classification ( classifier, img):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transformation = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    #print(img)
    test_image = image.load_img(img, grayscale=False, )
    test_image = np.array(test_image)

    test_image=Image.fromarray(test_image)
    img_out = test_image.convert('RGB')

    input = train_transformation(img_out)
    input = input.unsqueeze(0)

    input_var = torch.autograd.Variable(input, volatile=True)
    output, x_p = classifier(input_var)
    top_3_prob,top_3_lab=torch.topk(output,N_CLASS)
    pred = (top_3_lab[0][0])
    print (pred)
    return pred

def label_func(curr_pred):
    label = ""
    if (curr_pred == 0):
        label = "mountains"
    elif (curr_pred == 1):
        label = "green_roads"
    elif (curr_pred == 2):
        label = "urban"
    elif (curr_pred == 3):
        label = "forest"
    elif (curr_pred == 4):
        label = "water"

    return label

def main():
    classifier=model_loading()
    classifier.eval()

    images_array=glob.glob(path+"*.jpg")
    print(images_array[0])

    # order the array
    images_array = sorted(images_array)

    # **********************************
    # give the first and last positions
    # **********************************

    first = images_array[0]
    lat_A = (first.split("_")[1]).split(",")[0]
    lng_A = (first.split("_")[1]).split(",")[1].split("&")[0]

    last = images_array[-1]
    lat_B = (last.split("_")[1]).split(",")[0]
    lng_B = (last.split("_")[1]).split(",")[1].split("&")[0]

    SecondPartHTML = "lat_1=" + str(lat_A) + ";\n  lng_1=" + str(lng_A) + ";\nlat_2=" + lat_B + ";\nlng_2=" + str(
        lng_B) + \
                     ";\nloc_A = { lat : lat_1, lng : lng_1};\nloc_B = { lat : lat_2, lng : lng_2};\n"


    counter = 0

    markers_string = "var markers = [ "

    for img in images_array :
        print(img)
        counter+=1
        #if counter < 15 :
        if True :

            img_lat = (img.split("_")[1]).split(",")[0]
            img_lng = (img.split("_")[1]).split(",")[1].split("&")[0]

            curr_pred = int( classification(classifier, img) )
            label = label_func(curr_pred)
            markers_string += " new google.maps.Marker({  position: { lat: "+ str(img_lat) +", lng:"+ str(img_lng)+"} , map:map, icon: { url:'icons/"+label+".png',scaledSize: new google.maps.Size(30, 30) } } ), "

            print("MARKER STRING IS : " +str(markers_string))
        else :
            break


    markers_string += "];"


    # create the html file

    file = open("testfile.html", "w")
    file.write(FirstPartHTML)
    file.write(SecondPartHTML)
    file.write(markers_string)
    file.write(ThirdPartHTML)

    file.close()


main()
