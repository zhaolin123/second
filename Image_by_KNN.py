
import cv2
import os
import os.path
import sys
os.chdir('/content/color_recognition/src')

from color_recognition_api import color_histogram_feature_extraction
from color_recognition_api import knn_classifier


def Save_Image(image_array, save_path):
    if image_array.dtype == 'float32':
        # cv2.imwrite(save_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)*255)
        cv2.imwrite(save_path, image_array * 255)
    elif image_array.dtype == 'uint8':
        cv2.imwrite(save_path, image_array)
    else:
        raise ValueError('Unrecognize type of image array: {}', image_array.dtype)

paths="/content/gdrive/My Drive//shortsleevetop/violet/"
for filename in os.listdir(paths): 
    try:
        
        # os.chdir('/content/color_recognition/src')
        print(filename)
        F = paths + filename
        # img = cv2.imread(F)  

         # Read image
        source_image = cv2.imread(F)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        #source_image = root_path + filename
        # print(source_image)
      
        # get the prediction
        color_histogram_feature_extraction.color_histogram_of_test_image(source_image)
        prediction = knn_classifier.main('training.data', 'test.data')

        # print(prediction.__class__)
        print('Detected color is:', prediction)
        # cv2.putText(
        #     source_image,
        #     'Prediction: ' + prediction,
        #     (15, 45),
        #     cv2.FONT_HERSHEY_PLAIN,
        #     3,
        #     200,
        #     )   
        
        # checking whether the training data is ready
        PATH = './training.data'

        if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
            print ('training data is ready, classifier is loading...')
        else:
            print ('training data is being created...')
            open('training.data', 'w')
            color_histogram_feature_extraction.training()
            print ('training data is ready, classifier is loading...')


        # # print(label)
        # if(prediction == 'red'):
        #     print("red")
        #     Save_Image(source_image, '/content/gdrive/My Drive/shortsleevetop/red/red1' + filename)
        # if(prediction == 'black'):
        #     print("black")
        #     Save_Image(source_image, '/content/gdrive/My Drive/shortsleevetop/black/' + filename)
        # if(prediction == 'blue'):
        #     print("blue")
        #     Save_Image(source_image, '/content/gdrive/My Drive/shortsleevetop/blue/' + filename)
        # if(prediction == 'green'):
        #     Save_Image(source_image, '/content/gdrive/My Drive/shortsleevetop/green/' + filename)
        # if(prediction == 'orange'):
        #     Save_Image(source_image, '/content/gdrive/My Drive/shortsleevetop/orange/' + filename)
        # if(prediction == 'violet'):
        #     Save_Image(source_image, '/content/gdrive/My Drive/shortsleevetop/violet/' + filename)
        # if(prediction == 'white'):
        #     Save_Image(source_image, '/content/gdrive/My Drive/shortsleevetop/white/' + filename)
        # if(prediction == 'yellow'):
        #     Save_Image(source_image, '/content/gdrive/My Drive/shortsleevetop/yellow/' + filename)
        
#     Save_Image(img_crop, '/content/gdrive/My Drive/long_sleeve_outwear/' + filename)
 
        if(prediction == 'violet'):
            Save_Image(source_image, '/content/gdrive/My Drive/shortsleevetop/violet/violet1/' + filename)
      
    except:
        continue
