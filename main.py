import RetinaFace1
from RetinaFace1 import extract_faces
from model_test import return_feature
from losses_test import arcface_loss, embedding_distance
import cv2
import matplotlib.pyplot as plt

def pipline(imagepath1, imagepath2):
    faces1 = extract_faces(img_path = imagepath1, align = True)
    faces2 = extract_faces(img_path = imagepath2, align = True)


    faces1 = faces1[0]
    faces2 = faces2[0]

    feature1, feature2 = return_feature(faces1), return_feature(faces2)

    #loss_arr = []
    #loss_arr.append(arcface_loss(feature1, feature2))
    loss1 = arcface_loss(feature1, feature2)
    distance1 = embedding_distance(feature1, feature2)
    print("Distance: ", distance1)
    print("Distance of two image: ", loss1)
    #threshhold

    threshhold = 0.1
    if distance1 > threshhold:
        print("Two image is different")
    else:
        print("Two image is one!")



if __name__ == '__main__':
    pipline("data/sample1.jpg", "data/sample2.jpg") # two image is different
    pipline("data/sample15.jpg", "data/sample16.jpg") # two image is one