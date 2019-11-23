import cv2, os, time
import numpy as np



    
def akaze_desc(images_path, good_image_path, pref, f,name_dataset):
    images = []
    for file in os.listdir(images_path):
        if file.endswith(".jpg"):
            images.append(cv2.imread(images_path+file))

    gray_images = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_images.append(gray);

    good_image = cv2.imread(good_image_path)
    good_gray_image = cv2.cvtColor(good_image, cv2.COLOR_BGR2GRAY)

    detector = cv2.AKAZE_create()

    results = []
    for image in gray_images:
        (kps, desc) = detector.detectAndCompute(image, None)
        results.append((kps,desc,image))

    (kps_good, descs_good) = detector.detectAndCompute(good_gray_image, None)

    i = 0
    
    for (kps, desc, image) in results:
        f.write(pref+" Image "+name_dataset+str(i)+"\n")
        f.write("keypoints: {}, descriptors: {}".format(len(kps), desc.shape)+"\n")
        print("keypoints: {}, descriptors: {}".format(len(kps), desc.shape))
        print("i: ",i)
        start_time = time.time()

        
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=10)   # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
        matches = matcher.knnMatch(desc,descs_good,k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])

        img3 = cv2.drawMatchesKnn(image, kps, good_gray_image, kps_good, good, None, flags=2)
        cv2.imwrite("results_akaze/"+pref+"res"+name_dataset+str(i)+".jpg", img3)
        f.write("Time: {}\n".format(time.time()-start_time))
        i = i + 1

def main():
    good_image_dir = "../data1/"
    bad_image_dir = "../no_data1/"
    good_image = "../data1/img1.jpg"
    #good_image_dir = "../data2/"
    #bad_image_dir = "../no_data2/"
    #good_image = "../data2/img1.jpg"
    f = open("results.txt", "w")
    f.write("Results:\n")
    f.close()
    f = open("results.txt", "a")
    f1 = open("results_2.txt", "w")
    f1.write("Results:\n")
    f1.close()
    f1 = open("results.txt", "a")
    name_dataset = "utyug"

  
    akaze_desc(good_image_dir,good_image,"good",f, name_dataset)
    akaze_desc(bad_image_dir,good_image,"bad",f, name_dataset)
    good_image_dir = "../data2/"
    bad_image_dir = "../no_data2/"
    good_image = "../data2/img1.jpg"
    name_dataset = "pingvin"
    akaze_desc(good_image_dir,good_image,"good",f1,name_dataset)
    akaze_desc(bad_image_dir,good_image,"bad",f1, name_dataset)
    
main()