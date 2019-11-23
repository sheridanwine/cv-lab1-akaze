import cv2, os, time
import numpy as np


def orb_desc(images_path, good_image_path, pref, f, name_dataset):
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

    detector = cv2.ORB_create()
    results = []
    for image in gray_images:
        (kps,desc) = detector.detectAndCompute(image, None)
        #br = cv2.BRISK_create();
        #(kps,desc) = br.compute(image, kps)
        results.append((kps,desc,image))

    (kps_good,descs_good) = detector.detectAndCompute(good_gray_image, None)
   # br_g = cv2.BRISK_create();
    #(kps_good,descs_good) = br_g.compute(image, kps_good)

    i = 0
    
    for (kps, desc, image) in results:
        f.write(pref+" Image "+name_dataset+str(i)+"\n")
        f.write("keypoints: {}, descriptors: {}".format(len(kps), desc.shape)+"\n")
        print("keypoints: {}, descriptors: {}".format(len(kps), desc.shape))
        print("i: ",i)
        start_time = time.time()

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
        matches = bf.match(desc,descs_good)
        print("matches")
        
# Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        matches2 = np.asarray(matches)
        good = np.array([])
        #for m,n in matches:
        #    if m.distance < 0.9:
        #       good.append([m])
        img3 = cv2.drawMatches(image, kps, good_gray_image, kps_good, matches, good, flags=2)
        cv2.imwrite("results_orb/"+pref+"res"+name_dataset+str(i)+".jpg", img3)

       # img3 = cv2.drawMatches(image, kps, good_gray_image, kps_good, good, None, flags=2)
        #cv2.imwrite("../results_fast/"+pref+"res"+name_dataset+str(i)+".jpg", img3)
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

  
    orb_desc(good_image_dir,good_image,"good",f, name_dataset)
    orb_desc(bad_image_dir,good_image,"bad",f, name_dataset)
    #good_image_dir = "../data2/"
    #bad_image_dir = "../no_data2/"
    #good_image = "../data2/img1.jpg"
    #name_dataset = "pingvin"
    #orb_desc(good_image_dir,good_image,"good",f1,name_dataset)
    #orb_desc(bad_image_dir,good_image,"bad",f1, name_dataset)
    #print("bad matched")
main()