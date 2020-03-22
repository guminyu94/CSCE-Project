To run the codes
1.first install pip
2.pip install mxnet-cu101mkl
3.pip install --upgrade gluoncv
4.pip install opencv-python

put fine_tune_net_2_best_epoch9.params and yolo3_mobilenet1.0_voc_best.params into the same directory as sleep_detection_main.py
download them from https://drive.google.com/open?id=1OphHLm0RhsdZbFEzbm7Pk8PUCZGja4Zi

I use two steps to acheive the human-sleep detection
1. Firstly, using yolo with mobile-net to find human bounding box from each video frame
2. Secondly, A image classifier (total 16 categories of actions) made by ResNet50 is used to classify human action.

sleep_detection_main.py run the detection of sleeping from videos
sleep_classifier.py train the classifier networks
train_yolo3.py train the obj detection networks

youtube videos of demos, https://www.youtube.com/playlist?list=PLiwuVFlHteGGf_YQfn3aMm92na6_nukvc

GitHub repository URL, https://github.com/guminyu94/CSCE-Project