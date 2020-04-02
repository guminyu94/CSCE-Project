# pip install mxnet, cv2, matplotlib

import json
from gluoncv import model_zoo, data
from mxnet import nd, gpu, init
import numpy as np
import cv2
from matplotlib import pyplot as plt
import mxnet as mx
from mxnet.gluon import nn
from matplotlib import patches
import skvideo
skvideo.setFFmpegPath('C:/Users/gumin/Desktop/machineLearning/ffmpeg-20200315-c467328-win64-shared/bin')
import skvideo.io

def video_cap(filename):
    cap = cv2.VideoCapture(filename)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
    fc = 0
    ret = True

    while (fc < frameCount  and ret):
        ret, buf[fc] = cap.read()
        fc += 1
    cap.release()

    sampling_rate = 30
    sampling_rate = int(30/sampling_rate)

    sampled_frame = np.empty((frameCount//sampling_rate+1, frameHeight, frameWidth, 3), np.dtype('uint8'))
    for i in range(frameCount):
        if (i % sampling_rate) == 0:
            sampled_frame[i // sampling_rate] = buf[i]
           # for j in range(3):
                # flip the image as it is capatured upside down
                #sampled_frame[i // sampling_rate,:,:,j] = np.flipud(sampled_frame[i // sampling_rate,:,:,j])
                #sampled_frame[i // sampling_rate,:,:,j] = np.fliplr(sampled_frame[i // sampling_rate,:,:,j])
            sampled_frame[i // sampling_rate,:,:,:] = cv2.cvtColor(sampled_frame[i // sampling_rate,:,:,:], cv2.COLOR_BGR2RGB)
    return sampled_frame, frameCount//sampling_rate+1


# define net for obj detection
def detect_net():
    net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True,ctx=gpu(0))
    net.load_parameters("yolo3_darknet53_best.params")
    return net


def detect(net, frame_image):
    x, img = data.transforms.presets.yolo.transform_test(mx.nd.array(frame_image), short=512)
    class_IDs, scores, bounding_boxs = net(x.as_in_context(gpu(0)))
    #ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0],
                            #class_IDs[0], class_names=net.classes)
    #plt.show()
    final_bounding_box = []
    i=0
    scores_max =0
    for IDs in class_IDs[0]:
        if IDs == 14 and scores[0][i] > 0.3:
            if scores[0][i] > scores_max:
                final_bounding_box = bounding_boxs[0][i]
                scores_max = scores[0][i]
        i = i + 1
    if len(final_bounding_box):
        # obtain human boundbox
        [xmin, ymin, xmax, ymax] = final_bounding_box
        # clip images
        clipped_image = img[int(ymin.asnumpy()):int(ymax.asnumpy()), int(xmin.asnumpy()):int(xmax.asnumpy()), :]
        #plt.imshow(clipped_image)
        #plt.show()
        return clipped_image, [int(xmin.asnumpy()),int(ymin.asnumpy()), int(xmax.asnumpy()),int(ymax.asnumpy())], img
    else:
        return [],[],img


def classify(net,image):
    # one gpu is necessary
    if  len(image):
        transformed_img = data.transforms.presets.imagenet.transform_eval(mx.nd.array(image).as_in_context(gpu(0)))
        pred = net(transformed_img)
        ind = nd.argmax(pred, axis=1).astype('int')
        action = class_list[ind.asscalar()]
        print(action, nd.softmax(pred)[0][ind].asscalar())
        return action,nd.softmax(pred)[0][ind].asscalar()
    else:
        return "None Action", 0.0


def classification_net(classes):
    model_name = 'ResNet101_v2'
    finetune_net = model_zoo.get_model(model_name, pretrained=True)
    with finetune_net.name_scope():
        finetune_net.output = nn.Dense(classes)
    finetune_net.output.initialize(init.Xavier(), ctx=gpu(0))
    finetune_net.collect_params().reset_ctx(gpu(0))
    finetune_net.hybridize()
    # load net
    finetune_net.load_parameters("fine_tune_net_v2_best.params")
    print(finetune_net)
    return finetune_net

# add action label
class_list = ['applauding', 'brushing_teeth', 'cleaning_the_floor','cooking','drinking','gardening','phoning','playing_guitar','pouring_liquid','reading','smoking','using_a_computer','washing_dishes','watching_TV','writing_on_a_book']
class_list.append('sleeping')
class_list.sort()


# change to your video's name, mp4 format
vid_file_name = "VID_20200401_202733"
frames, clip_count = video_cap(vid_file_name + ".mp4")
current_frame = frames[1]
# detect and clip
detect_net = detect_net()
clip_image, bounding_box, img = detect(detect_net, current_frame)
if len(bounding_box):
    [xmin, ymin, xmax, ymax] = bounding_box
    # classify
    classification_net = classification_net(len(class_list))
    action_name = classify(classification_net,clip_image)

    # plot first frame's result
    plt.imshow(img)
    plt.gca().add_patch(patches.Rectangle((xmin,ymin),xmax-xmin,ymax-ymin,linewidth=1,edgecolor='r',facecolor='none'))
    plt.text(xmin+(xmax-xmin)/2, ymin - 20, action_name, fontsize=12,  color='red')
    plt.show()

frames_vid = np.empty(shape=(clip_count,)+(512,910,3))
i=0
data_json = {}
data_json['action'] = []
for current_frame in frames:
    # detect and clip
    clip_image, bounding_box, img = detect(detect_net, current_frame)
    if len(bounding_box):
        [xmin, ymin, xmax, ymax] = bounding_box
        print(bounding_box)
        if xmin < xmax and ymin < ymax and clip_image.shape[0]*clip_image.shape[1]>0 and xmin > 0 and ymin > 0:
            # check if detection fails
            # classify
            print(clip_image.shape)
            action_name,scores = classify(classification_net, clip_image)

        # write bounding box
            img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255,0,0), 3)
            cv2.putText(img, action_name, (int(xmin + (xmax - xmin) / 2), int(ymin - 20)), 1, 2, (255, 0, 0))
            data_json['action'].append([float(i * 1 / 40), action_name,float(scores)])
        else:
            data_json['action'].append([float(i * 1 / 40), action_name,float(0)])
    else:
        data_json['action'].append([float(i * 1 / 40), action_name,float(0)])
    #uncomment this part to see each frame's prediction result
    #cv2.imshow('image', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    # save frame to ndarray of vid
    frames_vid[i] = img
    i = i + 1

with open(vid_file_name + "_detection"+'.json', 'w') as outfile:
    json.dump(data_json, outfile)
skvideo.io.vwrite(vid_file_name + "_v2_detection" + ".mp4", frames_vid,inputdict={},outputdict={'-r': str(30)})

