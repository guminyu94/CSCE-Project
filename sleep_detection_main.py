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
#import skvideo
# mpeg root dir, comment if not outputting video
#skvideo.setFFmpegPath('C:/Users/gumin/Desktop/machineLearning/ffmpeg-20200315-c467328-win64-shared/bin')
#import skvideo.io

# import video from mp4 to ndarray
def video_cap(filename):
    cap = cv2.VideoCapture(filename)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = round(cap.get(cv2.CAP_PROP_FPS))

    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
    fc = 0
    ret = True

    while (fc < frameCount  and ret):
        ret, buf[fc] = cap.read()
        fc += 1
    cap.release()

    sampling_rate = 1

    sampled_frame = np.empty((frameCount//sampling_rate+1, frameHeight, frameWidth, 3), np.dtype('uint8'))
    for i in range(frameCount):
        if (i % sampling_rate) == 0:
            sampled_frame[i // sampling_rate] = buf[i]
           # for j in range(3):
                # flip the image as it is capatured upside down
                #sampled_frame[i // sampling_rate,:,:,j] = np.flipud(sampled_frame[i // sampling_rate,:,:,j])
                #sampled_frame[i // sampling_rate,:,:,j] = np.fliplr(sampled_frame[i // sampling_rate,:,:,j])
            sampled_frame[i // sampling_rate,:,:,:] = cv2.cvtColor(sampled_frame[i // sampling_rate,:,:,:], cv2.COLOR_BGR2RGB)
    return sampled_frame, frameCount//sampling_rate+1, fps

# non-max suppression
def nms(bounding_boxes_in, confidence_score_in, threshold):
    bounding_boxes = []
    confidence_score = []
    for j in range(len(bounding_boxes_in)):
        [xmin, ymin, xmax, ymax] = bounding_boxes_in[j]
        bounding_boxes.append([int(xmin.asnumpy()),int(ymin.asnumpy()), int(xmax.asnumpy()),int(ymax.asnumpy())])
        confidence_score.append((confidence_score_in[j].asnumpy())[0])

    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)


        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes, picked_score

# define net for obj detection
def detect_net():
    net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True,ctx=gpu(0))
    net.load_parameters("yolo3_darknet53_best.params")
    return net

# forwards detection
def detect(net, frame_image):
    x, img = data.transforms.presets.yolo.transform_test(mx.nd.array(frame_image), short=512)
    class_IDs, scores_in, bounding_boxs_in = net(x.as_in_context(gpu(0)))
    #ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0],
                            #class_IDs[0], class_names=net.classes)
    #plt.show()
    bounding_box = []
    scores = []
    clipped_image = []
    i = 0
    print(class_IDs[0])

    # find human
    for IDs in class_IDs[0]:
        if IDs == 14 and scores_in[0][i] > 0.4:
            bounding_box.append(bounding_boxs_in[0][i])
            scores.append(scores_in[0][i])
        i = i+1

    if len(bounding_box):
        # obtain human boundbox
        [final_box, final_scores] = nms(bounding_box,scores,0.6)
        for j in range(len(final_box)):
            [xmin, ymin, xmax, ymax] = final_box[j]
        return final_box, img
    else:
        return [],img

# action classify net
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

# action classify
def classify(net,image):
    # one gpu is necessary
    if  len(image):
        transformed_img = data.transforms.presets.imagenet.transform_eval(mx.nd.array(image).as_in_context(gpu(0)))
        pred = net(transformed_img)
        ind = nd.argmax(pred, axis=1).astype('int')
        print(class_list[ind.asscalar()], nd.softmax(pred)[0][ind].asscalar())
        return ind.asscalar(), nd.softmax(pred)[0][ind].asscalar()
    else:
        return 999, 0.0

# a sliding weighted decision maker to determine the current action based on previous action label
def weighted_sliding_select(window_frame_n,new_action_index,scores,labels_q,scores_q):
    # create array to save the scores of each frame
    window_score = np.zeros(16)
    # update the label and scores list
    if (len(labels_q)) > window_frame_n-1:
        # update the q
        labels_q.pop(0)
        labels_q.append(new_action_index)
        scores_q.pop(0)
        scores_q.append(scores)
    else:
        labels_q.append(new_action_index)
        scores_q.append(scores)

    for i in range(len(labels_q)):
        # a linear weight
        window_score[labels_q[i]] = window_score[labels_q[i]] + scores_q[i] * (i+1)
    # returns the index of label
    return np.argmax(window_score), labels_q, scores_q

# IOU computation
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

# match the new bds with the bds existed and saved in a list
def match_bounding_box(bounding_box, obj_bd_list, obj_label_list,obj_scores_list):
    new_obj_bd_list = []
    new_obj_scores_list = []
    new_obj_label_list = []
    for i in range(len(bounding_box)):
        [xmin, ymin, xmax, ymax] = bounding_box[i]
        if xmin < xmax and ymin < ymax and xmin > 0 and ymin > 0:
            # check if detection fails
            find_mark = 0
            for j in range(len(obj_bd_list)):
                # check if current bounding box overlap the bounding box existed in the obj list
                if bb_intersection_over_union(bounding_box[i],obj_bd_list[j]) > 0.6 and find_mark == 0 :
                    # update the bd location and score with the new one
                    new_obj_bd_list.append(bounding_box[i])
                    # copy the old label and scores list
                    new_obj_scores_list.append(obj_scores_list[j])
                    new_obj_label_list.append(obj_label_list[j])
                    # make sure the bd only matched once
                    find_mark = 1
            if find_mark == 0:
                # if not found then add new obj to the list
                new_obj_bd_list.append(bounding_box[i])
                new_obj_scores_list.append([])
                new_obj_label_list.append([])
        # will throw any obj contains in old list that is not in the new bd, considering obj exits the frame
    return new_obj_bd_list, new_obj_label_list, new_obj_scores_list

# add action label
class_list = ['applauding', 'brushing_teeth', 'cleaning_the_floor','cooking','drinking','gardening','phoning','playing_guitar','pouring_liquid','reading','smoking','using_a_computer','washing_dishes','watching_TV','writing_on_a_book']
class_list.append('sleeping')
class_list.sort()

# lists for tracking the obj
obj_bd_list = []
labels_q_list = []
scores_q_list = []

# change to your video's name, mp4 format
vid_file_name = "VID_20200416_190235"
frames, clip_count, fps = video_cap(vid_file_name + ".mp4")
# omit the first frame
current_frame = frames[1]
# set number of frame for decision window
window_frame_n = fps
# test the first frame with the nets
# detect and clip
detect_net = detect_net()
bounding_box, img = detect(detect_net, current_frame)
# classify
classification_net = classification_net(len(class_list))
# plot the first frame's result
for i in range(len(bounding_box)):
    [xmin, ymin, xmax, ymax] = bounding_box[i]
    clipped_image = img[(ymin):(ymax), (xmin):(xmax), :]
    action_index,scores = classify(classification_net,clipped_image)

    plt.imshow(img)
    plt.gca().add_patch(patches.Rectangle((xmin,ymin),xmax-xmin,ymax-ymin,linewidth=1,edgecolor='r',facecolor='none'))
    plt.text(xmin+(xmax-xmin)/2, ymin - 20, class_list[action_index], fontsize=12,  color='red')
plt.show()


frames_vid = np.empty(shape=(clip_count,)+(512,910,3))
i = 0
# json data saves the action
data_json = {}
data_json['action'] = []

# start iterative each frame
for current_frame in frames:
    # detect and clip, may generate multi bd
    bounding_box, img = detect(detect_net, current_frame)
    # match the new obj with the previous one
    obj_bd_list, labels_q_list, scores_q_list = match_bounding_box(bounding_box, obj_bd_list, labels_q_list, scores_q_list)
    actions_frame = []
    # iterative each bd
    for j in range(len(obj_bd_list)):
        [xmin, ymin, xmax, ymax] = obj_bd_list[j]
        print(obj_bd_list[j])
        clipped_image = img[(ymin):(ymax), (xmin):(xmax), :]
        if clipped_image.shape[0] * clipped_image.shape[1] > 1:
            # check if size of clip image is normal
            # classify
            action_index, scores = classify(classification_net, clipped_image)
            action_index, labels_q_list[j], scores_q_list[j] = weighted_sliding_select(window_frame_n,action_index,scores,labels_q_list[j],scores_q_list[j])

            if action_index != 999:
                action_name = class_list[action_index]
            else:
                action_name = "None Action"
            actions_frame.append(action_name)
        # write bounding box
            img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255,0,0), 3)
            cv2.putText(img, action_name, (int(xmin + (xmax - xmin) / 2), int(ymin - 20)), 1, 2, (255, 0, 0))

            # assume we cap every frame of input mp4 and the fr of mp4 is 40, save one data per sec
            action_label_json_save_interval = fps
            # output the label every ... frames
            data_json['action'].append([float(i * 1 / action_label_json_save_interval), action_name, float(scores)])
        else:
            data_json['action'].append([float(i * 1 / action_label_json_save_interval), "None Action", float(0)])

    #uncomment this part to see each frame's prediction result
    cv2.imshow('image', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # save frame to ndarray of vid
    frames_vid[i] = img
    i = i + 1

# output json and video
with open(vid_file_name + "_detection"+'.json', 'w') as outfile:
    json.dump(data_json, outfile)
#skvideo.io.vwrite(vid_file_name + "_v2_detection" + ".mp4", frames_vid,inputdict={},outputdict={'-r': str(30)})

