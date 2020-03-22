from gluoncv import data, utils
import matplotlib.pyplot as plt
from gluoncv.data import VOCDetection
import mxnet as mx
import numpy as np
import os, time, shutil
from multiprocessing import cpu_count
from mxnet import gluon, image, init, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from gluoncv.utils import makedirs
from gluoncv.model_zoo import get_model

# convert vol to folder format dataset
# list of names of classes to be loaded
class_list = ['applauding', 'brushing_teeth', 'cleaning_the_floor','cooking','drinking','gardening','phoning','playing_guitar','pouring_liquid','reading','smoking','using_a_computer','washing_dishes','watching_TV','writing_on_a_book']

"""
# only import person
class VOCLike(VOCDetection):
    CLASSES = ['person']
    def __init__(self, root, splits, transform=None, index_map=None, preload_label=True):
        super(VOCLike, self).__init__(root, splits, transform, index_map, preload_label)
        
for i in range(len(class_list)):
    # for class_name in class_list:
    # iterate through class label, both for training, testing, and val
    dataset = VOCDetection(root='C:/Users/gumin/Desktop/machineLearning/stanford40', splits=((2018, class_list[i]+'_train'),))
    # make dir first
    if not os.path.exists('Action_data_set/train/' + class_list[i]):
        os.makedirs('Action_data_set/train/' + class_list[i])
    for j in range(len(dataset)):
        # iterative every image in the dataset
        dataset_image_temp, dataset_label_temp = dataset[j]
        # obtain human boundbox
        [xmin, ymin, xmax, ymax, cls_id, difficult] = dataset_label_temp[0]
        image_temp = dataset_image_temp.asnumpy()
        # clip images
        clipped_image = image_temp[int(ymin):int(ymax),int(xmin):int(xmax),:]
        # save images
        plt.imsave(os.getcwd() + '/Action_data_set/train/'+class_list[i]+'/' + str(j) + '.jpg', clipped_image)
        print(os.getcwd() + '/Action_data_set/train/'+class_list[i]+'/' + str(j) + '.jpg')

    dataset = VOCDetection(root='C:/Users/gumin/Desktop/machineLearning/stanford40', splits=((2018, class_list[i]+'_test'),))
    if not os.path.exists('Action_data_set/test/' + class_list[i]):
        os.makedirs('Action_data_set/test/' + class_list[i])
    if not os.path.exists('Action_data_set/val/' + class_list[i]):
        os.makedirs('Action_data_set/val/' + class_list[i])
    for j in range(len(dataset)):
        # iterative every image in the dataset
        dataset_image_temp, dataset_label_temp = dataset[j]
        [xmin, ymin, xmax, ymax, cls_id, difficult] = dataset_label_temp[0]
        image_temp = dataset_image_temp.asnumpy()
        clipped_image = image_temp[int(ymin):int(ymax),int(xmin):int(xmax),:]
        # split original test data into half
        if int(j) % 2 == 0:
            plt.imsave(os.getcwd() + '/Action_data_set/test/' + class_list[i] + '/' + str(int(j)//2) + '.jpg', clipped_image)
            print(os.getcwd() + '/Action_data_set/test/' + class_list[i] + '/' + str(int(j)//2) + '.jpg')
        else:
            plt.imsave(os.getcwd() + '/Action_data_set/val/' + class_list[i] + '/' + str(int(j)//2) + '.jpg', clipped_image)
            print(os.getcwd() + '/Action_data_set/val/' + class_list[i] + '/' + str(int(j)//2) + '.jpg')
"""
class_list.append('sleeping')

# indexing the val of sleep
"""
for i in range(50,99):
    image_temp=plt.imread(os.getcwd() + '/Action_data_set/val/'+'sleeping/'+str(i)+'.jpg')
    plt.imsave(os.getcwd() + '/Action_data_set/val/'+'sleeping/'+str(i-50)+'.jpg',image_temp)
"""

# visualize the image
# plt.imshow(clipped_image)
# plt.imsave(os.getcwd() +'/Action_data_set/applauding_test/'+str(0)+'.jpg',clipped_image)
# plt.show()

"""
# read the image from VOC Dataset and convert them to folder and jpg
for i in 2:
    print(i)
    dataset_image_temp, dataset_label_temp = dataset[i]
    plt.imsave(str(i)+'.jpg', dataset_image_temp)

# convert the images from folder to
dataiter = mx.io.NDArrayIter(dataset_image, dataset_label, batchsize, False, last_batch_handle='discard')
for batch in dataiter:
     print(batch.data[0].asnumpy())
     batch.data[0].shape

train_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(train_path).transform_first(transform_train),
    batch_size=batch_size, shuffle=True, num_workers=num_workers)
bounding_boxes = train_label[:, :4]
print('Num of objects:', bounding_boxes.shape[0])
print('Bounding boxes (num_boxes, x_min, y_min, x_max, y_max):\n',
      bounding_boxes)
class_ids = train_label[:, 4:5]
print('Class IDs (num_boxes, ):\n', class_ids)
end
utils.viz.plot_bbox(train_image.asnumpy(), bounding_boxes, scores=None,
                    labels=class_ids, class_names=dataset.classes)
plt.show()
"""

################################################################################
# We set the hyperparameters as following:

classes = len(class_list)

epochs = 200
lr = 0.001
per_device_batch_size = 32
momentum = 0.9
wd = 0.0001

lr_factor = 0.75
lr_steps = [10, 20, 30, np.inf]

num_gpus = 1
num_workers = 1
ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
batch_size = per_device_batch_size * max(num_gpus, 1)

################################################################################

# define the image aug
jitter_param = 0.4
lighting_param = 0.1

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomFlipLeftRight(),
    transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                 saturation=jitter_param),
    transforms.RandomLighting(lighting_param),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

################################################################################
# With the data augmentation functions, we can define our data loaders

train_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset('C:/Users/gumin/.conda/envs/gluon/source/Action_data_set/train').transform_first(transform_train),
    batch_size=batch_size, shuffle=True, num_workers=num_workers, thread_pool=True)

val_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset('C:/Users/gumin/.conda/envs/gluon/source/Action_data_set/val').transform_first(transform_test),
    batch_size=batch_size, shuffle=False, num_workers = num_workers, thread_pool=True)

test_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset('C:/Users/gumin/.conda/envs/gluon/source/Action_data_set/test').transform_first(transform_test),
    batch_size=batch_size, shuffle=False, num_workers = num_workers, thread_pool=True)

################################################################################




################################################################################
#
# Note that only ``train_data`` uses ``transform_train``, while
# ``val_data`` and ``test_data`` use ``transform_test`` to produce deterministic
# results for evaluation.
#
# Model and Trainer
# -----------------
#
# We use a pre-trained ``ResNet50_v2`` model, which has balanced accuracy and
# computation cost.

model_name = 'ResNet101_v2'
finetune_net = get_model(model_name, pretrained=True)
with finetune_net.name_scope():
    finetune_net.output = nn.Dense(classes)
finetune_net.output.initialize(init.Xavier(), ctx = ctx)
finetune_net.collect_params().reset_ctx(ctx)
finetune_net.hybridize()

trainer = gluon.Trainer(finetune_net.collect_params(), 'sgd', {
                        'learning_rate': lr, 'momentum': momentum, 'wd': wd})
metric = mx.metric.Accuracy()
L = gluon.loss.SoftmaxCrossEntropyLoss()

################################################################################
# Here's an illustration of the pre-trained model
# and our newly defined model:
#
# |image-model|
#
# Specifically, we define the new model by::
#
# 1. load the pre-trained model
# 2. re-define the output layer for the new task
# 3. train the network
#
# This is called "fine-tuning", i.e. we have a model trained on another task,
# and we would like to tune it for the dataset we have in hand.
#
# We define a evaluation function for validation and testing.

def test(net, val_data, ctx):
    metric = mx.metric.Accuracy()
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        print( a )
        outputs = [net(X) for X in data]
        metric.update(label, outputs)

    return metric.get()

################################################################################
# Training Loop
# -------------
#
# Following is the main training loop. It is the same as the loop in
# `CIFAR10 <dive_deep_cifar10.html>`__
# and ImageNet.
#
# .. note::
#
#     Once again, in order to go through the tutorial faster, we are training on a small
#     subset of the original ``MINC-2500`` dataset, and for only 5 epochs. By training on the
#     full dataset with 40 epochs, it is expected to get accuracy around 80% on test data.

lr_counter = 0
num_batch = len(train_data)
save_interval = 10
best_acc = 0



for epoch in range(epochs):
    if epoch == lr_steps[lr_counter]:
        trainer.set_learning_rate(trainer.learning_rate*lr_factor)
        lr_counter += 1

    tic = time.time()
    train_loss = 0
    metric.reset()

    for i, batch in enumerate(train_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        print(label)
        with ag.record():
            outputs = [finetune_net(X) for X in data]
            loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
        for l in loss:
            l.backward()

        trainer.step(batch_size)
        train_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)

        metric.update(label, outputs)

    _, train_acc = metric.get()
    train_loss /= num_batch

    _, val_acc = test(finetune_net, val_data, ctx)

    print('[Epoch %d] Train-acc: %.3f, loss: %.3f | Val-acc: %.3f | time: %.1f' %
             (epoch, train_acc, train_loss, val_acc, time.time() - tic))

    # save the net parameter
    if not (epoch + 1) % save_interval:
        if val_acc > best_acc:
            file_name = "fine_tune_net_3_best_epoch"+str(epoch)+".params"
            finetune_net.save_parameters(file_name)
            best_acc = val_acc
        else:
            file_name = "fine_tune_net_3_epoch" + str(epoch) + ".params"
            finetune_net.save_parameters(file_name)

_, test_acc = test(finetune_net, test_data, ctx)
print('[Finished] Test-acc: %.3f' % (test_acc))

file_name = "fine_tuned_net_2_last.params"
finetune_net.save_parameters(file_name)


# load a image for testing
img = mx.image.imread('C:/Users/gumin/.conda/envs/gluon/source/Action_data_set/test/sleeping/0.jpg')
#img = image.imresize(img,256,256)
#plt.imshow(img.asnumpy())
#plt.show()
img_t = transform_test(img)
val = gluon.data.DataLoader(img_t, batch_size=1, shuffle=False)

# load net
finetune_net.load_parameters("fine_tune_net_2_best_epoch9.params")
# print(finetune_net)

for i, batch in enumerate(val):
    data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
    a = data[0]
    outputs = [finetune_net(X) for X in data]
    print(outputs)
    #ind = nd.argmax(pred, axis=1).astype('int')
    #print('The input picture is classified as [%s], with probability %.3f.'%
          #(class_list[ind.asscalar()], nd.softmax(pred)[0][ind].asscalar()))
