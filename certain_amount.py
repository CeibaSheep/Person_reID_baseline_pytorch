# -*- coding: utf-8 -*-

# from __future__ import print_function, division
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
import math
from model import ft_net, ft_net_dense, ft_net_NAS, PCB, PCB_test
#import cv2
import multiprocessing
import multiprocessing.queues
import tensorflow as tf
import ffmpeg
import dotracker
from PIL import Image 
import paramiko

import const as C

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


######################################################################
# Global Variable
######################################################################


# multiple process state  
server_list = ['tcp://192.168.1.98:8300','tcp://192.168.1.245:8300', 'tcp://192.168.1.136:8300', 'tcp://192.168.1.155:8500']
# server_list = ['tcp://192.168.1.98:8300']
server_ip = ['192.168.1.98','192.168.1.245','192.168.1.136','192.168.1.155']

raspi = C.raspi

process_list = []
queue_list = []
buffer_process_list = list()


######################################################################
# Configuration
######################################################################


parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='./Market/pytorch',type=str, help='./test_data')
parser.add_argument('--name', default='', type=str, help='save model path')
parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--PCB', action='store_true', help='use PCB' )
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--fp16', action='store_true', help='use fp16.' )
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')



#fp16
try:
    from apex.fp16_utils import *
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')


SCORE_THRES = 0.5
target_cam_addr = ''

opt = parser.parse_args()
###load config###
# load the training config
config_path = os.path.join('./model2/','opts.yaml')
# config_path = os.path.join('./model2',opt.name,'opts.yaml')

with open(config_path, 'r') as stream:
        config = yaml.load(stream)
opt.fp16 = config['fp16'] 
# opt.PCB = config['PCB']
opt.use_dense = config['use_dense']
opt.use_NAS = config['use_NAS']
opt.stride = config['stride']

if 'nclasses' in config: # tp compatible with old config files
    opt.nclasses = config['nclasses']
else: 
    opt.nclasses = 751 

str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

print('We use the scale: %s'%opt.ms)
str_ms = opt.ms.split(',')
ms = []
for s in str_ms: # multiclass
    s_f = float(s)
    ms.append(math.sqrt(s_f))

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

# Load model
#---------------------------
def load_network(network):
    save_path = os.path.join('./model2/','net_%s.pth'%opt.which_epoch)
    # os.path.join('./model',name,'net_%s.pth'%opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network


use_gpu = torch.cuda.is_available()

# Load Collected data Trained model
print('-------test-----------')
if opt.use_dense:
    model_structure = ft_net_dense(opt.nclasses)
elif opt.use_NAS:
    model_structure = ft_net_NAS(opt.nclasses)
else:
    model_structure = ft_net(opt.nclasses, stride = opt.stride)

model = load_network(model_structure)
model.classifier.classifier = nn.Sequential()
model = model.eval()
if use_gpu:
    model = model.cuda()

# encoder = create_box_encoder('./mars-small128.pb', batch_size = 32)


######################################################################
# Base Functions
######################################################################

class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()
        print("Elapsed Time:", end_time-start_time)
        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))
        scores = scores[0].tolist()
        classes = [int(x) for x in classes[0].tolist()]
        item = []
        feature = []
        all_boxes = []
        features = []                    
        for i in range(len(boxes_list)):
            if classes[i] == 1 and scores[i] > 0.8 :
                item = np.array(boxes_list[i])
                # item[2] -= item[0]
                # item[3] -= item[1]
                item = [item[1], item[0], item[3], item[2]]
                feature = image[item[1]:item[3], item[0]:item[2]]
                all_boxes.append(item)
                features.append(feature)
        return all_boxes, features

    def close(self):
        self.sess.close()
        self.default_graph.close()

# Load Data
data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256,128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# get boxes model and extract feature model
model_path = './frozen_inference_graph.pb'
odapi = DetectorAPI(path_to_ckpt=model_path)

# get online frame 
class SharedCounter(object):
    def __init__(self, n = 0):
        self.count = multiprocessing.Value('i', n)

    def increment(self, n = 1):
        """ Increment the counter by n (default = 1) """
        with self.count.get_lock():
            self.count.value += n

    @property
    def value(self):
        """ Return the value of the counter """
        return self.count.value

class Queue(multiprocessing.queues.Queue):

    def __init__(self, *args, **kwargs):

        super(Queue, self).__init__(ctx=multiprocessing.get_context(),*args, **kwargs)
        self.size = SharedCounter(0)

    def put(self, *args, **kwargs):
        self.size.increment(1)
        super(Queue, self).put(*args, **kwargs)

    def get(self, *args, **kwargs):
        self.size.increment(-1)
        return super(Queue, self).get(*args, **kwargs)

    def qsize(self):
        """ Reliable implementation of multiprocessing.Queue.qsize() """
        return self.size.value

    def empty(self):
        """ Reliable implementation of multiprocessing.Queue.empty() """
        return not self.qsize()



######################################################################
# ReID Processing
######################################################################


def extract_feature(image):
    # with torch.no_grad():
    features = torch.FloatTensor()
    input_img = Variable(image.cuda())
    ff = torch.FloatTensor(input_img.shape[0],512).zero_().cuda()
    # input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
    print (input_img.shape)    
    outputs = model(input_img)
    print ('*'*10)
    print (type(outputs))
    # outputs = np.array(outputs)
    ff += outputs
    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
    ff = ff.div(fnorm.expand_as(ff))
    features = torch.cat((features,ff.data.cpu()), 0)
    return features


def start_raspi(server_ip):
    for i in server_ip:
        ssh = paramiko.SSHClient()
        key = paramiko.AutoAddPolicy()
        ssh.set_missing_host_key_policy(key)
        ssh.connect(i,22,'pi',"wuyaoliu",timeout=5)
        stdin,stdout,stderr=ssh.exec_command("killall python")
        stdin,stdout,stderr=ssh.exec_command("cd server && python server_4.py")
    print ("wait for camera initialization")
    time.sleep(2)

# start_raspi(server_ip)


def get_current_frames(server_list, filename, node_id):
    no_target = True
    width = 640
    height = 480
    def worker(i,server_list,process_list):
        process = process_list[i]
        while no_target:
            in_bytes = process.stdout.read(width * height * 3)
            if not in_bytes:
                print ("none",i)
                break
            video = (np.frombuffer(in_bytes,np.uint8).reshape([height,width,3]))
            queue_list[i].put(video)
        return

    for i in range(len(server_list)):
        print ('begin process1')
        process1 = (
            ffmpeg
            .input(server_list[i],vcodec='h264',r = 24,probesize=32,fflags="nobuffer",flags="low_delay",analyzeduration=1, flush_packets=1)
            .output('pipe:', format='rawvideo',pix_fmt="rgb24")
            .run_async(pipe_stdout=True)
        )
        process_list.append(process1)
        q = Queue()
        queue_list.append(q)
    print ('*'*10)

    for i in range(len(process_list)):
        p = multiprocessing.Process(target=worker, args=(i,server_list,process_list))
        p.start()
        buffer_process_list.append(p)
    
    server_num = len(server_list)
    max_similarity = np.array([-1.0]*server_num)
    max_sim_frame = [None]*server_num
    max_sim_box = [None]*server_num
    start_time = time.time()
    query_data = scipy.io.loadmat(filename)
    query_feature = torch.FloatTensor(query_data['query_feature']).cuda()
    for frame_index in range(24):
        for i in range(len(process_list)):
            video = queue_list[i].get()
            boxes, boxes_img = odapi.processFrame(video)
            if max_sim_frame[i] is None:
                max_sim_frame[i] = video
            if len(boxes) == 0 :
                print ('no person in this frame at camera ', i)
                continue
            print ('*'*10)
            print (len(boxes_img))
            num_person =  len(boxes_img)
            if (len(boxes_img)) > 1:
                boxes_img = [data_transforms(k) for k in boxes_img]
                gallery_img = torch.stack(boxes_img,0)
                with torch.no_grad():
                    features = extract_feature(gallery_img)
            else:
                gallery_img = data_transforms(boxes_img[0])
                gallery_img = gallery_img.unsqueeze(0)
                with torch.no_grad():
                    features = extract_feature(torch.cat((gallery_img, gallery_img),dim=0))
            gallery_feature = features.cuda()

            scores = torch.cosine_similarity(query_feature, gallery_feature, dim=-1)
            print ('*'*10)
            print (scores)
            print ("current camera_idx",i)

            if max_similarity[i] < torch.max(scores).item():
                max_sim_box[i] = boxes[torch.argmax(scores)]
                max_similarity[i] = torch.max(scores).item()
                max_sim_frame[i] = video
                print ("find better frame")
                print (torch.max(scores).item())

    print (max_similarity)
    print ("camera_id",np.argmax(max_similarity))
    end_time = time.time()
    print ("process time",end_time-start_time)
    for i in process_list:
        i.kill()
    for i in buffer_process_list:
        i.terminate()
        i.join()
    img_name = []
    for i, im_to_save in enumerate(max_sim_frame):
        im = Image.fromarray(im_to_save)
        im_name = node_id + 'raspi' + str(i + 1) + '.jpeg'
        img_name.append(im_name)
        im.save(im_name)
    return max_similarity, max_sim_frame, max_sim_box, img_name


def boxes_transform(box):
    if box is not None:
        box[2] -= box[0]
        box[3] -= box[1]
    return box


def show_result(max_sim_frame, max_sim_box, max_similarity):
    for i in range(len(max_similarity)):
        bbox = max_sim_box[i]
        fig = plt.figure("result")
        ax = fig.add_subplot(1,len(max_similarity),1+i)
        if bbox is not None:
            r = patches.Rectangle((bbox[0],bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', fill=False)
            ax.add_patch(r)
        ax.imshow(np.uint8(max_sim_frame[i]))
    # plt.ion()
    plt.show()
    # plt.pause(0.001)
    # plt.clf()




def run(server_list, filename, node_id):
    global process_list
    global buffer_process_list
    global queue_list
    # max_similarity: list [] * server_num
    # max_sim_frame: list [numpy, numpy, numpy], numpy: 640 * 480 * 3
    # max_sim_box: [list ] * server_num, list: [1, 2, 3, 4]
    max_similarity, max_sim_frame, max_sim_box, img_name = get_current_frames(server_list, filename, node_id) # target_process, target_queue, target_addr
    max_idx = np.argmax(max_similarity)
    print (max_similarity)
    print (max_sim_box)
    # show_result(max_sim_frame, [boxes_transform(i) for i in max_sim_box], max_similarity)
    for process in process_list:
        process.kill()
    queue_list = []
    for buffer_process in buffer_process_list:
        buffer_process.terminate()
        buffer_process.join()
    process_list = []
    buffer_process_list = []
    return max_similarity, max_sim_box, img_name

# print ('get it')

# dotracker.main(process_list[target_process_id], queue_list[target_process_id], target_box, target_video)

