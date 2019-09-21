from __future__ import division
import sys
import os
import numpy as np
from PIL import Image
import src.siamese as siam
from src.tracker import tracker
from src.parse_arguments import parse_arguments
from src.region_to_bbox import region_to_bbox
import cv2
# import tensorflow as tf
import time
import ffmpeg
import tensorflow as tf

def main():
    # avoid printing TF debugging information
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # TODO: allow parameters from command line or leave everything in json files?
    hp, evaluation, run, env, design = parse_arguments()
    # Set size for use with tf.image.resize_images with align_corners=True.
    # For example,
    #   [1 4 7] =>   [1 2 3 4 5 6 7]    (length 3*(3-1)+1)
    # instead of
    # [1 4 7] => [1 1 2 3 4 5 6 7 7]  (length 3*3)
    final_score_sz = hp.response_up * (design.score_sz - 1) + 1
    # build TF graph once for all
    image, templates_z, scores = siam.build_tracking_graph(final_score_sz, design, env)
    
    # read radio
    width = 640
    height = 480
    process1 = (
        ffmpeg
        .input('tcp://192.168.1.155:8300',vcodec='h264',r = 24,probesize=32,fflags="nobuffer",flags="low_delay",analyzeduration=1)
        .output('pipe:', format='rawvideo',pix_fmt="rgb24")
        .run_async(pipe_stdout=True)
    )
    ## model 
    model_path = './frozen_inference_graph.pb'
    odapi = DetectorAPI(path_to_ckpt=model_path)
    # while True :
    in_bytes = process1.stdout.read(width * height * 3)
    if not in_bytes :
        print ("none")
    video = (np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3]))
    video = cv2.cvtColor(video, cv2.COLOR_RGB2BGR)
    box = odapi.processFrame(video)
    print ('box', box)
    pos_x, pos_y, target_w, target_h = box[0], box[1], box[2], box[3]
    bboxes, speed = tracker(hp, run, design, video, pos_x, pos_y, target_w, target_h, final_score_sz,
                            image, templates_z, scores, process1)
        # update parameters

        # bboxes, speed = tracker(hp, run, design, frame_name_list, pos_x, pos_y, target_w, target_h, final_score_sz,
        #                     filename, image, templates_z, scores, evaluation.start_frame)
        # frame_idx += 1
    # iterate through all videos of evaluation.dataset
    # gt, frame_name_list, _, _ = _init_video(env, evaluation, evaluation.video)
    # pos_x, pos_y, target_w, target_h = region_to_bbox(gt[evaluation.start_frame])
    # bboxes, speed = tracker(hp, run, design, frame_name_list, pos_x, pos_y, target_w, target_h, final_score_sz,
    #                         filename, image, templates_z, scores, evaluation.start_frame)
    # _, precision, precision_auc, iou = _compile_results(gt, bboxes, evaluation.dist_threshold)
    # print (evaluation.video + \
    #         ' -- Precision ' + "(%d px)" % evaluation.dist_threshold + ': ' + "%.2f" % precision +\
    #         ' -- Precision AUC: ' + "%.2f" % precision_auc + \
    #         ' -- IOU: ' + "%.2f" % iou + \
    #         ' -- Speed: ' + "%.2f" % speed + ' --')
    print ('done')


def _compile_results(gt, bboxes, dist_threshold):
    l = np.size(bboxes, 0)
    gt4 = np.zeros((l, 4))
    new_distances = np.zeros(l)
    new_ious = np.zeros(l)
    n_thresholds = 50
    precisions_ths = np.zeros(n_thresholds)

    for i in range(l):
        gt4[i, :] = region_to_bbox(gt[i, :], center=False)
        new_distances[i] = _compute_distance(bboxes[i, :], gt4[i, :])
        new_ious[i] = _compute_iou(bboxes[i, :], gt4[i, :])

    # what's the percentage of frame in which center displacement is inferior to given threshold? (OTB metric)
    precision = sum(new_distances < dist_threshold)/np.size(new_distances) * 100

    # find above result for many thresholds, then report the AUC
    thresholds = np.linspace(0, 25, n_thresholds+1)
    thresholds = thresholds[-n_thresholds:]
    # reverse it so that higher values of precision goes at the beginning
    thresholds = thresholds[::-1]
    for i in range(n_thresholds):
        precisions_ths[i] = sum(new_distances < thresholds[i])/np.size(new_distances)

    # integrate over the thresholds
    precision_auc = np.trapz(precisions_ths)    

    # per frame averaged intersection over union (OTB metric)
    iou = np.mean(new_ious) * 100

    return l, precision, precision_auc, iou


def _init_video(env, evaluation, video):
    video_folder = os.path.join(env.root_dataset, evaluation.dataset, video)
    frame_name_list = [f for f in os.listdir(video_folder) if f.endswith(".jpg")]
    frame_name_list = [os.path.join(env.root_dataset, evaluation.dataset, video, '') + s for s in frame_name_list]
    frame_name_list.sort()
    with Image.open(frame_name_list[0]) as img:
        frame_sz = np.asarray(img.size)
        frame_sz[1], frame_sz[0] = frame_sz[0], frame_sz[1]

    # read the initialization from ground truth
    gt_file = os.path.join(video_folder, 'groundtruth.txt')
    gt = np.genfromtxt(gt_file, delimiter=',')
    n_frames = len(frame_name_list)
    assert n_frames == len(gt), 'Number of frames and number of GT lines should be equal.'

    return gt, frame_name_list, frame_sz, n_frames


def _compute_distance(boxA, boxB):
    a = np.array((boxA[0]+boxA[2]/2, boxA[1]+boxA[3]/2))
    b = np.array((boxB[0]+boxB[2]/2, boxB[1]+boxB[3]/2))
    dist = np.linalg.norm(a - b)

    assert dist >= 0
    assert dist != float('Inf')

    return dist


def _compute_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    if xA < xB and yA < yB:
        # compute the area of intersection rectangle
        interArea = (xB - xA) * (yB - yA)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
    else:
        iou = 0

    assert iou >= 0
    assert iou <= 1.01

    return iou

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
        all_boxes = []                    
        for i in range(len(boxes_list)):
            if classes[i] == 1 and scores[i] > 0.7 :
                item = np.array(boxes_list[i])
                item[2] -= item[0]
                item[3] -= item[1]
                item = [item[1], item[0], item[3], item[2]]
                item[0] += item[2]/2
                item[1] += item[3]/2
                return item
                # item = np.insert(item, 0, [frame_idx, -1])
                # item = item.tolist()
                # item.extend([scores[i], -1, -1, -1])
                # all_boxes.append(item)

        return all_boxes
        # return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])


if __name__ == '__main__':
    sys.exit(main())
