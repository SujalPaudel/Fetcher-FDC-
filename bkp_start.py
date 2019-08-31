import argparse
import os
import time
import align.detect_face as detect_face
import cv2
import numpy as np
import tensorflow as tf
from lib.face_utils import judge_side_face
from lib.utils import Logger, mkdir, save_to_file
from project_root_dir import project_dir
from src.sort import Sort
import random
from matplotlib import pyplot
import PIL.Image as Image
import math
from dateutil import parser
from datetime import timedelta

logger = Logger()

g = []
index = 0
start_time = []


def preprocess_image(image, image_name,path_to_db):
    global index
    width = float(image.size[0])
    height = float(image.size[1])

    resize_width = float(100)
    resize_height = float(math.ceil(((height / width) * resize_width)))

    image = image.resize(
        (int(resize_width), int(resize_height)), Image.ANTIALIAS)

    x = image.size[0]
    y = image.size[1]

    x_mid = int(round(x / 2))
    y_mid = int(round(y / 2))

    # calculating the mid point
    x_start = x_mid - 48
    y_start = y_mid - 48

    outfile1 = image_name + str(index) + ".jpg"
    region = image.crop((x_start, y_start, x_start + 96, y_start + 96))

    with tf.gfile.Open(path_to_db, 'w') as fid:
        region.save(fid, 'JPEG')

    return region

def save(image, boxes,path_to_db):
    image_list = []

    global index
    image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(image_np)
    recogintion_label = []



   
    xmin = boxes[0]
    ymin = boxes[1]
    xmax = boxes[2]
    ymax = boxes[3]

    img_width, img_height, _ = image.shape

    image = cv2.rectangle(image, (int(round(xmin *
                                            img_width)), int(round(ymin *
                                                                   img_height))), (int(round(xmax *
                                                                                             img_width)), int(round(ymax *
                                                                                                                    img_height))), (255, 0, 0), 2)
    x = round(xmin)
    y = round(ymin)
    width = round(xmax * img_width)
    height = round(ymax * img_height)

    add_width = ((4.0 / 100.0) * xmax)
    add_height = ((10.0 / 100.0) * ymax)
    outfile1 = "usb/usbdetect" + str(index) + ".jpg"

    region = im.crop(
        (xmin-int(add_width),
         ymin-int(add_height),
        xmax+int(add_width),
        ymax+int(add_height)))
    image_list.append(region)

    with tf.gfile.Open(outfile1, 'w') as fid:
        region.save(fid, 'JPEG')


    preprocess = preprocess_image(region, "pree",path_to_db)

    index += 1

    return image_list, image

def main(called_from, stream_name, vid_strt_time, video_folder):
   
    global colours, img_size
    args = parse_args(video_folder)
    videos_dir = args.videos_dir
    output_path = args.output_path
    no_display = args.no_display
    detect_interval = args.detect_interval  # you need to keep a balance between performance and fluency
    margin = args.margin  # if the face is big in your video ,you can set it bigger for tracking easiler
    scale_rate = args.scale_rate  # if set it smaller will make input frames smaller
    show_rate = args.show_rate  # if set it smaller will dispaly smaller frames
    face_score_threshold = args.face_score_threshold

    mkdir(output_path)
    # for display
    if not no_display:
        colours = np.random.rand(100, 3)

    # init tracker
    tracker = Sort()  # create instance of the SORT tracker

    logger.info('Start track and extract......')
    with tf.Graph().as_default():
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),
                                              log_device_placement=False)) as sess:
            pnet, rnet, onet = detect_face.create_mtcnn(sess, os.path.join(project_dir, "align"))

            minsize = 70  # minimum size of face for mtcnn to detect
            threshold = [0.6, 0.7, 0.7]  # three steps's threshold
            factor = 0.709  # scale factor

            for filename in os.listdir(videos_dir):
                logger.info('All files:{}'.format(filename))
            for filename in os.listdir(videos_dir):
                suffix = filename.split('.')[1]
                if suffix != 'mp4' and suffix != 'avi' and suffix != 'flv':  # you can specify more video formats if you need
                    continue
                video_name = os.path.join(videos_dir, filename)
                directoryname = os.path.join(output_path, filename.split('.')[0])
                logger.info('Video_name:{}'.format(video_name))
                cam = cv2.VideoCapture(video_name)
                c = 0
                while True:
                    final_faces = []
                    addtional_attribute_list = []
                    
                    

                    ret, frame = cam.read()

                    start_time.append(time.time())
                    # print(time.time())

                    if not ret:
                        logger.warning("ret false")
                        break
                    if frame is None:
                        logger.warning("frame drop")
                        break

                    frame = cv2.resize(frame, (0, 0), fx=scale_rate, fy=scale_rate)
                    r_g_b_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if c % detect_interval == 0:
                        img_size = np.asarray(frame.shape)[0:2]
                        mtcnn_starttime = time.time()
                        faces, points = detect_face.detect_face(r_g_b_frame, minsize, pnet, rnet, onet, threshold,
                                                                factor)

                        logger.info("MTCNN detect face cost time : {} s".format(
                            round(time.time() - mtcnn_starttime, 3)))  # mtcnn detect ,slow
                        face_sums = faces.shape[0]
                        if face_sums > 0:
                            face_list = []
                            for i, item in enumerate(faces):
                                score = round(faces[i, 4], 6)
                                if score > face_score_threshold:
                                    det = np.squeeze(faces[i, 0:4])


                                    # face rectangle
                                    det[0] = np.maximum(det[0] - margin, 0)
                                    det[1] = np.maximum(det[1] - margin, 0)
                                    det[2] = np.minimum(det[2] + margin, img_size[1])
                                    det[3] = np.minimum(det[3] + margin, img_size[0])
                                    face_list.append(item)

                                    # face cropped
                                    bb = np.array(det, dtype=np.int32)

                                    # use 5 face landmarks  to judge the face is front or side
                                    squeeze_points = np.squeeze(points[:, i])
                                    tolist = squeeze_points.tolist()
                                    facial_landmarks = []
                                    for j in range(5):
                                        item = [tolist[j], tolist[(j + 5)]]
                                        facial_landmarks.append(item)
                                    if args.face_landmarks:
                                        for (x, y) in facial_landmarks:
                                            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
                                    cropped = frame[bb[1]:bb[3], bb[0]:bb[2], :].copy()


                                    dist_rate, high_ratio_variance, width_rate = judge_side_face(
                                        np.array(facial_landmarks))

                                    # face addtional attribute(index 0:face score; index 1:0 represents front face and 1 for side face )
                                    item_list = [cropped, score, dist_rate, high_ratio_variance, width_rate]
                                    addtional_attribute_list.append(item_list)

                            final_faces = np.array(face_list)

                    trackers = tracker.update(final_faces, img_size, directoryname, addtional_attribute_list, detect_interval)


                    c += 1

                    for d in trackers:

                        d = d.astype(np.int32)

                        g.append(str(start_time[4]))
                        if not no_display:

                            d = d.astype(np.int32)

                            if final_faces != []:
                                
                                try:
                                    os.mkdir('DB/' + called_from + str(d[4]))

                                    first_time = round(start_time[0])
                              
                                    face_time = round(time.time())
                                    entryTime = face_time - first_time

                                    dr = parser.parse(vid_strt_time)

                                    a = dr + timedelta(seconds = entryTime)

                                    real_enter_time =  a.strftime("%H:%M:%S")

                                    f = open("DB/entryTime.txt", "a")
                                    f.write(called_from + str(d[4]) + ',' + real_enter_time + "\n")
                                    f.close()

                                    f = open("DB/stream_of_folder.txt", "a")
                                    f.write(stream_name + ',' + called_from + str(d[4]) + "\n")
                                    f.close()

                                    profile_id = called_from + str(d[4])

                                except Exception:
                                    print("The folder already exists!!")
                                
                                image_path = "DB/" + called_from + str(d[4]) + '/'  + str(random.randint(32, 12141212)) + ".jpg"

                              
                                d[0] = d[0]
                                d[1] = d[1]

                                save(frame, d, image_path)

                                original_height = (d[3] - d[1])

                    if not no_display:
                        frame = cv2.resize(frame, (0, 0), fx=show_rate, fy=show_rate)
                        # cv2.imshow("Frame", frame)
                        # if cv2.waitKey(1) & 0xFF == ord('q'):
                        #     break


def parse_args(video_folder):
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_dir", type=str,
                        help='Path to the data directory containing aligned your face patches.', default=video_folder)
    parser.add_argument('--output_path', type=str,
                        help='Path to save face',
                        default='facepics')
    parser.add_argument('--detect_interval',
                        help='how many frames to make a detection',
                        type=int, default=3)
    parser.add_argument('--margin',
                        help='add margin for face',
                        type=int, default=10)
    parser.add_argument('--scale_rate',
                        help='Scale down or enlarge the original video img',
                        type=float, default=0.7)
    parser.add_argument('--show_rate',
                        help='Scale down or enlarge the imgs drawn by opencv',
                        type=float, default=1)
    parser.add_argument('--face_score_threshold',
                        help='The threshold of the extracted faces,range 0<x<=1',
                        type=float, default=0.85)
    parser.add_argument('--face_landmarks',
                        help='Draw five face landmarks on extracted face or not ', action="store_true")
    parser.add_argument('--no_display',
                        help='Display or not', action='store_true')
    args = parser.parse_args()
    return args

