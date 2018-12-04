import glob
import h5py
import cv2
import errno
import numpy as np
from tqdm import tqdm
from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path
import matplotlib.pyplot as plt


def detect_body_landmarks(image, estimator):
    """

    :param image:
    :param estimator:
    :return:
    """
    width = image.shape[1]
    height = image.shape[0]
    humans = estimator.inference(image, upsample_size=4.0)
    human_count = len(humans)
    landmarks = []
    if human_count > 0:
        human = humans[0]
        for j in human.body_parts:
            body_part = human.body_parts[j]
            x = int(body_part.x * width + 0.5)
            y = int(body_part.y * height + 0.5)
            landmark = [x, y]
            landmarks.append(landmark)
    return np.array(landmarks)

def draw_landmarks(image, landmarks, landmark_color, activity = None):
    centers = {}
    for i in range(landmarks.shape[0]):  # for each body part
        # We have set -1 to the landmarks which were not found in  image
        x = int(landmarks[i, 0])
        y = int(landmarks[i, 1])
        if x != 0 and y != 0:
            center = (x, y)
            centers[i] = center
            cv2.circle(image, center, 3, landmark_color, thickness=1, lineType=4, shift=0)
    for pair_order, pair in enumerate(common.CocoPairsRender):
        # The body landmark not found has -1 as the value
        if landmarks[pair[0], 0] != 0 and landmarks[pair[0], 1] != 0 and \
                landmarks[pair[1], 0] != 0 and landmarks[pair[1], 1] != 0:
            cv2.line(image, centers[pair[0]], centers[pair[1]], landmark_color, 3)


def write_text_image(image, text):
    """
    :param image: image to write on
    :param text: Text  to write
    :return:
    """

    cv2.putText(image, text, (100, 100), cv2.FONT_HERSHEY_DUPLEX, 0.4, (40, 40, 255), 1)
    return np.uint8(image)


def read_videos_and_labels(path, model, logfile):
    """
    A method to read the videos and labels from the directory
    :param path: path of the videos directory
    :param model: openpose model
    :param logfile: Path of the output logs
    :return:
    """
    files = glob.glob(path)
    labels = []
    width = 640
    height = 480
    e = TfPoseEstimator(get_graph_path(model), target_size=(width, height))
    boundary_frames_seperator = 15
    frame_per_clip = 30
    frame_interval = 1
    landmarks_count = common.CocoPart.Background.value
    all_video_clips = []
    for i in range(len(files)):
        if i % 10 == 0:
            print("File Count:%d"%i)
        file = files[i]
        msg = ("Processing File"+file+"\n")
        with open(logfile, 'a') as f:
            f.write(msg)
        print(msg)
        label = file.split("\\")[1]
        if label == '0':
            boundary_frames = 0
        else:
            boundary_frames = boundary_frames_seperator
        try:
            cap = cv2.VideoCapture(file)
            frame_counter = 0
            current_clip = np.zeros((0, landmarks_count, 2))
            while True:
                ret, frame = cap.read()
                if not ret:
                    cap.release()
                    break

                # check if boundary frames has been skipped and it is the desired frame like 3rd or 5th etc
                if frame_counter >= boundary_frames and\
                        frame_counter % frame_interval == 0:
                    humans = e.inference(frame, upsample_size=4.0)
                    max_pro_human = 0
                    if len(humans) == 0:
                        continue
                    if len(humans) > 1:
                        # only choose the one with higher score
                        max_avg_score = 0
                        for h in range(len(humans)):
                            human = humans[h]
                            avg = 0
                            for hb in human.body_parts:
                                avg += human.body_parts[hb].score
                            avg = avg/len(human.body_parts)
                            print("Score%d\t%f" % (h, avg))
                            if avg > max_avg_score:
                                max_avg_score = avg
                                max_pro_human = h
                        print("%d Humans" %len(humans))
                    human = humans[max_pro_human]
                    landmarks_current_frame = np.zeros((landmarks_count, 2))
                    for j in range(landmarks_count):
                        if j in human.body_parts:
                            body_part = human.body_parts[j]
                            x = int(body_part.x * width + 0.5)
                            y = int(body_part.y * height + 0.5)
                            landmarks_current_frame[j, 0] = x
                            landmarks_current_frame[j, 1] = y
                        else:
                            landmarks_current_frame[j, 0] = 0
                            landmarks_current_frame[j, 1] = 0
                    if current_clip.shape[0] > frame_per_clip:
                        print("Error")
                    landmarks_current_frame = np.uint16(landmarks_current_frame)
                    current_clip = np.concatenate((
                        current_clip, landmarks_current_frame.reshape(-1,
                                                                      landmarks_current_frame.shape[0],
                                                                      landmarks_current_frame.shape[1])), axis=0)

                    if frame_counter >= 2 * boundary_frames-1 and current_clip.shape[0] == frame_per_clip:
                        # delete first row
                        if current_clip.shape[0] > frame_per_clip:
                            print("Greater than %d", frame_per_clip)
                        all_video_clips.append(current_clip)
                        labels.append(int(label))
                        if current_clip.shape[0] == frame_per_clip:
                            current_clip = np.delete(current_clip, 0, axis=0)
                frame_counter += 1

            # Number of frames to be discarded
            discard_frames = boundary_frames
            if frame_counter < 3 * boundary_frames:
                discard_frames = frame_counter - (2 * boundary_frames)
            # Loop and discard the frames
            for k in range(discard_frames):
                del(all_video_clips[len(all_video_clips)-1])
                del(labels[len(labels)-1])
        except IOError as exc:
            if exc.errno != errno.EISDIR:
                raise
        except KeyboardInterrupt:
            cap.release()
    return np.array(all_video_clips), np.array(labels)


def write_data_to_h5(file, pixels, labels):
    """
    To write frames and labels to HDF5 file
    :param file: name of the file to write the data to
    :param pixels: image pixels to write
    :param labels: labels to write
    :return:
    """
    hf = h5py.File(file, 'w')
    hf.create_dataset('pixels', data=pixels)
    hf.create_dataset('labels', data=labels)
    hf.close()


def read_data_from_h5(file):
    """
    To read frames and labels from HDF5 file
    :param file: file to read from
    :return: pixels and labels
    """
    hf = h5py.File(file, 'r')
    pixels_h5 = hf.get('landmarks')
    labels_h5 = hf.get('labels')
    pixels = pixels_h5.value
    labels = labels_h5.value
    hf.close()
    return pixels, labels


def plot_body_landmarks(landmarks):
    plt.figure(figsize=(20, 100))
    plt.scatter(landmarks[:, 0], 480 - landmarks[:, 1])
    plt.xticks([])
    plt.yticks([])
    plt.xlim(0, 640)
    plt.ylim(0, 480)
    plt.show()


def convert_landscape_potrait(image):
    """
    :param image: Image to rotate
    :return: rotated image
    """
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), -90, 1)
    image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return image
