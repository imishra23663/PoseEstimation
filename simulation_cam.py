import glob
import time
import cv2
import numpy as np
import errno
from DNN import DNN
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path
from functions import draw_landmarks, convert_landscape_potrait, write_text_image

import matplotlib.pyplot as plt
root_dir = "/home/hrishi/workspace/repo/PoseEstimation/"
model = "mobilenet_thin"

landmark_color = [0, 255, 0]
pose_classifier = DNN()
pose_classifier.load(root_dir+'model/pose_classifier.h5')
width = 640
height = 480
e = TfPoseEstimator(get_graph_path(model), target_size=(width, height))
landmarks_count = 18
required_landmarks_count = 8  # We we only need 8 landmarks for our model
frame_counter = 0
frame_per_clip = 10
significant_frame_counter = 0
files = glob.glob("Sample2/*")
expected_landmarks = 18
frame_counter = 0
boundary_frames_seperator = 15
frame_interval = 1
cam_url = 'http://192.168.42.129:8080/video'

boundary_frames = 0
try:
    i = 0
    cap = cv2.VideoCapture(0)
    current_clip = np.zeros((frame_per_clip, required_landmarks_count, 2))
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            break
        #frame = convert_landscape_potrait(frame)
        start_time = time.time()
        #frame = cv2.resize(frame, (640, 480))
        frame_counter += 1
        # check if boundary frames has been skipped and it is the desired frame like 3rd or 5th etc
        humans = e.inference(frame, upsample_size=4.0)
        max_pro_human = 0
        if len(humans) == 0:
            print("No Human Detected")
        else:
            if len(humans) > 1:
                # only choose the one with higher score
                max_avg_score = 0
                for h in range(len(humans)):
                    human = humans[h]
                    avg = 0
                    for hb in human.body_parts:
                        avg += human.body_parts[hb].score
                    avg = avg / len(human.body_parts)
                    print("Score%d\t%f" % (h, avg))
                    if avg > max_avg_score:
                        max_avg_score = avg
                        max_pro_human = h
                print("%d Humans" % len(humans))
            human = humans[max_pro_human]

            landmarks_current_frame = np.zeros((landmarks_count, 2))
            required_landmark_frame = np.zeros((required_landmarks_count, 2))
            for j in range(landmarks_count):
                if j in human.body_parts:
                    body_part = human.body_parts[j]
                    x = int(body_part.x * width + 0.5)
                    y = int(body_part.y * height + 0.5)
                    landmarks_current_frame[j, 0] = x
                    landmarks_current_frame[j, 1] = y

                    # if the landmark index is less then required_landmarks_count
                    if j < required_landmarks_count:
                        required_landmark_frame[j] = landmarks_current_frame[j]
                else:
                    landmarks_current_frame[j, 0] = 0
                    landmarks_current_frame[j, 1] = 0

            required_landmark_frame[:, 0] = required_landmark_frame[:, 0] / 640.0
            required_landmark_frame[:, 1] = required_landmark_frame[:, 1] / 480.0

            current_clip = np.concatenate((
                current_clip, required_landmark_frame.reshape(-1,
                                                              required_landmark_frame.shape[0],
                                                              required_landmark_frame.shape[1])), axis=0)
            current_clip = np.delete(current_clip, 0, axis=0)
            frame_counter += 1

            # scale the values
            activity = pose_classifier.predict(current_clip.ravel().reshape(1, -1))
            # Draw the landmarks
            frame = draw_landmarks(frame, landmarks_current_frame, landmark_color)
            # Write the landmarks
            if activity is not None:
                if activity == 1:
                    # Write the predicted label
                    frame = write_text_image(frame, "Waving Hands", [25, 75])

        end_time = time.time()
        duration = np.round(end_time-start_time, 3)
        frame = write_text_image(frame, "Frame Processing Time: "+str(duration), [25, 25])
        cv2.imshow('image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except IOError as exc:
    if exc.errno != errno.EISDIR:
        raise
except KeyboardInterrupt:
    cap.release()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()