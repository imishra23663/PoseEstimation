import glob
import cv2
import numpy as np
import errno
from DNN import DNN
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path
from functions import draw_landmarks, convert_landscape_potrait, write_text_image

import matplotlib.pyplot as plt

data_directory = "videos3/"
file = data_directory+"1/2.mp4"
filename = data_directory+"simulation.mp4"
model = "mobilenet_thin"
landmark_color = [0, 255, 0]
pose_classifier = DNN()
pose_classifier.load('model/pose_classifier.h5')
width = 640
height = 480
e = TfPoseEstimator(get_graph_path(model), target_size=(width, height))
max_frames = 5000
feature_frame_count = 310
frame_interval = 1
landmarks_count = 18
data = np.zeros((max_frames, landmarks_count, 2))
frame_counter = 0
significant_frame_counter = 0
files = glob.glob("Sample2/*")
estimator = TfPoseEstimator(get_graph_path(model), target_size=(width, height))
expected_landmarks = 18
frame_counter = 0
boundary_frames_seperator = 15
frame_interval = 1
for i in range(len(files)):
    file = files[i]
    label = file.split("\\")[1]
    boundary_frames = 0
    try:
        i = 0
        cap = cv2.VideoCapture(file)
        current_clip = np.zeros((30, landmarks_count, 2))
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.release()
                break
            #frame = convert_landscape_potrait(frame)
            frame = cv2.resize(frame, (640, 480))
            frame_counter += 1
            # check if boundary frames has been skipped and it is the desired frame like 3rd or 5th etc
            humans = e.inference(frame, upsample_size=4.0)
            max_pro_human = 0
            if len(humans) == 0:
                print("No Human")
                continue
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
            landmarks_current_frame = landmarks_current_frame
            missing_landmarks = expected_landmarks - landmarks_current_frame.shape[0]
            landmarks = landmarks_current_frame.copy()
            landmarks_current_frame[:, 0] = landmarks_current_frame[:, 0] / 640.0
            landmarks_current_frame[:, 1] = landmarks_current_frame[:, 1] / 480.0

            current_clip = np.concatenate((
                current_clip, landmarks_current_frame.reshape(-1,
                                                              landmarks_current_frame.shape[0],
                                                              landmarks_current_frame.shape[1])), axis=0)
            current_clip = np.delete(current_clip, 0, axis=0)
            frame_counter += 1

            # scale the values
            activity = pose_classifier.predict(current_clip.ravel().reshape(1, -1))
            if activity is not None:
                if activity == 1:
                    activity_name = "Waving Hands"
                else:
                    activity_name = "No Waving Hands"

            print("Activity%d" %activity)

            # Draw the landmarks
            # frame = draw_landmarks(frame, landmarks, landmark_color)
            # Write the landmarks
            if activity is not None:
                if activity == 1:
                    activity_name = "Waving Hands"
                else:
                    activity_name = "No Waving Hands"

            # Write the predicted label
            frame = write_text_image(frame, activity_name)

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