import h5py
from functions import read_videos_and_labels

"""
Tp process the videos and labels and create desired format to train the model
"""
root_dir = root_dir = "./"
data_h5 = root_dir+'data/body_pos.h5'
model = "mobilenet_thin"
logfile = root_dir+"logs.txt"


landmarks, labels = read_videos_and_labels(root_dir+'videos3/*/*', model, logfile)
hf = h5py.File(root_dir+'/body_pos_clips.h5', 'w')
hf.create_dataset('landmarks', data=landmarks)
hf.create_dataset('labels', data=labels)
hf.close()
