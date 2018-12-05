
![](https://github.com/imishra23663/PoseEstimation/blob/master/videos/Architsimulation.gif?style=centerme)

<br>**Steps to run the simulation:**

1. Have tensorflow version of tf-pose installed before moving ahead. Please use the link below to see the steps: https://github.com/ildoonet/tf-pose-estimation
2. Do an installation of the required packages using "pip install -r requirements"
3. **videos** directory contains all the video files, divided into two sub folders: 1(contains positive samples) and 0(contains negative samples)
4. You can directly run the **simulation.py** to see the demo(incase it fails, you can always follow the steps below), since we already have the model saved and the training file ready.
5. First, prepare the data for the training by running **data_format.py**, this will save the data in **data/** directory.
6. Train the model using **train.py**, the model will be saved in the **model/** directory.
7. Now to run the simulation all you have to do is run the **simulation.py**, this will run on the sample videos stored in the sample folder.

**Main files and its purpose:**
1. data_format.py: Will read videos from videos/ folder and create training specific data, will get all the frames from the videos, the frames will be then run through tf-pose estimation to extract the body points and then we will divide it into rolling 10 frames and the label for the these frames will depend on the which category the video is from.
	
2. functions.py: Contians all the necessary util functions required for pre-processing the data, reading and writing videos, reading and writing from h5 files and plotting the bodymarks.

3. DNN.py: Defines the neural network model.

4. simulation.py: Will show how the model works and will generate a video with class for each frame to be "Waving"  vs "Non-Waving". 

5. simulation_cam.py: Will do a real-time detection of waving vs non-waving, it will use the system's webcam. 
You can also stream the video live from your mobile-phone/tab. For this you need to install DroidCam on phone and then start streaming the video and then use the ip address given by DroidCam as input to cam_url and the will live-stream the hand waving detection.

6. train.py: Run this file to train the neural network.


**Some important directory:**
1. data/: Contains the output of data_format.py i.e data and labels in h5
2. model/: Save the output of train.py as model
3. videos/: Contains the positive(waving) and negative(non-waving) samples for the generation of data 
4. Sample/: Contains some sample vidoes which simulation.py uses to show the simulation.

**Sample data videos:**
**Negative case - No hand Waving **
![](https://github.com/imishra23663/PoseEstimation/blob/master/videos/no_hand_wave.mp4)

**Psotive case - hand Waving **
![](https://github.com/imishra23663/PoseEstimation/blob/master/videos/handwave.mp4)

**Results:**
![](https://github.com/imishra23663/PoseEstimation/blob/master/videos/testsumulation.mp4)








