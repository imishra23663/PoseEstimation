Steps to run the simulation:
1. Have tensorflow version of tf-pose installed before moving ahead. Please use the link below to see the steps: https://github.com/ildoonet/tf-pose-estimation
2. Do an installation of the required packages using "pip install -r requirements"
3. **videos** directory contains all the video files, divided into two sub folders: 1(contains positive samples) and 0(contains negative samples)
4. You can directly run the **simulation.py** to see the demo(incase it fails, you can always follow the steps below), since we already have the model saved and the training file ready.
5. First, prepare the data for the training by running **data_format.py**, this will save the data in **data/** directory.
6. Train the model using **train.py**, the model will be saved in the **model/** directory.
7. Now to run the simulation all you have to do is run the **simulation.py**, this will run on the sample videos stored in the sample folder.

