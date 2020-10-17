# Gesture_Recognition_Drone
The plan is to execute the idea in two stages

STAGE 1: Build a PIXHAWK Microprocessor based drone that can execute predetermined flight objectives. (STATUS- Complete)

STAGE 2: Use Machine Learning and Computer Vision to detect handkeypoints and use them to calculate roll, pitch and yaw measurements. (STATUS- Complete)

ABOUT THE QUADCOPTER

Parts:
  * Quadcopter Frame Model: F450 HJ450 DJI
  * A2212 1000KV (brushless motor)
  * 30A Electronic Speed Controllers 
  * Pair of 1045 propellers
  * ORANGE LiPo Battery 3000mAh, 30C, 3 cell 11.1V
  * Pixhawk 2.4.6 32bit ARM RC Flight Controller (has in-built GPS and Magnetometer)
  * Remote Controller Model: FlySky FS-i6 2.4G 6CH AFHDS RC Transmitter With FS-iA6B Receiver
  * Current Firmware: PX4

HAND DETECTION MODEL:
  * I have used the model from CMU's Paper on Hand Keypoint Detection in Single Images using Multiview Bootstrapping.
  * an initial keypoint detector is used to produce noisy labels in multiple views of the hand
  * the noisy detections are then triangulated in 3D using multiview geometry or marked as outliers
  * finally, the reprojected triangulations are used as new labeled training data to improve the detector
  * this process is repeated, generating more labeled data in each iteration. A result is derived analytically relating the minimum number of views to achieve target true and false positive rates for a given detector. The method is used to train a hand keypoint detector for single images.
  
  LINK TO THE PAPER: https://arxiv.org/pdf/1704.07809v1.pdf

LATEST UPDATE:
  * The quadcopter has logged in more than 5 hours of flight time and can traverse predetermined flighpaths based communicated using MAVSDK Python library.
  * The hand keypoint detector while very accurate is taking 5 seconds for detection per frame on my system which is not fast enough for a safe flight. The paper does add that the model can be run in realtime using GPUs which I don't have at the moment. Currently working on improving results.
