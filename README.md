# Self-Driving Car Engineer - Final Project

Final (Capstone) Project of [Udacity's Self-Driving Car Engineer Nanodegree](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013) Program

## Introducing

The goal of the project is to capsulate the whole gained knowledge of the "Self-Driving Car Engineer"-Nanodegree course from Udacity and use it to control a real-driving-car on a test-track in San Francisco - California.
The project includes the usage of the [Roboter-Operating-System](https://en.wikipedia.org/wiki/Robot_Operating_System) (short: ROS) which is a framework mostly used in the fields of robotics and autonomous systems.


## Team
The project was proudly developed by the following contributors (in alphabetical order).

| *Name*: | Adam Gyarmati | Gaurav Asati | Kevin Hubert | Lukas Leonard Köning |
|:---:|:---:|:---:|:---:|:---:|
|      *Team role*       |                          Developer                           |                          Developer                           |                     Developer & Teamlead                     |                          Developer                           |
|      *Time zone*       |                  GMT-8<br />(San Francisco)                  |                            GMT+5                             |                     GMT+1<br />(Berlin)                      |                     GMT+1<br />(Munich)                      |
| *GitHub profile image* | [![Adam_GitHubImage](https://avatars1.githubusercontent.com/u/40522904?v=4&s=400)](https://github.com/gyadam) | [![Gaurav_GitHubImage](https://avatars0.githubusercontent.com/u/33591870?s=400&v=4)](https://github.com/gasatig) | [![Kevin_GitHubImage](https://avatars3.githubusercontent.com/u/34512569?s=400&v=4)](https://github.com/KevinHubert-Dev) | [![LukasLeonard_GitHubImage](https://avatars2.githubusercontent.com/u/6058766?s=400&v=4)](https://github.com/LukasLeonardKoening) |

** *Click on the picture of the respective person to view their GitHub profile.*

---

## Setup

We recommend to use the Docker container provided by Udacity. To do so, first build the docker container and then run the docker file:

```sh
# Ensure that you are in the directory including the provided Dockerfile
# Build
docker build . -t capstone

# Run
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone

# Setup the Docker container
chmod +x setup_docker.bash
./setup_docker.bash
```

To run the server-program on your local enviroment:

``` sh
# Clone this repository

git clone <Repository>
cd <Repository-Folder>

# Install requirements
pip install -r requirements.txt

# Build project and source ros-enviroment-variables
cd ros
catkin_make
source ./devel/setup.bash

# Launch
roslaunch ./launch/styx.launch
```

To see the result you need to download the [simulator](https://github.com/udacity/CarND-Capstone/releases) (we have tested it using version 1.3) and run it. Ensure that port 4567 is free to use.

---

## Referencens and additional reading material

The following references/links/papers gave us inspiration and helped us to solve the project.

- [Faster R-CNN paper](https://arxiv.org/pdf/1506.01497.pdf): Original paper of Faster R-CNN
- [Review: Faster R-CNN (Object Detection)](https://towardsdatascience.com/review-faster-r-cnn-object-detection-f5685cb30202): A brief explanation of Faster R-CNN 
- [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md): A list of pretrained models for detection and classification in a single network.
- [Tutorial - TensorFlow image recognition and object detection api](https://missinglink.ai/guides/tensorflow/tensorflow-image-recognition-object-detection-api-two-quick-tutorials/): Tutorial how to train existing detection and classification networks for own different data-sets.

We used data for the Traffic Light Detection from following sources:
- Alex Lechner's [GitHub repository](https://github.com/alex-lechner/Traffic-Light-Classification) ("The Lazy Approach")

More about the Self-Driving Car Engineer Nanodegree Program at Udacity can be found here:
[Become a Self-Driving Car Engineer](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013)

---

## Documentation

In this documentation the structure and the way how we achieved the goal (control the car) is briefly explained. The README and documentation was written by all members (especially for their own part of the project) and reviewed by Lukas Leonard Köning.

### Table of Contents

1. ROS System Architecture
2. Traffic Light Detection
3. Waypoint Updater
4. Drive-By-Wire

### 1.) ROS System Architecture

![](imgs/ROS_architecture.png)

The image above shows the system architecture of our implementation. There are three main subsystems in an autonomous vehicle architecture: Perception, Planning, Control. The perception subsystem processes sensor information (like camera images, LiDAR point clouds or IMU data) to information that can be used by the planning subsystem. The planning subsystem takes in this information and decides what the car should do. It creates trajectorys for the control subsystem. The control subsystem takes in this information, uses different controllers to create throttle, break and steering commands and sends them to the components of the car. 

Our task was to implement the Traffic Light Detection Node, the Waypoint Updater Node and the DBW Node. These are explained in the next sections in more detail.

### 2.) Traffic Light Detection

![](imgs/ROS_TLD_Node.png)

The Traffic Light Node is responsible for handling the traffic lights. It subscribes to the `base_waypoints` ROS topic (all waypoints of the track), the `image_color` ROS topic (image stream of the camera) and the `current_pose` ROS topic (position data to determine the current location). We also used the `/vehicle/traffic_lights` ROS node for testing purposes which delivers us information about all traffic lights from the simulator. The Traffic Light Node only publishes the index of the waypoint of the next upcoming red light's stop line to the `traffic_waypoint` ROS topic.

We used a Neural Network approach to the traffic light detection and classification problem. Initially we implemented [RetinaNET](https://towardsdatascience.com/review-retinanet-focal-loss-object-detection-38fba6afabe4) which worked really good but unfortunately is not compatible with the TensorFlow version (1.3) required by the real testing vehicle (Carla). So we went a step back to an older approach to the problem: Faster R-CNN.

**Data collection**

Images from the simulator world are different from real world data. One of the main differences are the traffic lights. On the test track the traffic lights are inside a yellow casings while they are in a black casing in the simulator. 

For the real world data we had to extract images from shared ROS bag files. To gather images from the simulator we had to use the "Manual" option. Once we would have completed this, we had to label (classify) the images of traffic lights by Red (STOP) / Green(GO) / Yellow (Decide...). There are various tools available for this. Once labeling is completed we would have a XML-file with the coordinates and the class of all traffic lights per image. The XML-file needs to be converted to TensorFlow records that can be used to train object detection models.

In this project we have skipped the explained data collection by re-using already available data (s. references section above).

**Object Detection and Classification Model**

For the traffic light detection and classification we are using a combined approach toward detection and classification. Basically you can separate detection and classification and use two neural networks or even mix machine learning and hard coded solutions (e.g. SSD and hard coded classifier). On the other hand you can combine detection and classification into one neural network. We are using Faster R-CNN which consists of following parts: a region proposal network, region of interest pooling and finally classification and linear regression. You can find more information about Faster R-CNN in the articles linked in the references chapter above.

We used a pre-trained model, froze most of the layers and retrained only the classification layers to classify red, yellow and green traffic lights solely. After training on GPU for a long time we achieved the performance needed. Particularly we had to train two separate models for simulator and real-world data.

In the beginning we had serious latency problems (car passed red lights), but we fixed them by.... **TODO: Complete...** 

**Classification Pipeline**

**TODO: Short explanation of classification pipeline**



The Traffic Light Detection was implemented by Adam Gyarmati and Gaurav Asati. It was reviewed by Kevin Hubert and Lukas Leonard Köning.

### 3.) Waypoint Updater

The Waypoint Updater is responsible for planning the path to drive from the current to the next waypoint. Therefore the Waypoint Updater uses the waypoint, where the vehicle recently drove by and uses splines or n-degree polynomial curves to fit the curvature between the given waypoints.

The implementation of the Waypoint Updater can be found in the [./ros/src/waypoint_updater/waypoint_updater.py](https://github.com/gyadam/self-driving-system-integration/blob/master/ros/src/waypoint_updater/waypoint_updater.py) file. 

The development was split up into two separate feature branches. Moreover we decided to have a developer responsible for implementation and one for quality assurance by doing a code review.


| Branch | Implementation | Code Review |
|:---:|:---:|:---:|
|[waypoint_updater_partial](https://github.com/gyadam/self-driving-system-integration/tree/waypoint_updater_partial)|Adam Gyarmati|Kevin Hubert|
|[waypoints_updater_full](https://github.com/gyadam/self-driving-system-integration/tree/waypoints_updater_full)|Kevin Hubert|Adam Gyarmati|

The Waypoint Updater subscribes to four ROS topics, processes the given data and publishes to a single topic (see visualization below) which represents the final waypoints to drive along. 
![visualization of waypoint_updater-node](imgs/ROS_WPU_Node.png)

| Action | ROS topic | Explaination |
|:----:|:----:|:----|
|Subscribe|base_waypoints|This topic is only published once when the simulator starts. It returns a list of waypoints (e.g.: GPS-waypoints) which represents waypoints to drive along.|
|Subscribe|current_pose|This nodes is used to get information about the current position of the car (x-y-coordinates and heading-direction).|
|Subscribe|traffic_waypoint|Get information about upcoming traffic lights and state of the traffic-light to plan behavior (e.g. deceleration).|
|Publish|final_waypoints|The final waypoints contains the waypoints that the car should follow, informations about the speed based on the maximum allowed speed for the given waypoints and deceleration information for upcoming traffic-lights (especially when they are red).|

Below you can find some code snippets used for the waypoint_updater. All snippets are part of the `waypoint_updater.py` file.

- Plan next waypoints based on the planning-distance, maximum velocity and required acceleration due to traffic-lights:

```python
# Get the range of waypoint of interest for the current planning
closest_idx = self.get_closest_waypoint_idx()
farthest_idx = closest_idx + LOOKAHEAD_WPS
base_waypoints = self.base_lane.waypoints[closest_idx:farthest_idx]

# If there is no stopsign or the given stopsign is out of the range of interest, just keep driving
if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= farthest_idx):
  lane.waypoints = base_waypoints
else: # If the stopsign is relevant for us (in our planning range), calculate when the decelerate
  lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx)
```

- Iterate through waypoints to apply deceleration-curve for a smooth stop infront of a traffic-light:

> Note: For minimization of the code the comments were shorted/changed in the following snippet
```python
for i, wp in enumerate(waypoints):
  p = Waypoint() # ROS Waypoint instance
  p.pose = wp.pose # Copy original waypoint position

  # Get position where to stop. -2 is required cause by car-length
  stop_idx = max(self.stopline_wp_idx - closest_idx - 2, 0)
  # Use distance as coefficient for deceleration-curve
  dist = self.distance(waypoints, i, stop_idx)

  # Calculate velocity for specific waypoit
  vel = math.sqrt(2 * MAX_DECEL * dist)
  if vel < 1.:
    vel = 0.

  # Use the smaller value of the following two:
  #   - maximum-speed due to speed-limit
  #   - decelerating cause by detected red traffic light
  p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x) 
  temp.append(p)
  return temp
```

The Waypoint Updater was implemented by Adam Gyarmati and Kevin Hubert.

### 4.) Drive-By-Wire

![](imgs/ROS_DBW_Node.png)

The DBW Node (Drive-By-Wire) is responsible for sending throttle, steering and brake commands to the car. Drive by wire is the use of electrical or electro-mechanical systems for performing vehicle functions traditionally achieved by mechanical systems, e.g. hitting the brake pedal. 

In this implementation the DBW Node subscribes to the `current_velocity` ROS topic (current velocity of the car), to the `twist_cmd` ROS topic (velocity commands from the Waypoint Follower Node) and to the `dbw_enabled` ROS topic (information about the DBW system status). The DBW Node publishes onto the `throttle_cmd`, `steering_cmd` and the `brake_cmd` ROS topics. 

The implementation is splitted up into two files: `ros/src/twist_controller/dbw_node.py` and `ros/src/twist_controller/twist_controller.py`. Inside the `dbw_node.py` file is the whole ROS logic implemented (Subscribers, Publisher) and also the loop, which runs at 50Hz the controller. It only publishes onto the topics, if drive by wire is enabled. There are cases, where drive by wire is disabled, e.g. when the car stands at a red traffic light. 

The controller is inside the `twist_controller.py` file. It uses a Lowpass Filter, a PID controller and a Yaw controller (`yaw_controller.py`) to achieve the throttle and steering commands. To reduce oscillation from the velocity command (from the `twist_cmd` topic) the Lowpass filter is used. Since the integrational error of the PID controller would increase, when the vehicle does not move, the PID controller has to be reseted every time the DBW is disabled.
The brake command is calculated seperatly as torque value:

```python
## Brake value calculation
# Since the car has an automatic transmission, a brake value is needed to keep the car in position
if linear_vel == 0. and current_vel < 0.1:
  throttle = 0
  brake = 700 #N*m - Torque
# If we want to decrease velocity, brake is used
elif throttle < .1 and delta_v < 0:
  throttle = 0
  decel = max(delta_v, self.decel_limit)
  brake = abs(decel)*self.vehicle_m*self.wheel_r #N*m - Torque
```

The torque for the brake motor is calculated by `deceleration * vehicle_mass * wheel_radius`.

**The "overshooting" problem**

During testing we encountered an "overshooting" problem in the steering (s. image left). This was caused by the Waypoint Follower node which only sends new angular velocity commands, if the vehicle drifts of the trajectory further then a specific threshold. We fixed this issue by removing the threshold and computing the angular velocity at every rate.

| "Overshooting" issue       | Fixed "Overshooting"             |
| -------------------------- | -------------------------------- |
| ![GIF to visualize overshooting](imgs/overshooting.gif) | ![GIF to visualize fixed overshooting](imgs/overshooting_fixed.gif) |

The DBW Node was implemented by Lukas Leonard Köning and was reviewed by Adam Gyarmati.

---
## Result

Below you have two excellent visualization of our resulting algorithm.

In the first image you can see the acceleration after the traffic-light switched from red to green.
![GIF to visualize overshooting](imgs/result_redlight.gif) 

In the second image you can clearly see the deceleration due the fact that the car realises a red-traffic-light ahead. The deceleration is kinda smooth and even as the traffic-light turns green while the cars is decelerting the change is realised by the car and it stop the deceleration-process and continues the driving by accelerate. The detection, classification and behavior planning is done in realtime.

![GIF to visualize fixed overshooting](imgs/result_slowdown.gif) |


As you can see in the first image we realised that there is a little delay between the traffic light change from red -> green and the start of the acceleration of the car.
This happend due to two factors:
1.) Instead of a faster and less reliable neural-network we used the more reliable approach because in real self-driving cars there is more computation-power that in our local enviroment and so we have both the accurancy which is required for safty and the required speed.
2.) Before take in account of a different classification than before we ensure that the classified state was found three times to avoid wrong behavior based on a wrong classification which may can be caused by noisy-image-data or similar problems.

Last but not least we also decided that the safest option is to stop the car which means that if we are unsure about the current prediction we are more likly to choose the saftest-option which is stopping the car and wait for more data.

---

## License

Copyright (c) 2020 Adam Gyarmati, Gaurav A., Kevin Hubert, Lukas Leonard Köning

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
