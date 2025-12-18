# DevRob – Pick & Place Simulation

## Overview

This project implements a complete pick-and-place simulation using a robotic arm. A Franka Panda robot is simulated in Gazebo Harmonic, plans collision-free trajectories using MoveIt 2, and executes a side-grasp pick-and-place pipeline to move a cuboid object from position A to B. The entire system runs in Docker and is reproducible on any Linux machine.


### Pick and Place Execution
## Demonstration
![Pick and Place Demo](media/pick_and_place.gif)


<!-- - [Pick and Place Simulation Video](media/Pick%20and%20place.mp4) -->


### Lift arm 
This Lift arm is used for drawing shapes in the 3d space. This arm is mainly designed for painting the walls and construction. 

This arm uses the Trapezoidal Velocity profile for smooth operation. 
<!-- - [Lift Arm Motion Video](media/lift_arm.mp4) -->
 
![Lift Arm Demo](media/lift_arm.gif)


Docker Hub Link: [https://hub.docker.com/r/manojkumareedhara2/ur10e](https://hub.docker.com/r/manojkumareedhara2/ur10e)

## System Setup

### Robot Configuration
- **Robot**: Franka Panda URDF and SRDF
- **Source**: https://github.com/justagist/franka_panda_description
- **Controllers**: ros2_control
- **End-effector**: Parallel gripper (simulated)
- **Simulator**: Gazebo Harmonic
- **Planner**: MoveIt 2

## Pick and Place Strategy

### 1. Perception Locking

To ensure stable and repeatable planning:
- Object detection is processed only when the robot is in a predefined HOME pose
- Object centroid, dimensions, and boundary polygon are locked for a single pick attempt
- If the grasp fails, the robot:
  - Retreats safely
  - Returns to HOME
  - Reacquires perception
  - Retries with fresh data

This prevents target drift and inconsistent grasp poses.

### 2. Object Orientation Estimation

The object's yaw orientation is computed using Principal Component Analysis (PCA) on the footprint boundary polygon:

1. Extract 2D boundary points
2. Compute the covariance matrix
3. Extract principal eigenvector
4. Convert the dominant axis to the yaw angle

This aligns the gripper with the object's longest face.

### 3. Side-Grasp Planning

A side-grasp strategy is used instead of a top-down grasp.

**Steps**:
1. Compute approach normals in the XY plane (± directions)
2. Select a side normal
3. Generate a pre-grasp pose outside the object
4. Generate a grasp pose at mid-height
5. Use Cartesian motion for approach and lift

This approach improves grasp stability for cuboid objects.

### 4. Motion Planning and Execution


#### Collision Management
During execution:
- The table and object are added as collision objects
- The object is attached to the gripper after grasp
- The object is detached after placement

This ensures collision-free motion throughout the task.

### 5. Gripper Control and Grasp Validation

The gripper is controlled via a simple open/close interface.

After closing:
- If the fingers fully close, the grasp is considered failed
- If the fingers partially close, the object is considered successfully grasped

This heuristic is sufficient and reliable in simulation.

## Installation

### Build Docker Image

```bash
docker build -t panda_pick_place .
```

### Run Docker Container

```bash
xhost +local:docker

docker run -it --rm \
  --net=host \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  panda_pick_place
```

### Source ROS Workspace

Inside the container:

```bash
source /opt/ros/humble/setup.bash
source /ws_moveit/install/setup.bash
```

## Usage

### Launch Sequence

Open separate terminals and run the following commands in order:

#### Terminal 1: Launch Gazebo Simulation
```bash
ros2 launch panda_description gazebo.launch.py
```

#### Terminal 2: Start Robot Controllers
```bash
ros2 launch panda_controller controllers.launch.py
```

#### Terminal 3: Launch MoveIt 2
```bash
ros2 launch panda_moveit moveit.launch.py
```

#### Terminal 4: Start Object Detection
```bash
ros2 run panda_object_detection object_detection_node
```

#### Terminal 5: Execute Pick and Place
```bash
ros2 run panda_object_detection pick_and_place
```

## Project Structure

```
DevRob/
├── Dockerfile
├── panda_description/       
├── panda_controller/         
├── panda_moveit/              
├── panda_object_detection/    
└── README.md
```

