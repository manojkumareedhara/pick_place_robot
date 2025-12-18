# DevRob – Pick & Place Simulation 


### Overview

This project implements a complete pick-and-place simulation using a robotic arm.

A Franka Panda robot is simulated in Gazebo Harmonic, plans collision-free trajectories using MoveIt 2, and executes a side-grasp pick-and-place pipeline to move a cuboid object from position A to B.

The entire system runs in Docker and is reproducible on any Linux machine.

### Setup

Robot: Franka Panda
URDF and SRDF Source:
https://github.com/justagist/franka_panda_description

Controllers: ros2_control

End-effector: Parallel gripper (simulated)

Simulator: Gazebo Classic

Planner: MoveIt 2

### Pick and Place Strategy.
Perception Locking

To ensure stable and repeatable planning:
Object detection is processed only when the robot is in a predefined HOME pose

Object centroid, dimensions, and boundary polygon are locked for a single pick attempt

If the grasp fails, the robot:
retreats safely  

returns to HOME

reacquires perception

retries with fresh data

This prevents target drift and inconsistent grasp poses.


Object Orientation Estimation

The object’s yaw orientation is computed using Principal Component Analysis (PCA) on the footprint boundary polygon:

Extract 2D boundary points

Compute the covariance matrix

Extract principal eigenvector

Convert the dominant axis to the yaw angle

This aligns the gripper with the object’s longest face.


Side-Grasp Planning

A side-grasp strategy is used instead of a top-down grasp.

Steps:

Compute approach normals in the XY plane (± directions)

Select a side normal

Generate a pre-grasp pose outside the object

Generate a grasp pose at mid-height

Use Cartesian motion for approach and lift

This approach improves grasp stability for cuboid objects.


Motion Planning and Execution
8.1 MoveIt 2 Planning

MoveIt 2 is used for:

inverse kinematics

collision checking

trajectory generation

8.2 Collision Management

During execution:

the table and object are added as collision objects

the object is attached to the gripper after grasp

the object is detached after placement

This ensures collision-free motion throughout the task.


Gripper Control and Grasp Validation

The gripper is controlled via a simple open/close interface.

After closing:

if the fingers fully close, the grasp is considered failed

if the fingers partially close, the object is considered successfully grasped

This heuristic is sufficient and reliable in simulation.

11. Running the Simulation
11.1 Build Docker Image
docker build -t panda_pick_place .

11.2 Run Docker Container
xhost +local:docker

docker run -it --rm \
  --net=host \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  panda_pick_place


Inside the container:

source /opt/ros/humble/setup.bash
source /ws_moveit/install/setup.bash

11.3 Launch Sequence
ros2 launch panda_description gazebo.launch.py
ros2 launch panda_controller controllers.launch.py
ros2 launch panda_moveit moveit.launch.py
ros2 run panda_object_detection object_detection_node
ros2 run pandaobject_detection pick_and_place


