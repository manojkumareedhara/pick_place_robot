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


