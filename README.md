# DevRob – Pick & Place Simulation (ROS 2 + MoveIt + Gazebo)


### Overview

This project implements a complete pick-and-place simulation using a robotic arm, satisfying the DevRob technical assessment requirements.

A Franka Panda robot is simulated in Gazebo, plans collision-free trajectories using MoveIt 2, and executes a side-grasp pick-and-place pipeline to move a cuboid object from position A to B.

The entire system runs in Docker and is reproducible on any Linux machine.
### Repo structure
```
devrob_pick_place/
├─ Dockerfile
├─ README.md
├─ src/
│  ├─ panda_description/            # or as git submodule/dependency
│  ├─ panda_controller/
│  ├─ panda_moveit/
│  ├─ devrob_pick_place/            # your new package
│  │  ├─ package.xml
│  │  ├─ setup.py
│  │  ├─ devrob_pick_place/
│  │  │  ├─ pick_place_node.py
│  │  │  └─ __init__.py
│  │  ├─ launch/
│  │  │  └─ sim_pick_place.launch.py
│  │  ├─ config/
│  │  │  └─ pick_place_params.yaml
│  │  └─ worlds/
│  │     └─ table_pick_place.world
└─ media/
   └─ demo.gif (or demo.mp4)
```
