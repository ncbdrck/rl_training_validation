# RL Training & Validation - For Training and Inference of UniROS Prebuilt Environments

This repository contains the necessary scripts to train and evaluate the `gymnasium-based` prebuilt tasks of [rl_environments](https://github.com/ncbdrck/rl_environments) using Stable Baselines3. 

## Prerequisites

### 1. rl_environments Repository
Before installing this package, make sure you have installed [rl_environments](https://github.com/ncbdrck/rl_environments) repository.

### 2. SB3 ROS Support Package

This package contains the necessary scripts to train and evaluate tasks using Stable Baselines3. You can download it from [here](https://github.com/ncbdrck/sb3_ros_support) and follow the instructions to install it.
```shell
# download the package
cd ~/catkin_ws/src
git clone https://github.com/ncbdrck/sb3_ros_support.git

# install the required Python packages by running
cd ~/catkin_ws/src/sb3_ros_support/
git checkout gymnasium
pip3 install -r requirements.txt

# build the ROS packages and source the environment:
cd ~/catkin_ws/
rosdep install --from-paths src --ignore-src -r -y
catkin build
source devel/setup.bash
```
Please note that the instructions assume you are using Ubuntu 20.04 and ROS Noetic. If you are using a different operating system or ROS version, make sure to adapt the commands accordingly.

## Installation

To install the `rl_training_validation` package, follow these steps:

1. Clone the repository:
    ```shell
    cd ~/catkin_ws/src
    git clone https://github.com/ncbdrck/rl_training_validation.git
    ```

2. This package relies on several Python packages. You can install them by running the following command:

    ```shell
    # Install pip if you haven't already by running this command
    sudo apt-get install python3-pip

    # install the required Python packages by running
    cd ~/catkin_ws/src/rl_training_validation/
    pip3 install -r requirements.txt
    ```
3. Build the ROS packages and source the environment:
    ```shell
   cd ~/catkin_ws/
   rosdep install --from-paths src --ignore-src -r -y
   catkin_make
   source devel/setup.bash
   rospack profile
    ```


## Usage

- The first step is to check the `{robot}_{task)_train_sim.py` or `{robot}_{task)_train_real.py` files in the scripts folder
and modify the parameters accordingly.
- The RL model parameters are in the `config` folder in this repo.
- The task configuration is also found in the `config` folder. (`{robot}_{task}_config.yaml`)



**Simulation**:
```shell
rosrun rl_training_validation {robot}_{task)_train_sim.py
```

**Real-World**:
```shell
rosrun rl_training_validation {robot}_{task)_train_real.py
```

## Contact

For questions, suggestions, or collaborations, feel free to reach out to the project maintainer at [j.kapukotuwa@research.ait.ie](mailto:j.kapukotuwa@research.ait.ie).
