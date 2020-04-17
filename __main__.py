#!/usr/bin/env python3

from compression.models import *

if __name__ == "__main__":
    pc = load_pointcloud("rgbd-dataset/apple/apple_1/apple_1_1_100.pcd")
    print (pc)

