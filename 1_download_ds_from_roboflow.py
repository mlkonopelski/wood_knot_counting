# -*- coding: utf-8 -*-
import configparser
from roboflow import Roboflow
import cv2

VERSION = 3

config = configparser.ConfigParser()
config.sections()
config.read('config.ini')

rf = Roboflow(config['roboflow']['key'])
project = rf.workspace(config['roboflow']['workspace']).project(config['roboflow']['project'])

version = project.version(VERSION)
dataset = version.download("yolov8", location='data')

EXAMPLE = 'frame136_jpg.rf.737c13f1e5b44ea32140d59c59a056ce.jpg'
im = cv2.imread(f"KnotCounting-{VERSION}/train/images/{EXAMPLE}")
assert im.shape == (1080, 1920, 3)
