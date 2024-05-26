import os
from pathlib import Path

baseDir = ["/Users/mnlmrc/Documents/data/SensoriMotorPrediction/",
           "/Volumes/diedrichsen_data$/data/SensoriMotorPrediction/",
           "/cifs/diedrichsen/data/SensoriMotorPrediction/",
           ]
for Dir in baseDir:
    if Path(Dir).exists():
        baseDir = Dir
        break
wbDir = "surfaceWB"
glmDir = "glm"
behavDir = "behavioural"
trainDir = "training"
imagingDir = "imaging_data"
RDM = "rdm"
ROI = 'ROI'

# if Path(baseDir).exists():
#     print("Switch to local directory")
#     baseDir = "/Volumes/diedrichsen_data$/data/SensoriMotorPrediction/"
#
#     # base_dir = '/content/drive/My Drive/UWO/SensoriMotorPrediction'
print("Base directory:", baseDir)
