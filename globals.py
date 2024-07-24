import os
from pathlib import Path

Dirs = ["/Volumes/diedrichsen_data$/data/SensoriMotorPrediction/",
        "/cifs/diedrichsen/data/SensoriMotorPrediction/",
        "/Users/mnlmrc/Documents/data/SensoriMotorPrediction/"]

# Find the first existing directory
baseDir = next((Dir for Dir in Dirs if Path(Dir).exists()), None)

if baseDir:
    print(f"Base directory found: {baseDir}")
else:
    print("No valid base directory found.")

wbDir = "surfaceWB"
glmDir = "glm"
behavDir = "behavioural"
trainDir = "training"
imagingDir = "imaging_data"
RDM = "rdm"
ROI = 'ROI'
pilotDir = 'pilot'

print("Base directory:", baseDir)

col_mov = {
    'smp0': ['trialNum', 'state', 'timeReal', 'time',
             'thumb', 'index', 'middle', 'ring', 'pinkie', 'indexViz', 'ringViz'],
    'smp1': ['trialNum', 'state', 'timeReal', 'time', 'TotTime', 'TR', 'TRtime', 'currentSlice',
             'thumb', 'index', 'middle', 'ring', 'pinkie', 'indexViz', 'ringViz'],
    'smp2': ['trialNum', 'state', 'timeReal', 'time', 'TotTime', 'TR', 'TRtime', 'currentSlice',
             'thumb', 'index', 'middle', 'ring', 'pinkie', 'indexViz', 'ringViz'],
}

participants = {
    'smp0': ['subj100',
             'subj101',
             'subj102',
             'subj103',
             'subj104',
             'subj105',
             'subj106',
             'subj107',
             'subj108',
             'subj109',
             'subj110'],
    'smp1': ['subj100',
             'subj101',
             'subj102',
             'subj103',
             'subj104',
             'subj105',
             'subj106'],
    'smp2': ['subj100',
             'subj101',
             'subj102',
             'subj103',
             'subj104']
}

clabels = ['0%', '25%', '50%', '75%', '100%']

channels = {'mov': ['thumb', 'index', 'middle', 'ring', 'pinkie']}

prestim = 1
poststim = 2
fsample_mov = 500
N = {
    'smp0': len(participants['smp0']),
    'smp1': len(participants['smp1']),
    'smp2': len(participants['smp2']),
}
planState = {
    'smp0': 2,
    'smp1': 3,
    'smp2': 3
}

cue_code = [93, 12, 44, 21, 39]
stimFinger_code = [91999, 99919]
