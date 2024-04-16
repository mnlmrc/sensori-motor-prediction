
import nibabel as nb
import nitools as nt
import matplotlib.pyplot as plt

subj_id = 'subj100'
typ = 'psc'

dataL = f'/Volumes/diedrichsen_data$/data/SensoriMotorPrediction/smp1/surfaceWB/{subj_id}/{typ}/{typ}.L.func.gii'
dataR = f'/Volumes/diedrichsen_data$/data/SensoriMotorPrediction/smp1/surfaceWB/{subj_id}/{typ}/{typ}.R.func.gii'
labelL = f'/Volumes/diedrichsen_data$/data/SensoriMotorPrediction/smp1/surfaceWB/{subj_id}/{subj_id}.L.32k.label.gii'
labelR = f'/Volumes/diedrichsen_data$/data/SensoriMotorPrediction/smp1/surfaceWB/{subj_id}/{subj_id}.R.32k.label.gii'

DL = nb.load(dataL)
DR = nb.load(dataR)

D_labelL = (labelL)

plt.plot(D_labelL.darrays[0].data)

plt.show()

