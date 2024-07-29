import nibabel as nb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import globals as gl
import os

# Load the GIfTI label file
atlas = 'ROI'

A = nb.load(os.path.join(gl.baseDir, 'smp1', 'atlases', f'{atlas}.32k.L.label.gii'))

# Extract the label data from the GIfTI file
keys = A.darrays[0].data  # Assumption: the label data is in the first data array
label = [(l.key, l.label) for l in A.labeltable.labels[1:]]
label_df = pd.DataFrame(label, columns=['key', 'name'])
name = ['???'] + label_df['name'].to_list()
key = [0] + label_df['key'].to_list()

# Number of labels
N = len(name)  # Adjust based on your data

# Load a colormap
cmap = plt.cm.get_cmap('tab20', N)

ctab = np.zeros((N, 5), dtype=int)

# Populate the color table with RGBA values and corresponding label IDs
for i in range(N):
    rgb = cmap.colors[i]
    rgb[:3] = rgb[:3] * 255
    rgb[3] = 0
    ctab[i, :4] = rgb  # Store RGBA values
    ctab[i, 4] = i + 1  # Label IDs starting from 1

outpath = os.path.join(gl.baseDir, 'smp1', 'atlases', f'{atlas}.32k.L.label.annot')
nb.freesurfer.io.write_annot(outpath, keys, ctab, name, fill_ctab=True)

# Load the annotation file
labels, ctab, names = nb.freesurfer.io.read_annot(outpath)
labelsC, ctabC, namesC = nb.freesurfer.io.read_annot(os.path.join(gl.baseDir, 'smp1',
                                                                  'surfaceFreesurfer', 'subj100', 'subj100','label', 'lh.aparc.annot'))

# Print outputs to check their correctness
print("Labels:", labels[:10])  # print first 10 labels to check
print("Color Table (first few rows):", ctab[:5])  # first 5 rows of the color table
print("Names:", names[:10])  # first 10 names

