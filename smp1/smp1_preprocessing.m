clc
close all
clear

%% params
subj_number = 1;
reference_image_option = 0;
prefix_for_functional_files = 'u';

%% Move from BIDS
% Unzip, move and rename T1 anatomical from BIDS to
% anatomicals/subj_id/<subj_id>_anatomical.nii.

disp('preprocessing BIDS:move_unzip_raw_anat')
smp1_imana('BIDS:move_unzip_raw_anat', 'sn', subj_number)

%% Reslice_LPI
% Reslice anatomical image within LPI coordinate systems. LPI stands for
% Left, Posterior, Inferior which means coordinates increase when you go
% from Left to Right, Posterior to Anterior, and Inferior to Superior in
% the anatomical image.

disp('preprocessing ANAT:reslice_LPI')
smp1_imana('ANAT:reslice_LPI', 'sn', subj_number)

%% Centre AC
% Set AC (Anterior Commissure) in T1 anatomicals as [0,0,0] in the
% anatomical image coordinates. Here is a nice guide to locate AC in the
% anatomical image. Open the Anatomical image in fsleyes (or similar apps)
% Find the Anterior Commissure by moving around the slices Put the center
% of the red cursor (in fsleyes) on AC Read the slice indices (X,Y,Z)
% Insert the X,Y,Z indices in the participants.tsv. The default
% participants.tsv (found on spmj_tools) has three columns named locACx,
% locACy, and locACz where you should insert the values.

disp('preprocessing ANAT:centre_AC')
smp1_imana('ANAT:centre_AC', 'sn', subj_number)

%% ANAT:segmentation
% Run the SPM12 batch script for segmentation and normalization.
%
% template_imana('ANAT:segmentation', 'sn', subj_number)
%
% Results in 5 .nii files starting with c1, c2, …, c5. These are masks
% separating the anatomical image into 5 different segments such as grey
% matter, white matter, skull, etc. disp('preprocessing ANAT:segmentation')

disp('ANAT:segmentation')
smp1_imana('ANAT:segmentation', 'sn', subj_number)

%% Move from BIDS
% (Optional) Unzip, move and rename fmap phase and magnitude from BIDS<> to
% fieldmaps/subj_id/sess<sess number>/<subj_id>_magnitude.nii and
% <subj_id>_phase.nii 

disp('preprocessing BIDS:move_unzip_raw_func')
smp1_imana('BIDS:move_unzip_raw_func', 'sn', subj_number)

% Unzip, move and rename functional runs from BIDS to
% imaging_data_raw/subj_id/sess<sess_number>/<subj_id>_run_<run_number>.nii

disp('preprocessing BIDS:move_unzip_raw_fmap')
smp1_imana('BIDS:move_unzip_raw_fmap', 'sn', subj_number)

%% Make VDM fieldmaps
% In some projects you may use fieldmaps to unwarp your functional images.
% Fieldmaps can quantify the deformations caused by inhomogeneities of the
% magnetic field. When the subject’s head is placed inside the scanner
% bore, the magnetic field of the scanner will get bent. The amount of
% bending is different in different locations in the brain (e.g. the
% bending is higher in hippocampus). Using fieldmaps and unwarping we could
% correct some of the deformations. The Siemens FieldMap sequence (used in
% CFMM), produces two magnitude images (magnitude1.nii and magnitude2.nii),
% and a subtracted phase image (phasediff.nii). SPM uses the phasediff.nii
% and the shorter TE magnitude image (magnitude1.nii) to generate a voxel
% displacement map (VDM) for each functional run. Then in the realign &
% unwarp process, it uses these VDMs to correct the deformations.

disp('preprocessing FUNC:make_fmap')
smp1_imana('FUNC:make_fmap', 'sn', subj_number)

%% Realignment (& Unwarping)
% Run the SPM12 spm_realign() (or spm_realign_unwarp()) script for motion
% correction. Realign process runs a rigid transformation (3 x,y,z
% translation parameters and 3 rotation parameters; overall 6 params) for
% every single functional volume to a reference image. This reference image
% is by default the 1st volume in the first run of the session.
% Alternatively in SPM, you can choose to first align to the 1st volume of
% the first run and then align everything to the mean volume of all runs.
%
% template_imana('FUNC:realign_unwarp', 'sn', subj_number,
%                                       'rtm', reference_image_option)
% OR
% 
% template_imana('FUNC:realign', 'sn', subj_number, ...
%                                'rtm', reference_image_option)
%
% ‘rtm’ option: rtm is short for register to mean. If this option is set as
% 0, SPM aligns all the volumes in a session to the 1st volume of the first
% run. If set as 1, SPM first aligns to the first volume and then aligns to
% mean of all volumes of all runs.


disp('preprocessing FUNC:realign_unwarp (it takes a lot of time...)')
smp1_imana('FUNC:realign_unwarp', 'sn', subj_number, ...
                                      'rtm', reference_image_option)

%%

smp1_imana('FUNC:move_realigned_images', 'sn', subj_number, ...
                                             'rtm', reference_image_option)

%%

smp1_imana('FUNC:meanimage_bias_correction', 'sn', subj_number, ...
                                                 'rtm', reference_image_option, ...
                                                 'prefix', prefix_for_functional_files)

%%

smp1_imana('FUNC:coreg', 'sn', subj_number, ...
                             'rtm', reference_image_option, ...
                             'prefix', prefix_for_functional_files)

%% Finalize co-registration
% After this step check co-registration on FSLeyes

smp1_imana('FUNC:make_samealign', 'sn', subj_number, ...
                                      'rtm', reference_image_option, ...
                                      'prefix', prefix_for_functional_files)

%%

smp1_imana('FUNC:make_maskImage', 'sn', subj_number, ...
                                      'rtm', reference_image_option, ...
                                      'prefix', prefix_for_functional_files)







