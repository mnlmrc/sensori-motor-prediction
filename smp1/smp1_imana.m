function varargout = smp1_imana(what,varargin)
    % Template function for preprocessing of the fMRI data.
    % Rename this function to <experiment_name>_imana.m 
    % Don't forget to add path the required tools!
    
    localPath = '/Users/mnlmrc/Documents/';
    cbsPath = '/home/ROBARTS/memanue5/Documents/';
    % Directory specification
    if isfolder(localPath)
        path = localPath;
    elseif isfolder(cbsPath)
        path = cbsPath;
    end
    addpath([path 'GitHub/spmj_tools/'])
    addpath([path 'GitHub/dataframe/util/'])
    addpath([path 'GitHub/dataframe/kinematics/'])
    addpath([path 'GitHub/dataframe/graph/'])
    addpath([path 'GitHub/dataframe/pivot/'])
    addpath([path 'GitHub/surfAnalysis/'])
    addpath([path 'MATLAB/spm12/'])
    addpath([path 'GitHub/rwls/'])
    addpath([path 'GitHub/surfing/surfing/'])
    % addpath([path 'GitHub/suit/'])
    addpath([path 'GitHub/rsatoolbox_matlab/'])
    addpath([path 'GitHub/surfing/toolbox_fast_marching/'])
    addpath([path 'GitHub/region/'])
    % Define the data base directory 
    
    % automatic detection of datashare location:
    % After mounting the diedrichsen datashare on a mac computer.
    if isfolder("/Volumes/diedrichsen_data$//data/SensoriMotorPrediction/smp1")
        workdir = "/Volumes/diedrichsen_data$//data/SensoriMotorPrediction/smp1";
    elseif isfolder("/cifs/diedrichsen/data/SensoriMotorPrediction/smp1")
        workdir = "/cifs/diedrichsen/data/SensoriMotorPrediction/smp1";
    else
        fprintf('Workdir not found. Mount or connect to server and try again.');
    end
    
    baseDir         = (sprintf('%s/',workdir));                            % Base directory of the project
    bidsDir        = 'BIDS';                                              % Raw data post AutoBids conversion
    behavDir = 'behavioural';       
    imagingRawDir   = 'imaging_data_raw';                                  % Temporary directory for raw functional data
    imagingDir      = 'imaging_data';                                      % Preprocesses functional data
    anatomicalDir   = 'anatomicals';                                       % Preprocessed anatomical data (LPI + center AC + segemnt)
    fmapDir         = 'fieldmaps';                                         % Fieldmap dir after moving from BIDS and SPM make fieldmap
    glmEstDir       = 'glm';
    regDir          = 'ROI';
    suitDir = 'suit';
    wbDir   = 'surfaceWB';
    numDummys       = 5;                                                   % number of dummy scans at the beginning of each run
    
    %% subject info
    
    % Read info from participants .tsv file 
    % pinfo = dload(fullfile(baseDir,'participants.tsv'));
    pinfo = dload(fullfile(baseDir,'participants.tsv'));
    
    %% MAIN OPERATION 
    switch(what)
        
        case 'BIDS:move_unzip_raw_anat'
            % Moves, unzips and renames anatomical images from BIDS
            % directory to anatomicalDir. After you run this function you 
            % will find a <subj_id>_anatomical_raw.nii file in the
            % <project_id>/anatomicals/<subj_id>/ directory.
                        
            % handling input args:
            sn = [];
            vararginoptions(varargin,{'sn'})
            if isempty(sn)
                error('BIDS:move_unzip_raw_anat -> ''sn'' must be passed to this function.')
            end
            subj_id = pinfo.subj_id{pinfo.sn==sn};
    
            % path to the subj anat data:
            anat_raw_path = fullfile(baseDir,bidsDir,sprintf('subj%.03d',sn), 'anat',[pinfo.AnatRawName{pinfo.sn==sn}, '.nii.gz']);
    
            % destination path:
            output_folder = fullfile(baseDir,anatomicalDir,subj_id);
            output_file = fullfile(output_folder,[subj_id '_anatomical_raw.nii.gz']);
    
            if ~exist(output_folder,"dir")
                mkdir(output_folder);
            end
    
            % copy file to destination:
            [status,msg] = copyfile(anat_raw_path,output_file);
            if ~status  
                error('ANAT:move_anatomical -> subj %d raw anatmoical was not moved from BIDS to the destenation:\n%s',sn,msg)
            end
    
            % unzip the .gz files to make usable for SPM:
            gunzip(output_file);
    
            % delete the compressed file:
            delete(output_file);
    
        case 'BIDS:move_unzip_raw_func'
            % Moves, unzips and renames raw functional (BOLD) images from 
            % BIDS directory. After you run this function you will find
            % nRuns Nifti files named <subj_id>_run_XX.nii in the 
            % <project_id>/imaging_data_raw/<subj_id>/sess<N>/ directory.
            
            % handling input args:
            sn = [];
            vararginoptions(varargin,{'sn'})
            if isempty(sn)
                error('BIDS:move_unzip_raw_func -> ''sn'' must be passed to this function.')
            end
            subj_id = pinfo.subj_id{pinfo.sn==sn};

            % loop on sessions:
            for sess = 1:pinfo.numSess(pinfo.sn==sn)
                
                % pull list of runs from the participant.tsv:
                run_list = pinfo.(['runsSess', num2str(sess)]){pinfo.sn==sn};
                run_list = str2double(split(run_list,'.'));
    
                % loop on runs of sess:
                for i = 1:length(run_list)
                    
                    % pull functional raw name from the participant.tsv:
                    FuncRawName_tmp = [pinfo.(['FuncRawNameSess', num2str(sess)]){pinfo.sn==sn} '.nii.gz'];  
    
                    % add run number to the name of the file:
                    FuncRawName_tmp = replace(FuncRawName_tmp,'XX',sprintf('%.02d',i));
    
                    % path to the subj func data:
                    func_raw_path = fullfile(baseDir,bidsDir,sprintf('subj%.02d',sn),'func',FuncRawName_tmp);
            
                    % destination path:
                    output_folder = fullfile(baseDir,imagingRawDir,subj_id,['sess' num2str(sess)]);
                    output_file = fullfile(output_folder,[subj_id sprintf('_run_%.02d.nii.gz',run_list(i))]);
                    
                    if ~exist(output_folder,"dir")
                        mkdir(output_folder);
                    end
                    
                    % copy file to destination:
                    [status,msg] = copyfile(func_raw_path,output_file);
                    if ~status  
                        error('FUNC:move_unzip_raw_func -> subj %d raw functional (BOLD) was not moved from BIDS to the destenation:\n%s',sn,msg)
                    end
            
                    % unzip the .gz files to make usable for SPM:
                    gunzip(output_file);
            
                    % delete the compressed file:
                    delete(output_file);
                end
            end    
    
        case 'BIDS:move_unzip_raw_fmap'
            % Moves, unzips and renames raw fmap images from BIDS
            % directory. After you run this function you will find
            % two files named <subj_id>_phase.nii and 
            % <subj_id>_magnitude.nii in the 
            % <project_id>/fieldmaps/<subj_id>/sess<N>/ directory.
            
            % handling input args:
            sn = [];
            vararginoptions(varargin,{'sn'})
            if isempty(sn)
                error('BIDS:move_unzip_raw_fmap -> ''sn'' must be passed to this function.')
            end
            subj_id = pinfo.subj_id{pinfo.sn==sn};
            
            % loop on sessions:
            for sess = 1:pinfo.numSess(pinfo.sn==sn)
                % pull fmap raw names from the participant.tsv:
                fmapMagnitudeName_tmp = pinfo.(['fmapMagnitudeNameSess', num2str(sess)]){pinfo.sn==sn};
                magnitude = [fmapMagnitudeName_tmp '.nii.gz'];
                
                fmapPhaseName_tmp = pinfo.(['fmapPhaseNameSess', num2str(sess)]){pinfo.sn==sn};
                phase = [fmapPhaseName_tmp '.nii.gz'];
    
                % path to the subj fmap data:
                magnitude_path = fullfile(baseDir,bidsDir,sprintf('subj%.02d',sn),'fmap',magnitude);
                phase_path = fullfile(baseDir,bidsDir,sprintf('subj%.02d',sn),'fmap',phase);
        
                % destination path:
                output_folder = fullfile(baseDir,fmapDir,subj_id,['sess' num2str(sess)]);
                output_magnitude = fullfile(output_folder,[subj_id '_magnitude.nii.gz']);
                output_phase = fullfile(output_folder,[subj_id '_phase.nii.gz']);
                
                if ~exist(output_folder,"dir")
                    mkdir(output_folder);
                end
                
                % copy magnitude to destination:
                [status,msg] = copyfile(magnitude_path,output_magnitude);
                if ~status  
                    error('BIDS:move_unzip_raw_fmap -> subj %d, fmap magnitude was not moved from BIDS to the destenation:\n%s',sn,msg)
                end
                % unzip the .gz files to make usable for SPM:
                gunzip(output_magnitude);
        
                % delete the compressed file:
                delete(output_magnitude);
    
                % copy phase to destination:
                [status,msg] = copyfile(phase_path,output_phase);
                if ~status  
                    error('BIDS:move_unzip_raw_fmap -> subj %d, fmap phase was not moved from BIDS to the destenation:\n%s',sn,msg)
                end
                % unzip the .gz files to make usable for SPM:
                gunzip(output_phase);
        
                % delete the compressed file:
                delete(output_phase);
            end 
    
        case 'ANAT:reslice_LPI'          
            % Reslice anatomical image within LPI coordinate systems. This
            % function creates a <subj_id>_anatomical.nii file in the
            % <project_id>/anatomicals/<subj_id>/ directory.


            % handling input args:
            sn = [];
            vararginoptions(varargin,{'sn'})
            if isempty(sn)
                error('ANAT:reslice_LPI -> ''sn'' must be passed to this function.')
            end
            subj_id = char(pinfo.subj_id(pinfo.sn==sn));
            
            % (1) Reslice anatomical image to set it within LPI co-ordinate frames
            source = fullfile(baseDir,anatomicalDir,subj_id,[subj_id '_anatomical_raw.nii']);
            dest = fullfile(baseDir,anatomicalDir,subj_id,[subj_id '_anatomical.nii']);
            spmj_reslice_LPI(source,'name', dest);
            
            % WE DONT NEED THIS!!!!
            % % (2) In the resliced image, set translation to zero (why?)
            % V               = spm_vol(dest);
            % dat             = spm_read_vols(V);
            % V.mat(1:3,4)    = [0 0 0];
            % spm_write_vol(V,dat);
    
    
        case 'ANAT:centre_AC'            
            % Description:
            % Recenters the anatomical data to the Anterior Commissure
            % coordiantes. Doing that, the [0,0,0] coordinate of subject's
            % anatomical image will be the Anterior Commissure.
    
            % You should manually find the voxel coordinates 
            % (1-based index --> fslyes starts from 0) AC for each from 
            % their anatomical scans and add it to the participants.tsv 
            % file under the loc_ACx loc_ACy loc_ACz columns.
    
            % This function runs for all subjects and sessions.
    
            % handling input args:
            sn = [];
            vararginoptions(varargin,{'sn'})
            if isempty(sn)
                error('ANAT:centre_AC -> ''sn'' must be passed to this function.')
            end
            subj_id = char(pinfo.subj_id(pinfo.sn==sn));
            
            % path to the raw anatomical:
            anat_raw_file = fullfile(baseDir,anatomicalDir,subj_id,[subj_id '_anatomical.nii']);
            if ~exist(anat_raw_file,"file")
                error('ANAT:centre_AC -> file %s was not found.',anat_raw_file)
            end
            
            % Get header info for the image:
            V = spm_vol(anat_raw_file);
            % Read the volume:
            dat = spm_read_vols(V);
            
            % changing the transform matrix translations to put AC near [0,0,0]
            % coordinates:
            R = V.mat(1:3,1:3);
            AC = [pinfo.locACx(pinfo.sn==sn),pinfo.locACy(pinfo.sn==sn),pinfo.locACz(pinfo.sn==sn)]';
            t = -1 * R * AC;
            V.mat(1:3,4) = t;
            sprintf('ACx: %d, ACy: %d, ACz: %d', pinfo.locACx(pinfo.sn==sn), pinfo.locACy(pinfo.sn==sn), pinfo.locACz(pinfo.sn==sn))
    
            % writing the image with the changed header:
            spm_write_vol(V,dat);
    
        
        case 'ANAT:segmentation'
            % Segmentation + Normalization. Manually check results when
            % done. This step creates five files named 
            % c1<subj_id>_anatomical.nii, c2<subj_id>_anatomical.nii, 
            % c3<subj_id>_anatomical.nii, c4<subj_id>_anatomical.nii, 
            % c5<subj_id>_anatomical.nii, in the 
            % <project_id>/anatomicals/<subj_id>/ directory. Each of these
            % files contains a segment (e.g., white matter, grey matter) of
            % the centered anatomical image.
    
            % handling input args:
            sn = [];
            vararginoptions(varargin,{'sn'})
            if isempty(sn)
                error('ANAT:segmentation -> ''sn'' must be passed to this function.')
            end
            subj_id = pinfo.subj_id{pinfo.sn==sn};

            anat_path = fullfile(baseDir,anatomicalDir,subj_id,[subj_id '_anatomical.nii,1']);

            % spmj_segmentation(anat_path);
            SPMhome=fileparts(which('spm.m'));
            J=[];
            % for s=sn WE DONT NEED THIS FOR LOOP 
            J.channel.vols = {fullfile(baseDir,anatomicalDir,subj_id,[subj_id '_anatomical.nii,1'])};
            J.channel.biasreg = 0.001;
            J.channel.biasfwhm = 60;
            J.channel.write = [0 0];
            J.tissue(1).tpm = {fullfile(SPMhome,'tpm/TPM.nii,1')};
            J.tissue(1).ngaus = 1;
            J.tissue(1).native = [1 0];
            J.tissue(1).warped = [0 0];
            J.tissue(2).tpm = {fullfile(SPMhome,'tpm/TPM.nii,2')};
            J.tissue(2).ngaus = 1;
            J.tissue(2).native = [1 0];
            J.tissue(2).warped = [0 0];
            J.tissue(3).tpm = {fullfile(SPMhome,'tpm/TPM.nii,3')};
            J.tissue(3).ngaus = 2;
            J.tissue(3).native = [1 0];
            J.tissue(3).warped = [0 0];
            J.tissue(4).tpm = {fullfile(SPMhome,'tpm/TPM.nii,4')};
            J.tissue(4).ngaus = 3;
            J.tissue(4).native = [1 0];
            J.tissue(4).warped = [0 0];
            J.tissue(5).tpm = {fullfile(SPMhome,'tpm/TPM.nii,5')};
            J.tissue(5).ngaus = 4;
            J.tissue(5).native = [1 0];
            J.tissue(5).warped = [0 0];
            J.tissue(6).tpm = {fullfile(SPMhome,'tpm/TPM.nii,6')};
            J.tissue(6).ngaus = 2;
            J.tissue(6).native = [0 0];
            J.tissue(6).warped = [0 0];
            J.warp.mrf = 1;
            J.warp.cleanup = 1;
            J.warp.reg = [0 0.001 0.5 0.05 0.2];
            J.warp.affreg = 'mni';
            J.warp.fwhm = 0;
            J.warp.samp = 3;
            J.warp.write = [0 0];
            matlabbatch{1}.spm.spatial.preproc=J;
            spm_jobman('run',matlabbatch);
%             end
    
        case 'FUNC:make_fmap'                
            % Description:
            % Generates VDM files from the presubtracted phase & magnitude
            % images acquired from the field map sequence. Also, just as a
            % quality control this function creates unwarped EPIs from the
            % functional data with the prefix 'u' for each run. After you
            % run this step you will have:
            % - nBlock vdm5*.nii files in the 
            %   <project_id>/fieldmaps/<subj_id>/sess<N> directory: these 
            %   are the voxel displacement  maps registered to the first 
            %   image (if you set 'image', 1 below) of each functional run.
            % - 
            % - nBlock wfmag*.nii forward warped fieldmap magnitude image used for 
            %   coregistration (?)
            % - 

            
            % handling input args:
            sn = [];
            vararginoptions(varargin,{'sn'})
            if isempty(sn)
                error('FUNC:make_fmap -> ''sn'' must be passed to this function.')
            end
            subj_id = pinfo.subj_id{pinfo.sn==sn};
            
            % loop on sessions:
            for sess = 1:pinfo.numSess(pinfo.sn==sn)
                % Prefix of the functional files:
                prefixepi  = '';
    
                [et1, et2, tert] = spmj_et1_et2_tert(baseDir, subj_id, sn);
    
                % pull list of runs from the participant.tsv:
                run_list = pinfo.(['runsSess', num2str(sess)]){pinfo.sn==sn};
                run_list = str2double(split(run_list,'.'));
                run_list = arrayfun(@(x) sprintf('%02d', x), run_list, 'UniformOutput', false);
    
                subfolderFieldmap = sprintf('sess%d',sess);
                % function to create the makefieldmap job and passing it to the SPM
                % job manager:
                spmj_makefieldmap(baseDir,subj_id,run_list, ...
                                  'et1', et1, ...
                                  'et2', et2, ...
                                  'tert', tert, ...
                                  'image', 1, ... % remove numDummys?
                                  'prefix',prefixepi, ...
                                  'rawdataDir',fullfile(baseDir,imagingRawDir,subj_id,sprintf('sess%d',sess)), ...
                                  'subfolderFieldmap',subfolderFieldmap);
            end
    
        case 'FUNC:realign_unwarp'      
            % Do spm_realign_unwarp
            startTR         = 1;                                                   % first TR after the dummy scans
            
            % handling input args:
            sn = [];
            rtm = 0;
            vararginoptions(varargin,{'sn','rtm'})
            if isempty(sn)
                error('FUNC:make_fmap -> ''sn'' must be passed to this function.')
            end
            subj_id = pinfo.subj_id{pinfo.sn==sn};
    
            % loop on sessions:
            for sess = 1:pinfo.numSess(pinfo.sn==sn)
                % Prefix of the functional files (default 'a')
                prefixepi  = '';
    
                % pull list of runs from the participant.tsv:
                run_list = pinfo.(['runsSess', num2str(sess)]){pinfo.sn==sn};
                run_list = str2double(split(run_list,'.'));
                run_list = arrayfun(@(x) sprintf('%02d', x), run_list, 'UniformOutput', false);
                
                subfolderFieldmap = sprintf('sess%d',sess);
                spmj_realign_unwarp(baseDir,subj_id,run_list, startTR, inf, ...
                                    'prefix',prefixepi,...
                                    'rawdataDir',fullfile(baseDir,imagingRawDir,subj_id,sprintf('sess%d',sess)),...
                                    'subfolderFieldmap',subfolderFieldmap,...
                                    'rtm',rtm);
            end
    
        % case 'FUNC:realign'          
        %     % realign functional images
        %     % SPM realigns all volumes to the mean volume of first run
        % 
        %     for s = sn
        %         spm_jobman('initcfg')
        % 
        %         data = {};
        %             % initialize data cell array which will contain file names for runs/TR images
        %             func_ses_subj_dir = fullfile(imaging_dir ,subj_id);
        % 
        %             for r = runs
        %                 % Obtain the number of TRs for the current run
        %                 for j = 1:numTRs - numDummys
        %                     data{r}{j,1} = fullfile(func_ses_subj_dir,sprintf('%s_run-%02d.nii,%d', subj_id, r,j));
        %                 end % j (TRs/images)
        %             end % r (runs)            
        %         spmj_realign(data);
        %         fprintf('- runs realigned for %s  ',subj_id);
        % 
        %     end % s (sn)

        case 'FUNC:inspect_realign_parameters'
            % looks for motion correction logs into imaging_data, needs to
            % be run after realigned images are moved there from
            % imaging_data_raw

            % handling input args:
            sn = [];
            sess = [];   
            vararginoptions(varargin,{'sn','sess'})
            if isempty(sn)
                error('FUNC:inspect_realign_parameters -> ''sn'' must be passed to this function.')
            end

            if isempty(sess)
                error('FUNC:inspect_realign_parameters -> ''sess'' must be passed to this function.')
            end

            subj_id = pinfo.subj_id{pinfo.sn==sn};
            
            % pull list of runs from the participant.tsv:
            run_list = pinfo.(['runsSess', num2str(sess)]){pinfo.sn==sn};
            run_list = str2double(split(run_list,'.'));
            run_list = arrayfun(@(x) sprintf('%02d', x), run_list, 'UniformOutput', false);

            file_list = cellfun(@(run) fullfile(baseDir,imagingDir, subj_id,...
                sprintf('sess%d',sess),['rp_', subj_id,...
                '_run_', run, '.txt']), run_list, 'UniformOutput', false);

            smpj_plot_mov_corr(file_list)
            
        case 'FUNC:move_realigned_images'          
            % Move images created by realign(+unwarp) into imaging_data
            
            % handling input args:
            sn = [];
            prefix = 'u';   % prefix of the 4D images after realign(+unwarp)
            rtm = 0;        % realign_unwarp registered to the first volume (0) or mean image (1).
            vararginoptions(varargin,{'sn','prefix','rtm'})
            if isempty(sn)
                error('FUNC:move_realigned_images -> ''sn'' must be passed to this function.')
            end

            subj_id = pinfo.subj_id{pinfo.sn==sn};
    
            % loop on sessions:
            for sess = 1:pinfo.numSess(pinfo.sn==sn)

                % pull list of runs from the participant.tsv:
                run_list = pinfo.(['runsSess', num2str(sess)]){pinfo.sn==sn};
                run_list = str2double(split(run_list,'.'));
                run_list = arrayfun(@(x) sprintf('%02d', x), run_list, 'UniformOutput', false);
                
                % loop on runs of the session:
                for r = 1:length(run_list)
                    % realigned (and unwarped) images names:
                    file_name = [prefix, char(pinfo.subj_id(pinfo.sn==sn)), '_run_', run_list{r}, '.nii'];
                    source = fullfile(baseDir,imagingRawDir,char(pinfo.subj_id(pinfo.sn==sn)),sprintf('sess%d',sess),file_name);
                    dest = fullfile(baseDir,imagingDir,char(pinfo.subj_id(pinfo.sn==sn)),sprintf('sess%d',sess));
                    if ~exist(dest,'dir')
                        mkdir(dest)
                    end

                    file_name = file_name(length(prefix) + 1:end); % skip prefix in realigned (and unwarped) files
                    dest = fullfile(baseDir,imagingDir,char(pinfo.subj_id(pinfo.sn==sn)),sprintf('sess%d',sess),file_name);
                    % move to destination:
                    [status,msg] = movefile(source,dest);
                    if ~status  
                        error('BIDS:move_realigned_images -> %s',msg)
                    end
    
                    % realign parameters names:
                    source = fullfile(baseDir,imagingRawDir,subj_id,sprintf('sess%d',sess),['rp_', subj_id, '_run_', run_list{r}, '.txt']);
                    dest = fullfile(baseDir,imagingDir,subj_id,sprintf('sess%d',sess),['rp_', subj_id, '_run_', run_list{r}, '.txt']);
                    % move to destination:
                    [status,msg] = movefile(source,dest);
                    if ~status  
                        error('BIDS:move_realigned_images -> %s',msg)
                    end
                end
                
                % mean epi name - the generated file name will be different for
                % rtm=0 and rtm=1. Extra note: rtm is an option in
                % realign_unwarp function. Refer to spmj_realign_unwarp().
                if rtm==0   % if registered to first volume of each run:
                    source = fullfile(baseDir,imagingRawDir,subj_id,sprintf('sess%d',sess),['mean', prefix, subj_id, '_run_', run_list{1}, '.nii']);
                    dest = fullfile(baseDir,imagingDir,subj_id,sprintf('sess%d',sess),['mean', prefix, subj_id, '_run_', run_list{1}, '.nii']);
                else        % if registered to mean image of each run:
                    source = fullfile(baseDir,imagingRawDir,subj_id,sprintf('sess%d',sess),[prefix, 'meanepi_', subj_id, '.nii']);
                    dest = fullfile(baseDir,imagingDir,subj_id,sprintf('sess%d',sess),[prefix, 'meanepi_', subj_id, '.nii']);
                end
                % move to destination:
                [status,msg] = movefile(source,dest);
                if ~status  
                    error('BIDS:move_realigned_images -> %s',msg)
                end
            end
        
        case 'FUNC:meanimage_bias_correction'                                         
            % EPI images often contain smooth artifacts caused by MRI
            % physics which make the intensity of signal from the same
            % tissue (e.g., grey matter, white matter) non-uniform. This
            % step perform bias correction and created an image where the
            % signal from each tissue type is more uniform. This image is
            % then co-registered to the anatomical image. Bias correction
            % help make co-registration more accurate. If the realignment
            % was done with respect to the first volume of each run of each 
            % session, the mean image will be calculated on the first run of
            % each session and will be called 'meanu*_run_01.nii' ('mean' 
            % indicates the image is average of the volumes and 'u' indicates
            % it's unwarped). Therefore, we do the bias correction on this 
            % file. But if you do the realignment to the mean epi of every run, the
            % generated mean file will be named 'umeanepi_*' and we do the bias
            % correction on this file. In addition, this step generates
            % five tissue probability maps (c1-5) for grey matter, white
            % matter, csf, bone and soft tissue.
    
            % handling input args:
            sn = [];
            prefix = 'u';   % prefix of the 4D images after realign(+unwarp)
            rtm = 0;        % realign_unwarp registered to the first volume (0) or mean image (1).
            vararginoptions(varargin,{'sn','prefix','rtm'})
            if isempty(sn)
                error('FUNC:meanimage_bias_correction -> ''sn'' must be passed to this function.')
            end

            subj_id = pinfo.subj_id{pinfo.sn==sn};
    
            % loop on sessions:
            for sess = 1:pinfo.numSess(pinfo.sn==sn)

                % pull list of runs from the participant.tsv:
                run_list = pinfo.(['runsSess', num2str(sess)]){pinfo.sn==sn};
                run_list = str2double(split(run_list,'.'));
                run_list = arrayfun(@(x) sprintf('%02d', x), run_list, 'UniformOutput', false);
    
                if rtm==0   % if registered to first volume of each run:
                    P{1} = fullfile(baseDir, imagingDir, subj_id, sprintf('sess%d',sess), ['mean', prefix,  subj_id, '_run_', run_list{1}, '.nii']);
                else        % if registered to mean image of each run:
                    P{1} = fullfile(baseDir, imagingDir, subj_id, sprintf('sess%d',sess), [prefix, 'meanepi_', subj_id, '.nii']);
                end
                spmj_bias_correct(P);
            end
    
        case 'FUNC:coreg'                                                      
            % coregister rbumean image to anatomical image for each session
            
            % handling input args:
            sn = [];
            prefix = 'u';   % prefix of the 4D images after realign(+unwarp)
            rtm = 0;        % realign_unwarp registered to the first volume (0) or mean image (1).
            vararginoptions(varargin,{'sn','prefix','rtm'})
            if isempty(sn)
                error('FUNC:coreg -> ''sn'' must be passed to this function.')
            end

            subj_id = pinfo.subj_id{pinfo.sn==sn};
            
            % loop on sessions:
            for sess = 1:pinfo.numSess(pinfo.sn==sn)
                % (1) Manually seed the functional/anatomical registration
                % - Open fsleyes
                % - Add anatomical image and b*mean*.nii (bias corrected mean) image to overlay
                % - click on the bias corrected mean image in the 'Overlay
                %   list' in the bottom left of the fsleyes window.
                %   list to highlight it.
                % - Open tools -> Nudge
                % - Manually adjust b*mean*.nii image to the anatomical by 
                %   changing the 6 paramters (tranlation xyz and rotation xyz) 
                %   and Do not change the scales! 
                % - When done, click apply and close the tool tab. Then to save
                %   the changes, click on the save icon next to the mean image 
                %   name in the 'Overlay list' and save the new image by adding
                %   'r' in the beginning of the name: rb*mean*.nii. If you don't
                %   set the format to be .nii, fsleyes automatically saves it as
                %   a .nii.gz so either set it or gunzip afterwards to make it
                %   compatible with SPM.
                
                % (2) Run automated co-registration to register bias-corrected meanimage to anatomical image
                
                % pull list of runs from the participant.tsv:
                run_list = pinfo.(['runsSess', num2str(sess)]){pinfo.sn==sn};
                run_list = str2double(split(run_list,'.'));
                run_list = arrayfun(@(x) sprintf('%02d', x), run_list, 'UniformOutput', false);
    
                if rtm==0   % if registered to first volume
                    mean_file_name = sprintf('mean%s%s_run_%s.nii', prefix, subj_id, run_list{1});
                else    % if registered to the mean image
                    mean_file_name = sprintf('rb%smeanepi_%s.nii', prefix, subj_id);
                end
                J.source = {fullfile(baseDir,imagingDir,subj_id,sprintf('sess%d',sess),mean_file_name)}; 
                J.ref = {fullfile(baseDir,anatomicalDir,subj_id,[subj_id, '_anatomical','.nii'])};
                J.other = {''};
                J.eoptions.cost_fun = 'nmi';
                J.eoptions.sep = [4 2];
                J.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
                J.eoptions.fwhm = [7 7];
                matlabbatch{1}.spm.spatial.coreg.estimate=J;
                spm_jobman('run',matlabbatch);
                
                % (3) Check alignment manually by using fsleyes similar to step
                % one.
            end
    
        case 'FUNC:make_samealign'
            % align to registered bias corrected mean image of each session
            % (rb*mean*.nii). Alignment happens only by changing the transform
            % matrix in the header files of the functional 4D .nii files to the
            % transform matrix that aligns them to anatomical. The reason that
            % it works is: 1) in the realignment (+unwarping) process, we have
            % registered every single volume of every single run to the first
            % volume of the first run of the session. 2) In the same step, for
            % each session, a mean functional image (meanepi*.nii or meanu*.nii
            % based on the rtm option) was generated. This mean image is alread
            % in the space of all the functional volumes. Later we coregister
            % this image to the anatomical space. Therefore, if we change the
            % transformation matrices of all the functional volumes to the
            % transform matrix of the coregistered image, they will all
            % tranform into the anatomical coordinates space.
    
            % handling input args:
            sn = [];
            prefix = 'u';   % prefix of the 4D images after realign(+unwarp)
            rtm = 0;        % realign_unwarp registered to the first volume (0) or mean image (1).
            vararginoptions(varargin,{'sn','prefix','rtm'})
            if isempty(sn)
                error('FUNC:make_samealign -> ''sn'' must be passed to this function.')
            end

            subj_id = pinfo.subj_id{pinfo.sn==sn};
    
            % loop on sessions:
            for sess = 1:pinfo.numSess(pinfo.sn==sn)

                % pull list of runs from the participant.tsv:
                run_list = pinfo.(['runsSess', num2str(sess)]){pinfo.sn==sn};
                run_list = str2double(split(run_list,'.'));
                run_list = arrayfun(@(x) sprintf('%02d', x), run_list, 'UniformOutput', false);
                
                % select the reference image:
                if rtm==0
                    P{1} = fullfile(baseDir,imagingDir,subj_id,sprintf('sess%d',sess),['mean' prefix subj_id '_run_' run_list{1} '.nii']);
                else
                    P{1} = fullfile(baseDir,imagingDir,subj_id,sprintf('sess%d',sess),['rb' prefix 'meanepi_' subj_id '.nii']);
                end
    
                % select images to be realigned:
                Q = {};
                for r = 1:length(run_list)
                    for i = 1:pinfo.numTR
                         Q{end+1} = fullfile(baseDir,imagingDir,subj_id,sprintf('sess%d',sess),[ subj_id '_run_' run_list{r} '.nii,' num2str(i)]);
                    end
                end
    
                spmj_makesamealign_nifti(char(P),char(Q));
            end
        
        case 'FUNC:make_maskImage'       
            % Make mask images (noskull and gray_only) for 1st level glm
            
            % handling input args:
            sn = [];
            prefix = 'u';   % prefix of the 4D images after realign(+unwarp)
            rtm = 0;        % realign_unwarp registered to the first volume (0) or mean image (1).
            vararginoptions(varargin,{'sn','prefix','rtm'})
            if isempty(sn)
                error('FUNC:make_maskImage -> ''sn'' must be passed to this function.')
            end

            subj_id = pinfo.subj_id{pinfo.sn==sn};
    
            % loop on sessions:
            for sess = 1:pinfo.numSess(pinfo.sn==sn)

                % pull list of runs from the participant.tsv:
                run_list = pinfo.(['runsSess', num2str(sess)]){pinfo.sn==sn};
                run_list = str2double(split(run_list,'.'));
                run_list = arrayfun(@(x) sprintf('%02d', x), run_list, 'UniformOutput', false);
            
                % bias corrected mean epi image:
                if rtm==0
                    nam{1} = fullfile(baseDir,imagingDir,subj_id,sprintf('sess%d',sess),['mean' prefix subj_id '_run_' run_list{1} '.nii']);
                else
                    nam{1} = fullfile(baseDir,imagingDir,subj_id,sprintf('sess%d',sess),['rb' prefix 'meanepi_' subj_id '.nii']);
                end
                nam{2}  = fullfile(baseDir,anatomicalDir,subj_id,['c1',subj_id, '_anatomical','.nii']);
                nam{3}  = fullfile(baseDir,anatomicalDir,subj_id,['c2',subj_id, '_anatomical','.nii']);
                nam{4}  = fullfile(baseDir,anatomicalDir,subj_id,['c3',subj_id, '_anatomical','.nii']);
                spm_imcalc(nam, fullfile(baseDir,imagingDir,subj_id,sprintf('sess%d',sess), 'rmask_noskull.nii'), 'i1>1 & (i2+i3+i4)>0.2')
                
                source = fullfile(baseDir,imagingDir,subj_id,sprintf('sess%d',sess), 'rmask_noskull.nii'); % does this need to have some flag for session?
                dest = fullfile(baseDir,anatomicalDir,subj_id,'rmask_noskull.nii');
                movefile(source,dest);
                
                % gray matter mask for covariance estimation
                % ------------------------------------------
                nam={};
                % nam{1}  = fullfile(imagingDir,subj_id{sn}, 'sess1', ['rb' prefix 'meanepi_' subj_id{sn} '.nii']);

                % IS THIS CHANGE CORRECT??
                % nam{1}  = fullfile(baseDir,imagingDir,char(pinfo.subj_id(pinfo.sn==sn)),sprintf('sess%d',sess), ['rb' prefix 'meanepi_' char(pinfo.subj_id(pinfo.sn==sn)) '.nii']);
                % bias corrected mean epi image:
                if rtm==0
                    nam{1} = fullfile(baseDir,imagingDir,subj_id,sprintf('sess%d',sess),['mean' prefix subj_id '_run_' run_list{1} '.nii']);
                else
                    nam{1} = fullfile(baseDir,imagingDir,subj_id,sprintf('sess%d',sess),['rb' prefix 'meanepi_' subj_id '.nii']);
                end

                nam{2}  = fullfile(baseDir,anatomicalDir,subj_id,['c1',subj_id, '_anatomical','.nii']);
                spm_imcalc(nam, fullfile(baseDir,imagingDir,subj_id,sprintf('sess%d',sess), 'rmask_gray.nii'), 'i1>1 & i2>0.4')
                
                source = fullfile(baseDir,imagingDir,subj_id,sprintf('sess%d',sess), 'rmask_gray.nii');
                dest = fullfile(baseDir,anatomicalDir,subj_id,'rmask_gray.nii');
                movefile(source,dest);
            end

        case 'GLM:make_events'

            sn = [];
            glm = [];
            vararginoptions(varargin,{'sn', 'glm'})

            subj_id = pinfo.subj_id{pinfo.sn==sn};
            
            operation  = sprintf('GLM:make_glm%d', glm);
            
            events = smp1_imana(operation, 'sn', sn);
            
            %% export
            output_folder = fullfile(baseDir, behavDir, subj_id);
            writetable(events, fullfile(output_folder, 'glm6_events.tsv'), 'FileType', 'text', 'Delimiter','\t')

        case 'GLM:calculate_VIF'

            sn = [];
            glm = [];
            vararginoptions(varargin,{'sn', 'glm'})

            subj_id = pinfo.subj_id{pinfo.sn==sn};

            SPM = load(fullfile(baseDir,[glmEstDir num2str(glm)],subj_id, 'SPM.mat')); SPM = SPM.SPM;

            X=(SPM.xX.X(:,SPM.xX.iC)); % Assuming 'X' is the field holding the design matrix
            
            % % Number of predictors
            % [n, p] = size(X);
            
            % Initialize VIF array
            vif_values = zeros(length(SPM.Sess), length(SPM.Sess(1).col));

            for s = 1:length(SPM.Sess)
                idx = 1;
                for i = SPM.Sess(s).col
                    % Separate the ith predictor
                    Xi = X(:, i);
                    % Remaining predictors
                    other_predictors = X(:, setdiff(SPM.Sess(s).col, i));
                    
                    % Fit a regression model of Xi on the other predictors
                    model = fitlm(other_predictors, Xi);
                    
                    % Get R-squared value
                    R_squared = model.Rsquared.Ordinary;
                    
                    % Compute VIF
                    vif_values(s, idx) = 1 / (1 - R_squared);
                    idx = idx + 1;
                end

            end

            figure
            bar([SPM.Sess(1).U.name], mean(vif_values, 1))
            set(groot, 'defaultTextInterpreter', 'none');
            hold on
            errorbar(categorical([SPM.Sess(1).U.name]), mean(vif_values, 1), std(vif_values, 1), 'k', 'linestyle', 'none')
             
            % Display VIF values
            vif_table = array2table( vif_values, 'VariableNames', [SPM.Sess(1).U.name]);
            disp(vif_table);


        case 'GLM:make_glm1'
            
            sn = [];
            vararginoptions(varargin,{'sn'})

            subj_id = pinfo.subj_id{pinfo.sn==sn};

            D = dload(fullfile(baseDir, behavDir, subj_id, ['smp1_' subj_id(5:end) '.dat']));

            go = strcmp(D.GoNogo, "go");
            
            %% execution
            exec.BN = D.BN(go);
            exec.TN = D.TN(go);
            exec.cue = D.cue(go);
            exec.stimFinger = D.stimFinger(go);
            exec.Onset = D.startTimeReal(go) + D.baselineWait(go) + D.planTime(go);
            exec.Duration = D.execMaxTime(go);
            
            for ntrial = 1:length(exec.BN)
            
                switch exec.cue(ntrial)
                    case 39
                        cue = 'cue100';
                    case 21
                        cue = 'cue75';
                    case 44
                        cue = 'cue50';
                    case 12
                        cue = 'cue25';
                    case 93
                        cue = 'cue0';
                end
            
                switch exec.stimFinger(ntrial)
                    case 91999
                        stimFinger = 'index';
                    case 99919
                        stimFinger = 'ring';
                end
                
                exec.eventtype{ntrial, 1} = [cue '_' stimFinger '_exec'];
                exec.cue_id{ntrial, 1} = cue;
                exec.stimFinger_id{ntrial, 1} = stimFinger;
                exec.epoch{ntrial, 1} = 'exec';
                exec.instruction{ntrial, 1} = 'go';
                
            end
            
            %% plan go
            planGo.BN = D.BN(go);
            planGo.TN = D.TN(go);
            planGo.cue = D.cue(go);
            planGo.stimFinger = D.stimFinger(go);
            planGo.Onset = D.startTimeReal(go) + D.baselineWait(go);
            planGo.Duration = D.planTime(go);
            
            for ntrial = 1:length(planGo.BN)
            
                switch planGo.cue(ntrial)
                    case 39
                        cue = 'cue100';
                    case 21
                        cue = 'cue75';
                    case 44
                        cue = 'cue50';
                    case 12
                        cue = 'cue25';
                    case 93
                        cue = 'cue0';
                end
            
                switch planGo.stimFinger(ntrial)
                    case 91999
                        stimFinger = 'index';
                    case 99919
                        stimFinger = 'ring';
                end
                
                planGo.eventtype{ntrial, 1} = [cue '_' stimFinger '_plan_go'];
                planGo.cue_id{ntrial, 1} = cue;
                planGo.stimFinger_id{ntrial, 1} = stimFinger;
                planGo.epoch{ntrial, 1} = 'plan';
                planGo.instruction{ntrial, 1} = 'go';
                
            end
            
            %% plan nogo
            planNoGo.BN = D.BN(~go);
            planNoGo.TN = D.TN(~go);
            planNoGo.cue = D.cue(~go);
            planNoGo.stimFinger = D.stimFinger(~go);
            planNoGo.Onset = D.startTimeReal(~go) + D.baselineWait(~go);
            planNoGo.Duration = D.planTime(~go);
            
            for ntrial = 1:length(planNoGo.BN)
            
                switch planNoGo.cue(ntrial)
                    case 39
                        cue = 'cue100';
                    case 21
                        cue = 'cue75';
                    case 44
                        cue = 'cue50';
                    case 12
                        cue = 'cue25';
                    case 93
                        cue = 'cue0';
                end
                
                planNoGo.eventtype{ntrial, 1} = [cue '_plan_nogo'];
                planNoGo.cue_id{ntrial, 1} = cue;
                planNoGo.stimFinger_id{ntrial, 1} = 'none';
                planNoGo.epoch{ntrial, 1} = 'plan';
                planNoGo.instruction{ntrial, 1} = 'nogo';
                
            end
            
            %% make table
            
            exec = struct2table(exec);
            planGo = struct2table(planGo);
            planNoGo = struct2table(planNoGo);
            % rest = struct2table(rest);
            events = [exec; planGo; planNoGo];
            
            %% convert to secs
            events.Onset = events.Onset ./ 1000;
            events.Duration = events.Duration ./ 1000;
            
            %% export
             output_folder = fullfile(baseDir, behavDir, subj_id);
             if ~exist(output_folder, "dir")
                mkdir(output_folder);
             end
            writetable(events, fullfile(output_folder,  'glm1_events.tsv'), 'FileType', 'text', 'Delimiter','\t')       

        case 'GLM:make_glm2'
            
            sn = [];
            vararginoptions(varargin,{'sn'})

            subj_id = pinfo.subj_id{pinfo.sn==sn};

            D = dload(fullfile(baseDir, behavDir, subj_id, ['smp1_' subj_id(5:end) '.dat']));

            go = strcmp(D.GoNogo, "go");
            
            %% execution
            exec.BN = D.BN(go);
            exec.TN = D.TN(go);
            exec.cue = D.cue(go);
            exec.stimFinger = D.stimFinger(go);
            exec.Onset = D.startTimeReal(go) + D.baselineWait(go) + D.planTime(go);
            exec.Duration = D.execMaxTime(go);
            
            for ntrial = 1:length(exec.BN)
            
                switch exec.cue(ntrial)
                    case 39
                        cue = 'cue100';
                    case 21
                        cue = 'cue75';
                    case 44
                        cue = 'cue50';
                    case 12
                        cue = 'cue25';
                    case 93
                        cue = 'cue0';
                end
            
                switch exec.stimFinger(ntrial)
                    case 91999
                        stimFinger = 'index';
                    case 99919
                        stimFinger = 'ring';
                end
                
                exec.eventtype{ntrial, 1} = [cue '_' stimFinger '_exec'];
                exec.cue_id{ntrial, 1} = cue;
                exec.stimFinger_id{ntrial, 1} = stimFinger;
                exec.epoch{ntrial, 1} = 'exec';
                exec.instruction{ntrial, 1} = 'go';
                
            end
            
            %% plan nogo
            planNoGo.BN = D.BN(~go);
            planNoGo.TN = D.TN(~go);
            planNoGo.cue = D.cue(~go);
            planNoGo.stimFinger = D.stimFinger(~go);
            planNoGo.Onset = D.startTimeReal(~go) + D.baselineWait(~go);
            planNoGo.Duration = D.planTime(~go);
            
            for ntrial = 1:length(planNoGo.BN)
            
                switch planNoGo.cue(ntrial)
                    case 39
                        cue = 'cue100';
                    case 21
                        cue = 'cue75';
                    case 44
                        cue = 'cue50';
                    case 12
                        cue = 'cue25';
                    case 93
                        cue = 'cue0';
                end
                
                planNoGo.eventtype{ntrial, 1} = [cue '_plan_nogo'];
                planNoGo.cue_id{ntrial, 1} = cue;
                planNoGo.stimFinger_id{ntrial, 1} = 'none';
                planNoGo.epoch{ntrial, 1} = 'plan';
                planNoGo.instruction{ntrial, 1} = 'nogo';
                
            end
            
            %% make table
            
            exec = struct2table(exec);
            planNoGo = struct2table(planNoGo);
            % rest = struct2table(rest);
            events = [exec; planNoGo];
            
            %% convert to secs
            events.Onset = events.Onset ./ 1000;
            events.Duration = events.Duration ./ 1000;
            
            %% export
             output_folder = fullfile(baseDir, behavDir, subj_id);
             if ~exist(output_folder, "dir")
                mkdir(output_folder);
             end
            writetable(events, fullfile(output_folder,  'glm2_events.tsv'), 'FileType', 'text', 'Delimiter','\t')     

        case 'GLM:make_glm3'
            
            sn = [];
            vararginoptions(varargin,{'sn'})

            subj_id = pinfo.subj_id{pinfo.sn==sn};

            D = dload(fullfile(baseDir, behavDir, subj_id, ['smp1_' subj_id(5:end) '.dat']));

            go = strcmp(D.GoNogo, "go");
            
            %% execution
            exec.BN = D.BN(go);
            exec.TN = D.TN(go);
            exec.cue = D.cue(go);
            exec.stimFinger = D.stimFinger(go);
            exec.Onset = D.startTimeReal(go) + D.baselineWait(go) + D.planTime(go);
            exec.Duration = zeros(length(exec.BN), 1);
            
            for ntrial = 1:length(exec.BN)
            
                switch exec.cue(ntrial)
                    case 39
                        cue = 'cue100';
                    case 21
                        cue = 'cue75';
                    case 44
                        cue = 'cue50';
                    case 12
                        cue = 'cue25';
                    case 93
                        cue = 'cue0';
                end
            
                switch exec.stimFinger(ntrial)
                    case 91999
                        stimFinger = 'index';
                    case 99919
                        stimFinger = 'ring';
                end
                
                exec.eventtype{ntrial, 1} = [cue '_' stimFinger '_exec'];
                exec.cue_id{ntrial, 1} = cue;
                exec.stimFinger_id{ntrial, 1} = stimFinger;
                exec.epoch{ntrial, 1} = 'exec';
                exec.instruction{ntrial, 1} = 'go';
                
            end
            
            %% plan go
            planGo.BN = D.BN(go);
            planGo.TN = D.TN(go);
            planGo.cue = D.cue(go);
            planGo.stimFinger = D.stimFinger(go);
            planGo.Onset = D.startTimeReal(go) + D.baselineWait(go);
            planGo.Duration = zeros(length(planGo.BN), 1);
            
            for ntrial = 1:length(planGo.BN)
            
                switch planGo.cue(ntrial)
                    case 39
                        cue = 'cue100';
                    case 21
                        cue = 'cue75';
                    case 44
                        cue = 'cue50';
                    case 12
                        cue = 'cue25';
                    case 93
                        cue = 'cue0';
                end
            
                switch planGo.stimFinger(ntrial)
                    case 91999
                        stimFinger = 'index';
                    case 99919
                        stimFinger = 'ring';
                end
                
                planGo.eventtype{ntrial, 1} = [cue '_' stimFinger '_plan_go'];
                planGo.cue_id{ntrial, 1} = cue;
                planGo.stimFinger_id{ntrial, 1} = stimFinger;
                planGo.epoch{ntrial, 1} = 'plan';
                planGo.instruction{ntrial, 1} = 'go';
                
            end
            
            %% plan nogo
            planNoGo.BN = D.BN(~go);
            planNoGo.TN = D.TN(~go);
            planNoGo.cue = D.cue(~go);
            planNoGo.stimFinger = D.stimFinger(~go);
            planNoGo.Onset = D.startTimeReal(~go) + D.baselineWait(~go);
            planNoGo.Duration = zeros(length(planNoGo.BN), 1);
            
            for ntrial = 1:length(planNoGo.BN)
            
                switch planNoGo.cue(ntrial)
                    case 39
                        cue = 'cue100';
                    case 21
                        cue = 'cue75';
                    case 44
                        cue = 'cue50';
                    case 12
                        cue = 'cue25';
                    case 93
                        cue = 'cue0';
                end
                
                planNoGo.eventtype{ntrial, 1} = [cue '_plan_nogo'];
                planNoGo.cue_id{ntrial, 1} = cue;
                planNoGo.stimFinger_id{ntrial, 1} = 'none';
                planNoGo.epoch{ntrial, 1} = 'plan';
                planNoGo.instruction{ntrial, 1} = 'nogo';
                
            end
            
            %% make table
            
            exec = struct2table(exec);
            planGo = struct2table(planGo);
            planNoGo = struct2table(planNoGo);
            % rest = struct2table(rest);
            events = [exec; planGo; planNoGo];
            
            %% convert to secs
            events.Onset = events.Onset ./ 1000;
            events.Duration = events.Duration ./ 1000;
            
            %% export
             output_folder = fullfile(baseDir, behavDir, subj_id);
             if ~exist(output_folder, "dir")
                mkdir(output_folder);
             end
            writetable(events, fullfile(output_folder,  'glm3_events.tsv'), 'FileType', 'text', 'Delimiter','\t')  

        case 'GLM:make_glm4'
            
            sn = [];
            vararginoptions(varargin,{'sn'})

            subj_id = pinfo.subj_id{pinfo.sn==sn};

            D = dload(fullfile(baseDir, behavDir, subj_id, ['smp1_' subj_id(5:end) '.dat']));

            go = strcmp(D.GoNogo, "go");
            
            %% execution
            exec.BN = D.BN(go);
            exec.TN = D.TN(go);
            exec.cue = D.cue(go);
            exec.stimFinger = D.stimFinger(go);
            exec.Onset = D.startTimeReal(go) + D.baselineWait(go) + D.planTime(go);
            exec.Duration = zeros(length(exec.BN), 1);
            
            for ntrial = 1:length(exec.BN)
            
                switch exec.cue(ntrial)
                    case 39
                        cue = 'cue100';
                    case 21
                        cue = 'cue75';
                    case 44
                        cue = 'cue50';
                    case 12
                        cue = 'cue25';
                    case 93
                        cue = 'cue0';
                end
            
                switch exec.stimFinger(ntrial)
                    case 91999
                        stimFinger = 'index';
                    case 99919
                        stimFinger = 'ring';
                end
                
                exec.eventtype{ntrial, 1} = [cue '_' stimFinger '_exec'];
                exec.cue_id{ntrial, 1} = cue;
                exec.stimFinger_id{ntrial, 1} = stimFinger;
                exec.epoch{ntrial, 1} = 'exec';
                exec.instruction{ntrial, 1} = 'go';
                
            end
            
            %% planning
            plan.BN = D.BN;
            plan.TN = D.TN;
            plan.cue = D.cue;
            plan.stimFinger = D.stimFinger;
            plan.Onset = D.startTimeReal + D.baselineWait;
            plan.Duration = zeros(length(plan.BN), 1);
            plan.instruction = D.GoNogo;
            
            for ntrial = 1:length(plan.BN)
            
                switch plan.cue(ntrial)
                    case 39
                        cue = 'cue100';
                    case 21
                        cue = 'cue75';
                    case 44
                        cue = 'cue50';
                    case 12
                        cue = 'cue25';
                    case 93
                        cue = 'cue0';
                end
            
                switch plan.stimFinger(ntrial)
                    case 91999
                        stimFinger = 'index';
                    case 99919
                        stimFinger = 'ring';
                    case 99999
                        stimFinger = 'none';
                end
                
                plan.eventtype{ntrial, 1} = [cue '_' stimFinger '_plan_go'];
                plan.cue_id{ntrial, 1} = cue;
                plan.stimFinger_id{ntrial, 1} = stimFinger;
                plan.epoch{ntrial, 1} = 'plan';
                
            end
            
            %% make table
            
            exec = struct2table(exec);
            plan = struct2table(plan);
            % rest = struct2table(rest);
            events = [exec; plan];
            
            %% convert to secs
            events.Onset = events.Onset ./ 1000;
            events.Duration = events.Duration ./ 1000;
            
            %% export
             output_folder = fullfile(baseDir, behavDir, subj_id);
             if ~exist(output_folder, "dir")
                mkdir(output_folder);
             end
            writetable(events, fullfile(output_folder,  'glm4_events.tsv'), 'FileType', 'text', 'Delimiter','\t')  

        case 'GLM:make_glm5'
            
            sn = [];
            vararginoptions(varargin,{'sn'})

            subj_id = pinfo.subj_id{pinfo.sn==sn};

            D = dload(fullfile(baseDir, behavDir, subj_id, ['smp1_' subj_id(5:end) '.dat']));

            go = strcmp(D.GoNogo, "go");
            
            %% execution
            exec.BN = D.BN(go);
            exec.TN = D.TN(go);
            exec.cue = D.cue(go);
            exec.stimFinger = D.stimFinger(go);
            exec.Onset = D.startTimeReal(go) + D.baselineWait(go) + D.planTime(go);
            exec.Duration = zeros(length(exec.BN), 1);
            exec.eventtype = repmat({'exec'}, [length(exec.BN), 1]);
            
            %% planning
            plan.BN = D.BN;
            plan.TN = D.TN;
            plan.cue = D.cue;
            plan.stimFinger = D.stimFinger;
            plan.Onset = D.startTimeReal + D.baselineWait;
            plan.Duration = zeros(length(plan.BN), 1);
            plan.eventtype = repmat({'plan'}, [length(plan.BN), 1]);         
            
            %% make table
            
            exec = struct2table(exec);
            plan = struct2table(plan);
            % rest = struct2table(rest);
            events = [exec; plan];
            
            %% convert to secs
            events.Onset = events.Onset ./ 1000;
            events.Duration = events.Duration ./ 1000;
            
            
             varargout{1}= events;
            % writetable(events, fullfile(output_folder,  'glm5_events.tsv'), 'FileType', 'text', 'Delimiter','\t')  

        case 'GLM:make_glm6'
            
            sn = [];
            vararginoptions(varargin,{'sn'})

            subj_id = pinfo.subj_id{pinfo.sn==sn};

            D = dload(fullfile(baseDir, behavDir, subj_id, ['smp1_' subj_id(5:end) '.dat']));

            go = strcmp(D.GoNogo, "go");
            
            %% execution
            exec.BN = D.BN(go);
            exec.TN = D.TN(go);
            exec.cue = D.cue(go);
            exec.stimFinger = D.stimFinger(go);
            exec.Onset = D.startTimeReal(go) + D.baselineWait(go) + D.planTime(go);
            exec.Duration = zeros(length(exec.BN), 1);
            exec.eventtype = repmat({'exec'}, [length(exec.BN), 1]);
            
            %% planning (nogo)
            plan.BN = D.BN(~go);
            plan.TN = D.TN(~go);
            plan.cue = D.cue(~go);
            plan.stimFinger = D.stimFinger(~go);
            plan.Onset = D.startTimeReal(~go) + D.baselineWait(~go);
            plan.Duration = zeros(length(plan.BN), 1);
            plan.eventtype = repmat({'plan_nogo'}, [length(plan.BN), 1]);         
            
            %% make table
            
            exec = struct2table(exec);
            plan = struct2table(plan);
            % rest = struct2table(rest);
            events = [exec; plan];
            
            %% convert to secs
            events.Onset = events.Onset ./ 1000;
            events.Duration = events.Duration ./ 1000;                 
                       
            varargout{1}= events;

        case 'GLM:design'
            
            sn = [];
            glm = [];
            vararginoptions(varargin,{'sn', 'glm'})

            if isempty(sn)
                error('GLM:design -> ''sn'' must be passed to this function.')
            end

            if isempty(sn)
                error('GLM:design -> ''glm'' must be passed to this function.')
            end

            subj_id = pinfo.subj_id{pinfo.sn==sn};
            
            % Load data once, outside of session loop
            % D = dload(fullfile(baseDir,behavDir,subj_id, sprintf('smp1_%d.dat', sn)));
            events_file = sprintf('glm%d_events.tsv', glm);

            Dd = dload(fullfile(baseDir,behavDir,subj_id, events_file));
            regressors = unique(Dd.eventtype);
            nRegr = length(regressors); 

            % pull list of runs from the participant.tsv:
            run_list = pinfo.('runsSess1'){pinfo.sn==sn};
            run_list = str2double(split(run_list,'.'));
            run_list = arrayfun(@(x) sprintf('%02d', x), run_list, 'UniformOutput', false);
        
            % init J
            J = [];
            T = [];
            J.dir = {fullfile(baseDir,sprintf('glm%d', glm),subj_id)};
            J.timing.units = 'secs';
            J.timing.RT = 1;

            % number of temporal bins in which the TR is divided,
            % defines the discrtization of the HRF inside each TR
            J.timing.fmri_t = 16;

            % slice number that corresponds to that acquired halfway in
            % each TR
            J.timing.fmri_t0 = 1;
        
            for run = 1:max(Dd.BN)
                % Setup scans for current session
                J.sess(run).scans = {fullfile(baseDir, imagingDir, subj_id,'sess1', [subj_id, '_run_', run_list{run}, '.nii'])};
        
        
                % Preallocate memory for conditions
                J.sess(run).cond = repmat(struct('name', '', 'onset', [], 'duration', []), nRegr, 1);
                
                for regr = 1:nRegr
                    % cue = Dd.cue(regr);
                    % stimFinger = Dd.stimFinger(regr);
                    rows = find(Dd.BN == run & strcmp(Dd.eventtype, regressors(regr)));
                    % cue_id = unique(Dd.cue_id(rows));
                    % stimFinger_id = unique(Dd.stimFinger_id(rows));
                    % epoch = unique(Dd.epoch(rows));
                    % instr = unique(Dd.instruction(rows));
                    
                    % Regressor name
                    J.sess(run).cond(regr).name = regressors{regr};
                    
                    % Define durationDuration(regr));
                    J.sess(run).cond(regr).duration = Dd.Duration(rows); % needs to be in seconds
                    
                    % Define onset
                    J.sess(run).cond(regr).onset  = Dd.Onset(rows);
                    
                    % Define time modulator
                    % Add a regressor that account for modulation of
                    % betas over time
                    J.sess(run).cond(regr).tmod = 0;
                    
                    % Orthogonalize parametric modulator
                    % Make the parametric modulator orthogonal to the
                    % main regressor
                    J.sess(run).cond(regr).orth = 0;
                    
                    % Define parametric modulators
                    % Add a parametric modulators, like force or
                    % reaction time. 
                    J.sess(run).cond(regr).pmod = struct('name', {}, 'param', {}, 'poly', {});

                    %
                    % filling in "reginfo"
                    TT.sn        = sn;
                    TT.run       = run;
                    TT.name      = regressors(regr);
                    % TT.cue       = cue_id;
                    % TT.epoch     = epoch;
                    % TT.stimFinger = stimFinger_id;
                    % TT.instr = instr;       

                    T = addstruct(T, TT);

                end

                % Specify high pass filter
                J.sess(run).hpf = Inf;

                % J.sess(run).multi
                % Purpose: Specifies multiple conditions for a session. Usage: It is used
                % to point to a file (.mat or .txt) that contains multiple conditions,
                % their onsets, durations, and names in a structured format. If you have a
                % complex design where specifying conditions manually within the script is
                % cumbersome, you can prepare this information in advance and just
                % reference the file here. Example Setting: J.sess(run).multi =
                % {'path/to/multiple_conditions_file.mat'}; If set to {' '}, it indicates
                % that you are not using an external file to specify multiple conditions,
                % and you will define conditions directly in the script (as seen with
                % J.sess(run).cond).
                J.sess(run).multi     = {''};                        

                % J.sess(run).regress
                % Purpose: Allows you to specify additional regressors that are not
                % explicitly modeled as part of the experimental design but may account for
                % observed variations in the BOLD signal. Usage: This could include
                % physiological measurements (like heart rate or respiration) or other
                % variables of interest. Each regressor has a name and a vector of values
                % corresponding to each scan/time point.
                J.sess(run).regress   = struct('name', {}, 'val', {});                        

                % J.sess(run).multi_reg Purpose: Specifies a file containing multiple
                % regressors that will be included in the model as covariates. Usage: This
                % is often used for motion correction, where the motion parameters
                % estimated during preprocessing are included as regressors to account for
                % motion-related artifacts in the BOLD signal. Example Setting:
                % J.sess(run).multi_reg = {'path/to/motion_parameters.txt'}; The file
                % should contain a matrix with as many columns as there are regressors and
                % as many rows as there are scans/time points. Each column represents a
                % different regressor (e.g., the six motion parameters from realignment),
                % and each row corresponds to the value of those regressors at each scan.
                J.sess(run).multi_reg = {''};
                
                % Specify factorial design
                J.fact             = struct('name', {}, 'levels', {});

                % Specify hrf parameters for convolution with
                % regressors
                J.bases.hrf.derivs = [0 0];
                J.bases.hrf.params = [4.5 11];  % positive and negative peak of HRF - set to [] if running wls (?)
                
                % Specify the order of the Volterra series expansion 
                % for modeling nonlinear interactions in the BOLD response
                % *Example Usage*: Most analyses use 1, assuming a linear
                % relationship between neural activity and the BOLD
                % signal.
                J.volt = 1;

                % Specifies the method for global normalization, which
                % is a step to account for global differences in signal
                % intensity across the entire brain or between scans.
                J.global = 'None';

                % remove voxels involving non-neural tissue (e.g., skull)
                J.mask = {fullfile(baseDir, imagingDir, subj_id, 'sess1', 'rmask_noskull.nii')};
                
                % Set threshold for brightness threshold for masking 
                % If supplying explicit mask, set to 0  (default is 0.8)
                J.mthresh = 0.;

                % Create map where non-sphericity correction must be
                % applied
                J.cvi_mask = {fullfile(baseDir, imagingDir, subj_id, 'sess1', 'rmask_gray.nii')};

                % Method for non sphericity correction
                J.cvi =  'fast';
                
            end

            % T.cue000 = strcmp(T.cue, 'cue0');
            % T.cue025 = strcmp(T.cue, 'cue25');
            % T.cue050 = strcmp(T.cue, 'cue50');
            % T.cue075 = strcmp(T.cue, 'cue75');
            % T.cue100 = strcmp(T.cue, 'cue100');
            % 
            % T.index = strcmp(T.stimFinger, 'index');
            % T.ring = strcmp(T.stimFinger, 'ring');
            % 
            % T.plan = strcmp(T.epoch, 'plan');
            % T.exec = strcmp(T.epoch, 'exec');
            % 
            % T.go = strcmp(T.instr, 'go');
            % T.nogo = strcmp(T.instr, 'nogo');
            % 
            % T.rest = strcmp(T.name, 'rest');
            
            dsave(fullfile(J.dir{1},sprintf('%s_reginfo.tsv', subj_id)), T);
            spm_rwls_run_fmri_spec(J);
            
            % fprintf('- estimates for glm_%d session %d has been saved for %s \n', glm, ses, subj_str{s});

        case 'GLM:design_matrix'
            
            sn = [];
            glm = [];
            vararginoptions(varargin,{'sn', 'glm'})

            if isempty(sn)
                error('GLM:visualize_design_matrix -> ''sn'' must be passed to this function.')
            end

            subj_id = pinfo.subj_id{pinfo.sn==sn};

            SPM = load(fullfile(baseDir,[glmEstDir num2str(glm)],subj_id, 'SPM.mat'));

            X = SPM.SPM.xX.X; % Assuming 'X' is the field holding the design matrix
            
            figure

            imagesc(X); % Plot the design matrix
            colormap('gray'); % Optional: Use a grayscale colormap for better visibility
            colorbar; % Optional: Include a colorbar to indicate scaling
            xlabel('Regressors');
            ylabel('Scans');
            title('Design Matrix');
        
        case 'GLM:estimate'      % estimate beta values
            
            sn = [];
            glm = [];
            vararginoptions(varargin, {'sn', 'glm'})

            if isempty(sn)
                error('GLM:estimate -> ''sn'' must be passed to this function.')
            end

            if isempty(glm)
                error('GLM:estimate -> ''glm'' must be passed to this function.')
            end

            subj_id = pinfo.subj_id{pinfo.sn==sn};
             
            for sess = 1:pinfo.numSess(pinfo.sn==sn)
                fprintf('- Doing glm%d estimation for subj %s\n', glm, subj_id);
                subj_est_dir = fullfile(baseDir, sprintf('glm%d', glm), subj_id);                
                SPM = load(fullfile(subj_est_dir,'SPM.mat'));
                SPM.SPM.swd = subj_est_dir;
            
                spm_rwls_spm(SPM.SPM);
            end

        case 'GLM:T_contrasts'

            sn             = [];    % subjects list
            glm            = [];              % glm number
            replace_xCon   = true;

            vararginoptions(varargin, {'sn', 'glm', 'condition', 'baseline', 'replace_xCon'})

            if isempty(sn)
                error('GLM:T_contrasts -> ''sn'' must be passed to this function.')
            end

            if isempty(glm)
                error('GLM:T_contrasts -> ''glm'' must be passed to this function.')
            end
            
            subj_id = pinfo.subj_id{pinfo.sn==sn};

            % get the subject id folder name
            fprintf('Contrasts for participant %s\n', subj_id)
            glm_dir = fullfile(baseDir, sprintf('glm%d', glm), subj_id); 

            % load the SPM.mat file
            SPM = load(fullfile(glm_dir, 'SPM.mat')); SPM=SPM.SPM;

            if replace_xCon
                SPM  = rmfield(SPM,'xCon');
            end

            T    = dload(fullfile(glm_dir, sprintf('%s_reginfo.tsv', subj_id)));
            contrasts = unique(T.name);
            
            % load contrasts table
            % contr = dload(fullfile(baseDir, sprintf('glm%d', glm), 'contr.tsv'));

            for c = 1:length(contrasts)
                % condition = split(contr.condition(c), ',');
                % baseline = split(contr.baseline(c), ',');
                % 
                % if strcmp(baseline, 'NA') baseline = {}; end

                % fprintf('%s: %s vs. %s\n', subj_id, char(contr.condition(c)), char(contr.baseline(c)))

%                 xcn = zeros(length(T.name));
%                 contrast1 = '';
%                 for cn=1:length(condition)             
%                     if cn > 1
%                         if sum(xcn .* T.(condition{cn})) == 0
%                             xcn = xcn + T.(condition{cn});
%                         else
%                             xcn = xcn .* T.(condition{cn});
%                         end
%                         contrast1 = [contrast1 '_' condition{cn}];
%                     else
%                         xcn = T.(condition{cn});
%                         contrast1 = condition{cn};
%                     end
%                 end
% 
%                 xbs = zeros(length(T.name));
%                 contrast2 = '';
%                 for bs=1:length(baseline)
%                     if bs > 1
%                         if sum(xbs .* T.(condition{bs})) == 0
%                             xbs = xbs + T.(condition{bs});
%                         else
%                             xbs = xbs .* T.(baseline{bs});
%                         end
%                         contrast2 = [contrast2 '_' baseline{bs}];
%                     else
%                         if ~strcmp(baseline{bs}, '')
%                             xbs = T.(baseline{bs});
%                             contrast2 = baseline{bs};
%                         end
%                     end
%                 end
% 
%                 xcon = zeros(size(SPM.xX.X,2), 1);
%                 for ic = 1:length(xcon) - max(T.run)
%                     if xcn(ic) == 1
%                         xcon(ic) = 1;
%                     elseif xbs(ic) == 1
%                         xcon(ic) = -1;
%                     end
%                 end
% 
% %                 if strcmp(baseline, '')
% %                     xcon(end-length(SPM.nscan):end) = -1;
% %                 end
% 
%                 xcon(xcon > 0) = xcon(xcon > 0) / sum(xcon(xcon > 0));
%                 xcon(xcon < 0) = xcon(xcon < 0) / abs(sum(xcon < 0));
% 
%                 if abs(sum(xcon)) > 1e-10
%                     warning(['sum of weight contrast vector is not zero: ' sprintf('%f', sum(xcon))])
%                 else
%                     fprintf('sum of weights: %f\n', sum(xcon))
%                 end

                contrast_name = contrasts{c};
                xcon = zeros(size(SPM.xX.X,2), 1);
                xcon(strcmp(T.name, contrast_name)) = 1;
                if ~isfield(SPM, 'xCon')
                    SPM.xCon = spm_FcUtil('Set', contrast_name, 'T', 'c', xcon, SPM.xX.xKXs);
                    cname_idx = 1;
                elseif sum(strcmp(contrast_name, {SPM.xCon.name})) > 0
                    idx = find(strcmp(contrast_name, {SPM.xCon.name}));
                    SPM.xCon(idx) = spm_FcUtil('Set', contrast_name, 'T', 'c', xcon, SPM.xX.xKXs);
                    cname_idx = idx;
                else
                    SPM.xCon(end+1) = spm_FcUtil('Set', contrast_name, 'T', 'c', xcon, SPM.xX.xKXs);
                    cname_idx = length(SPM.xCon);
                end
                SPM = spm_contrasts(SPM,1:length(SPM.xCon));
                save('SPM.mat', 'SPM','-v7.3');
                % SPM = rmfield(SPM,'xVi'); % 'xVi' take up a lot of space and slows down code!
                % save(fullfile(glm_dir, 'SPM_light.mat'), 'SPM')
    
                % rename contrast images and spmT images
                conName = {'con','spmT'};
                for n = 1:numel(conName)
                    oldName = fullfile(glm_dir, sprintf('%s_%2.4d.nii',conName{n},cname_idx));
                    newName = fullfile(glm_dir, sprintf('%s_%s.nii',conName{n},SPM.xCon(cname_idx).name));
                    movefile(oldName, newName);
                end % conditions (n, conName: con and spmT)

            end

        

        case 'GLM:calc_PSC'

            sn             = [];    % subjects list
            glm            = [];    % glm number

            vararginoptions(varargin, {'sn', 'glm'})

            if isempty(sn)
                error('GLM:T_contrast -> ''sn'' must be passed to this function.')
            end

            if isempty(glm)
                error('GLM:T_contrast -> ''glm'' must be passed to this function.')
            end

            subj_id = pinfo.subj_id{pinfo.sn==sn};            
            glm_dir = fullfile(baseDir, sprintf('glm%d', glm), subj_id); 

            % load the SPM.mat file
            SPM = load(fullfile(glm_dir, 'SPM.mat')); SPM=SPM.SPM;

%             psc = readtable(fullfile(baseDir, sprintf('glm%d', glm), 'psc.txt'));
%             contr = {SPM.xCon.name};
% 
%             intercept = {};
%             for k = 1:SPM.nscan
%                 intercept{end+1} = fullfile(glm_dir, sprintf('beta_%d.nii', 220+k));  
%             end
%             
%             for c = 1:size(psc, 1)
% 
%                 p = [psc.condition(c) '-'];
%                 nRegr = SPM.xCon(find(strcmp({SPM.xCon.name}, p))).c > 0;
% 
%                 P = fullfile(glm_dir, ['con_' p '.nii']);
%                 P = [P, intercept];
% 
%                 imean = sprintf('(i1 ./ %f)', nRegr);
%                 cmean = '((i2 + i3 + i4 + i5 + i6 + i7 + i8 + i9 + i10 + i11) ./ 10)';
% 
%                 formula = sprintf('100 .* (%s - %s) ./ %s', imean, cmean, cmean);
% 
%                 A = [];
%                 A.input = P;
%                 A.output = ['psc_' p];
%                 A.outdir = {glm_dir};
%                 A.expression = formula;
%                 A.var = struct('name', {}, 'value', {});
%                 A.options.dmtx = 0;
%                 A.options.mask = 0;
%                 A.options.interp = 1;
%                 A.options.dtype = 4;               
%     
%                 matlabbatch{1}.spm.util.imcalc=A;
%                 spm_jobman('run', matlabbatch);
%             end
            X=(SPM.xX.X(:,SPM.xX.iC)); % Design matrix - raw

            P={};
            numB=length(SPM.xX.iB);     % Partitions - runs
            for p=SPM.xX.iB
                P{end+1,1}=fullfile(baseDir, glmEstDir,subj_id, ...
                    sprintf('beta_%4.4d.nii',p));  % get the intercepts and use them to calculate the baseline (mean images)
            end
            
            t_con_name = extractfield(SPM.xCon, 'name');
            t_con_name = t_con_name(endsWith(t_con_name, '-'));

            % Create string with formula
            con_div_intercepts = '';
            for r=1:numB
                if r == numB
                    con_div_intercepts = sprintf('i%d./((%si%d)/%d)', ...
                        r+1, con_div_intercepts, r, numB);
                else
                    con_div_intercepts = sprintf('%si%d+', ...
                        con_div_intercepts, r);
                end
            end

            maxX = max(X);
            for con=1:length(t_con_name)  % all contrasts
                idx = find(strcmp({SPM.xCon.name}, t_con_name(1)));
                c = SPM.xCon(idx).c(SPM.xX.iC);
                h = min(maxX(c>0));

                P{numB+1,1}=fullfile(baseDir, glmEstDir, subj_id, ...
                    sprintf('con_%s.nii', t_con_name{con}));
                outname=fullfile(baseDir, glmEstDir, subj_id, ...
                    sprintf('psc_%s.nii', t_con_name{con}));

                formula=sprintf('100.*%f.*%s', h, con_div_intercepts);
                    
                A = [];
                A.input = P;
                A.output = outname;
                A.outdir = {glm_dir};
                A.expression = formula;
                A.var = struct('name', {}, 'value', {});
                A.options.dmtx = 0;
                A.options.mask = 0;
                A.options.interp = 1;
                A.options.dtype = 4;               

                matlabbatch{1}.spm.util.imcalc=A;
                spm_jobman('run', matlabbatch);

            end

            
             
%         case 'GLM:T_contrast'    % make T contrasts for each condition
%             %%% Calculating contrast images.
% 
%             sn             = [];    % subjects list
%             glm            = [];              % glm number
%             condition      = {};
%             baseline       = {};         % contrast will be calculated against base (available options: 'rest')
% 
%             vararginoptions(varargin, {'sn', 'glm', 'condition', 'baseline'})
% 
%             if isempty(sn)
%                 error('GLM:T_contrast -> ''sn'' must be passed to this function.')
%             end
% 
%             if isempty(glm)
%                 error('GLM:T_contrast -> ''glm'' must be passed to this function.')
%             end
% 
%             if isempty(condition)
%                 error('GLM:T_contrast -> ''condition'' must be passed to this function.')
%             end
% 
%             if isempty(condition)
%                 error('GLM:T_contrast -> ''baseline'' must be passed to this function.')
%             end
% 
%             subj_id = pinfo.subj_id{pinfo.sn==sn};
% 
%             % get the subject id folder name
%             fprintf('Contrasts for participant %s\n', subj_id)
%             glm_dir = fullfile(baseDir, sprintf('glm%d', glm), subj_id); 
% 
%             % load the SPM.mat file
%             SPM = load(fullfile(glm_dir, 'SPM.mat')); SPM=SPM.SPM;
% 
%             T    = dload(fullfile(glm_dir, sprintf('%s_reginfo.tsv', subj_id)));
% 
%             xcn = zeros(length(T.name));
%             for cn=1:length(condition)             
%                 if cn > 1
%                     if sum(xcn .* T.(condition{cn})) == 0
%                         xcn = xcn + T.(condition{cn});
%                     else
%                         xcn = xcn .* T.(condition{cn});
%                     end
%                     contrast1 = [contrast1 '&' condition{cn}];
%                 else
%                     xcn = T.(condition{cn});
%                     contrast1 = condition{cn};
%                 end
%             end
% 
%             xbs = zeros(length(T.name));
%             contrast2 = '';
%             for bs=1:length(baseline)
%                 if bs > 1
%                     if sum(xbs .* T.(condition{bs})) == 0
%                         xbs = xbs + T.(condition{bs});
%                     else
%                         xbs = xbs .* T.(baseline{bs});
%                     end
%                     contrast2 = [contrast2 '&' baseline{bs}];
%                 else
%                     xbs = T.(baseline{bs});
%                     contrast2 = baseline{bs};
%                 end
%             end
% 
%             xcon = zeros(size(SPM.xX.X,2), 1);
%             for ic = 1:length(xcon) - max(T.run)
%                 if xcn(ic) == 1
%                     xcon(ic) = 1;
%                 elseif xbs(ic) == 1
%                     xcon(ic) = -1;
%                 end
%             end
% 
%             xcon = xcon/sum(abs(xcon));
%             contrast_name = sprintf('%s-%s', contrast1, contrast2);
%             if ~isfield(SPM, 'xCon')
%                 SPM.xCon = spm_FcUtil('Set', contrast_name, 'T', 'c', xcon, SPM.xX.xKXs);
%                 cname_idx = 1;
%             elseif sum(strcmp(contrast_name, {SPM.xCon.name})) > 0
%                 idx = find(strcmp(contrast_name, {SPM.xCon.name}));
%                 SPM.xCon(idx) = spm_FcUtil('Set', contrast_name, 'T', 'c', xcon, SPM.xX.xKXs);
%                 cname_idx = idx;
%             else
%                 SPM.xCon(end+1) = spm_FcUtil('Set', contrast_name, 'T', 'c', xcon, SPM.xX.xKXs);
%                 cname_idx = length(SPM.xCon);
%             end
%             SPM = spm_contrasts(SPM,1:length(SPM.xCon));
%             save('SPM.mat', 'SPM','-v7.3');
% %             SPM = rmfield(SPM,'xVi'); % 'xVi' take up a lot of space and slows down code!
% %             save(fullfile(glm_dir, 'SPM_light.mat'), 'SPM')
% 
%             % rename contrast images and spmT images
%             conName = {'con','spmT'};
%             for n = 1:numel(conName)
%                 oldName = fullfile(glm_dir, sprintf('%s_%2.4d.nii',conName{n},cname_idx));
%                 newName = fullfile(glm_dir, sprintf('%s_%s.nii',conName{n},SPM.xCon(cname_idx).name));
%                 movefile(oldName, newName);
%             end % conditions (n, conName: con and spmT)
% 
%         % case 'SURF:reconall' % Freesurfer reconall routine
%         %     % Calls recon-all, which performs, all of the
%         %     % FreeSurfer cortical reconstruction process
%         % 
%         %     sn   = subj_id; % subject list
%         % 
%         %     vararginoptions(varargin, {'sn'});
%         % 
%         %     % Parent dir of anatomical images    
%         %     for s = sn
%         %         fprintf('- recon-all %s\n', subj_str{s});
%         %                     % Get the directory of subjects anatomical;
%         %         freesurfer_reconall(fs_dir, subj_str{s}, ...
%         %                   fullfile(anatomical_dir, subj_str{s}, 'anatomical.nii'));
%         %     end % s (sn)
            
        case 'SURF:fs2wb'          % Resampling subject from freesurfer fsaverage to fs_LR
            
            sn   = []; % subject list
            res  = 32;          % resolution of the atlas. options are: 32, 164
            % hemi = [1, 2];      % list of hemispheres
           
            vararginoptions(varargin, {'sn'});

            subj_id = pinfo.subj_id{pinfo.sn==sn};

            fsDir = fullfile(baseDir, 'surfaceFreesurfer', subj_id);

            % dircheck(outDir);
            surf_resliceFS2WB(subj_id, fsDir, fullfile(baseDir, wbDir), 'resolution', sprintf('%dk', res))

        case 'SURF:vol2surf'

            sn   = []; % subject list
            filename = [];
            res  = 32;          % resolution of the atlas. options are: 32, 164
            type = 'tval';
            id = [];
            glm = [];
            % hemi = [1, 2];      % list of hemispheres
           
            vararginoptions(varargin, {'sn', 'glm', 'type'});

            subj_id = pinfo.subj_id{pinfo.sn==sn};
            glmEstDir = [glmEstDir num2str(glm)];
            
            V = {};
            cols = {};
            if strcmp(type, 'spmT')
%                 filename = ['spmT_' id '.func.gii'];
                files = dir(fullfile(baseDir, glmEstDir, subj_id, 'spmT_*.nii'));
                for f = 1:length(files)
                    fprintf([files(f).name '\n'])
                    V{f} = fullfile(files(f).folder, files(f).name);
                    cols{f} = files(f).name;
                end
            elseif strcmp(type, 'beta')
                SPM = load(fullfile(baseDir, glmEstDir, subj_id, 'SPM.mat')); SPM=SPM.SPM;
                files = dir(fullfile(baseDir, glmEstDir, subj_id, 'beta_*.nii'));
                files = files(SPM.xX.iC);
                for f = 1:length(files)
                    fprintf([files(f).name '\n'])
                    V{f} = fullfile(files(f).folder, files(f).name);
                    cols{f} = files(f).name;
                end
            elseif strcmp(type, 'psc')
                files = dir(fullfile(baseDir, glmEstDir, subj_id, 'psc_*.nii'));
                for f = 1:length(files)
                    fprintf([files(f).name '\n'])
                    V{f} = fullfile(files(f).folder, files(f).name);
                    cols{f} = files(f).name;
                end
            elseif strcmp(type, 'con')
                files = dir(fullfile(baseDir, glmEstDir, subj_id, 'con_*.nii'));
                for f = 1:length(files)
                    fprintf([files(f).name '\n'])
                    V{f} = fullfile(files(f).folder, files(f).name);
                    cols{f} = files(f).name;
                end
            elseif strcmp(type, 'res')
                V{1} = fullfile(baseDir, glmEstDir, subj_id, 'ResMS.nii');
                cols{1} = 'ResMS';
            end

            hemLpial = fullfile(baseDir, wbDir, subj_id,  [subj_id '.L.pial.32k.surf.gii']);
            hemRpial = fullfile(baseDir, wbDir, subj_id, [subj_id '.R.pial.32k.surf.gii']);
            hemLwhite = fullfile(baseDir, wbDir, subj_id, [subj_id '.L.white.32k.surf.gii']);
            hemRwhite = fullfile(baseDir, wbDir, subj_id, [subj_id '.R.white.32k.surf.gii']);
            
            hemLpial = gifti(hemLpial);
            hemRpial = gifti(hemRpial);
            hemLwhite = gifti(hemLwhite);
            hemRwhite = gifti(hemRwhite);

            c1L = hemLpial.vertices;
            c2L = hemLwhite.vertices;
            c1R = hemRpial.vertices;
            c2R = hemRwhite.vertices;

            GL = surf_vol2surf(c1L,c2L,V,'anatomicalStruct','CortexLeft', 'exclude_thres', 0.9, 'faces', hemLpial.faces);
            GL = surf_makeFuncGifti(GL.cdata,'anatomicalStruct', 'CortexLeft', 'columnNames', cols);
    
            save(GL, fullfile(baseDir, wbDir, subj_id, [glmEstDir '.'                                               type '.L.func.gii']))
    
            GR = surf_vol2surf(c1R,c2R,V,'anatomicalStruct','CortexRight', 'exclude_thres', 0.9, 'faces', hemRpial.faces);
            GR = surf_makeFuncGifti(GR.cdata,'anatomicalStruct', 'CortexRight', 'columnNames', cols);

            save(GR, fullfile(baseDir, wbDir, subj_id, [glmEstDir '.' type '.R.func.gii']))
            

        case 'SURF:resample_labelFS2WB'

            subj_id = '';
            atlas_dir = [];
            resolution = '32k';
            atlas = 'aparc';

            vararginoptions(varargin, {'sn', 'resolution', 'atlas'});
            
            subj_id = pinfo.subj_id{pinfo.sn==sn};
 
            if isempty(atlas_dir)
%                 repro_dir=fileparts(which('surf_label2label'));
                atlas_dir = [path 'GitHub/surfAnalysis/standard_mesh/'];
            end

            fsDir = fullfile(baseDir, 'surfaceFreesurfer', subj_id, subj_id);
            
            out_dir = fullfile(baseDir, wbDir, subj_id);
            
%             cd(fullfile(subject_dir,subj_id)); 
            
            hem = {'lh', 'rh'};
            Hem = {'L', 'R'};
            aStruct = {'CortexLeft', 'CortexRight'};
            
            for h = 1:length(hem)
            
                reg_sphere = [fsDir '/surf/' hem{h} '.sphere.reg.surf.gii'];
                label = [fsDir '/label/' hem{h} '.' atlas '.annot'];
                surf = [fsDir '/surf/' hem{h} '.pial'];
                out_fs = [fsDir '/label/' hem{h} '.label.gii'];
                source_annot = [fsDir '/label/' hem{h} '.label.gii'];
                out_name = fullfile(out_dir,[subj_id '.' Hem{h} '.' resolution '.' atlas '.label.gii']); 
                atlas_name = fullfile(atlas_dir,'resample_fsaverage',['fs_LR-deformed_to-fsaverage.' Hem{h} '.sphere.' resolution '_fs_LR.surf.gii']);
            
                system(['mris_convert --annot ' label ' ' surf ' ' out_fs]);
            
                system(['wb_command -label-resample ' source_annot ' ' reg_sphere ' ' atlas_name ' BARYCENTRIC ' out_name]);

                A = gifti(out_name);
                cdata = A.cdata;
                keys = A.labels.key;
                name = A.labels.name;
                
                rgba = A.labels.rgba;
                G = surf_makeLabelGifti(cdata, 'anatomicalStruct', aStruct{h},'columnNames', {atlas}, 'labelNames', name, 'labelRGBA', rgba, 'keys', keys);
                save(G, out_name)
            
            end


        case 'SEARCH:define' 

            glm=[];
            sn=[];
            rad=12;
            vox=100;
            res='32';
            vararginoptions(varargin,{'sn','glm','rad','vox','surf'});

            subj_id = pinfo.subj_id{pinfo.sn==sn};
            
            mask = fullfile(baseDir, glmEstDir, subj_id, 'mask.nii');
            Vmask = spm_vol(mask);
            Vmask.data = spm_read_vols(Vmask);
            Mask = rsa.readMask(Vmask);
            
            % directory for pial and white
            surfDir = fullfile(baseDir, wbDir,subj_id);  

            white = {fullfile(surfDir, sprintf('%s.L.white.%sk.surf.gii', subj_id, res)),...
                fullfile(surfDir, sprintf('%s.R.white.%sk.surf.gii', subj_id, res))};
            pial = {fullfile(surfDir, sprintf('%s.L.pial.%sk.surf.gii', subj_id, res)),...
                fullfile(surfDir, sprintf('%s.R.pial.%sk.surf.gii', subj_id, res))};
            
            S = rsa.readSurf(white, pial);  S = [S{:}];
            
            L = rsa.defineSearchlight_surface(S, Mask, 'sphere', [rad vox]);
            save(fullfile(baseDir, anatomicalDir, subj_id, sprintf('%s_searchlight_%d.mat',subj_id,vox)),'-struct','L');
            varargout={L};
        
        case 'HRF:ROI_hrf_get'                   % Extract raw and estimated time series from ROIs
            
            sn = [];
            ROI = 'all';
            pre=10;
            post=10;
            atlas = 'ROI';
            glm = 4;

            vararginoptions(varargin,{'ROI','pre','post', 'glm', 'sn', 'atlas'});

            glmDir = fullfile(baseDir, [glmEstDir num2str(glm)]);
            T=[];

            subj_id = pinfo.subj_id{pinfo.sn==sn};
            fprintf('%s\n',subj_id);

            % load SPM.mat
            cd(fullfile(glmDir,subj_id));
            SPM = load('SPM.mat'); SPM=SPM.SPM;
            
            % load ROI definition (R)
            R = load(fullfile(baseDir, regDir,subj_id,[subj_id '_' atlas '_region.mat'])); R=R.R;
            
            % extract time series data
            [y_raw, y_adj, y_hat, y_res,B] = region_getts(SPM,R);
            
            D = spmj_get_ons_struct(SPM);
            
            for r=1:size(y_raw,2)
                for i=1:size(D.block,1)
                    D.y_adj(i,:)=cut(y_adj(:,r),pre,round(D.ons(i))-1,post,'padding','nan')';
                    D.y_hat(i,:)=cut(y_hat(:,r),pre,round(D.ons(i))-1,post,'padding','nan')';
                    D.y_res(i,:)=cut(y_res(:,r),pre,round(D.ons(i))-1,post,'padding','nan')';
                    D.y_raw(i,:)=cut(y_raw(:,r),pre,round(D.ons(i))-1,post,'padding','nan')';
                end
                
                % Add the event and region information to tje structure. 
                len = size(D.event,1);                
                D.SN        = ones(len,1)*sn;
                D.region    = ones(len,1)*r;
                D.name      = repmat({R{r}.name},len,1);
                D.hem       = repmat({R{r}.hem},len,1);
                D.type      = D.event; 
                T           = addstruct(T,D);
            end
            
            save(fullfile(baseDir,regDir, subj_id, sprintf('hrf_glm%d.mat', glm)),'T'); 
            varargout{1} = T;

        case 'ROI:define'
            
            sn = [];
            glm = 5;
            atlas = 'ROI';
            
            vararginoptions(varargin,{'sn', 'glm', 'atlas'});

            atlasDir = '/Volumes/diedrichsen_data$/data/Atlas_templates/fs_LR_32';
            atlasH = {sprintf('%s.32k.L.label.gii', atlas), sprintf('%s.32k.R.label.gii', atlas)};
            atlas_gii = {gifti(fullfile(atlasDir, atlasH{1})), gifti(fullfile(atlasDir, atlasH{1}))};

            subj_id = pinfo.subj_id{pinfo.sn==sn};

            Hem = {'L', 'R'};
            R = {};
            r = 1;
            for h = 1:length(Hem)
                for reg = 1:length(atlas_gii{h}.labels.name)

                    R{r}.white = fullfile(baseDir, wbDir, subj_id, [subj_id '.' Hem{h} '.white.32k.surf.gii']);
                    R{r}.pial = fullfile(baseDir, wbDir, subj_id, [subj_id '.' Hem{h} '.pial.32k.surf.gii']);
                    R{r}.image = fullfile(baseDir, [glmEstDir num2str(glm)], subj_id, 'mask.nii');
                    R{r}.linedef = [5 0 1];
                    key = atlas_gii{h}.labels.key(reg);
                    R{r}.location = find(atlas_gii{h}.cdata==key);
                    R{r}.hem = Hem{h};
                    R{r}.name = atlas_gii{h}.labels.name{reg};
                    R{r}.type = 'surf_nodes_wb';

                    r = r+1;
                end
            end

            R = region_calcregions(R);
            
            save(fullfile(baseDir, regDir, subj_id, sprintf('%s_%s_region.mat',subj_id, atlas)), 'R');

        case 'HRF:ROI_hrf_plot'                 % Plot extracted time series
            sn = [];
            roi = {'SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp', 'V1'};
            atlas = 'ROI';
            hem = 'L';
            eventname = [];

            vararginoptions(varargin,{'sn', 'roi', 'atlas', 'eventname', 'hem'});

            subj_id = pinfo.subj_id{pinfo.sn==sn};

            figure

            for r=1:length(roi)

                T = load(fullfile(baseDir, regDir, subj_id, 'hrf.mat')); T=T.T;
                T = getrow(T,strcmp(T.name, roi{r}));
                pre = 10;
                post = 10;
                
                % Select a specific subset of things to plot 
                subset      = find(contains(T.eventname, eventname) & strcmp(T.hem, hem));

                subplot(2, 4, r)
                
                % yyaxis left
                traceplot([-pre:post],T.y_hat, 'subset',subset, 'split', [], 'linestyle','--');
                hold on;
                traceplot([-pre:post],T.y_res, 'subset',subset, 'split', [], 'linestyle',':');
                xline(0);
                yline(0);
                % ax = gca;
                % ax.YColor = 'b';
                
    
                % yyaxis right
                traceplot([-pre:post],T.y_adj, 'subset',subset,'leg',[], 'leglocation','bestoutside', 'linestyle','-', 'linecolor', [1 0 0]);
                % ax = gca;
                % ax.YColor = 'r';
                
                if r==1
                    legend({'y_{hat}', 'residuals', '', '', 'y_{raw}'}, 'Location','northwest')
                end

                hold off;
                xlabel('TR');
                ylabel('activation');

                ylim([-1, 2])
    
                title(roi{r})

            end

            sgtitle(sprintf('%s\nhemisphere:%s, eventname:%s', subj_id, hem, eventname), 'interpreter', 'none')
            set(gcf, 'Position', [100 100, 1400, 800])

            drawnow;

            fig = gcf;

            saveas(fig, fullfile(baseDir, 'figures', subj_id, sprintf('hrf.%s.%s.png', hem, eventname)))
            
            
    end


end


