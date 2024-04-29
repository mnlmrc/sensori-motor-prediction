function varargout = op2multi_imana(what, varargin)
% multivariate analysis of the online planning experiment
%% ------------------------- Directories ----------------------------------
if ismac
    baseDir='/Volumes/Diedrichsen_data$/data/SequenceAndChord/OnlinePlanning/op2';
    atlasDir='/Volumes/Diedrichsen_data$/data/Atlas_templates';
elseif isunix
    baseDir='/srv/diedrichsen/data/SequenceAndChord/Onlineplanning/op2';
    atlasDir='/srv/diedrichsen/data/Atlas_templates';
end

imagingDir=fullfile(baseDir,'imaging');
anatomicalDir=fullfile(baseDir,'anatomicals');
freesurferDir=fullfile(baseDir,'surfaceFreesurfer');
behavDir=fullfile(baseDir,'behavior');
wbDir=fullfile(baseDir,'surfaceWB');
roiDir=fullfile(baseDir,'ROI');
standardmeshDir=fullfile('~','Documents','MATLAB','toolboxes','surfAnalysis');
pathToSave=fullfile('~','Documents','GitHub','OnlinePlanning','op2','Fig');
glmDir=[baseDir '/glm_%d'];
%% ------------------------- ROI things -----------------------------------
hemi={'lh','rh'}; % left & right hemi folder names/prefixes
hem={'L','R'}; % hemisphere: 1=LH 2=RH
hname={'CortexLeft','CortexRight'}; % 'CortexLeft', 'CortexRight', 'Cerebellum'

% roi names, independent of hemisphere
ROI_Brodmann_name={'S1','M1','PMd','PMv','SMA','V1','SPLa','SPLp'};
ROI_Wang_name = {'V1v','V1d','V2v','V2d','V3v','V3d','hV4','VO1','VO2','PHC1','PHC2',... 
    'TO2','TO1','LO2','LO1','V3B','V3A','IPS0','IPS1','IPS2','IPS3','IPS4',...
    'IPS5','SPL1','FEF'};
regSide=[1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2];
regType=[1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8];
%% ------------------------- Subject and task's things --------------------
subj_name={'s01','s02','s03','s04','s05','s06','s07','s08','s09','s10','s11',...
    's12','s13','s14','s15','s16','s17','s18','s19','s20','s21','s22'};
ns=numel(subj_name);
subj_vec=zeros(1,ns);
for i=1:ns;subj_vec(1,i)=str2double(subj_name{i}(2:3));end
%% ------------------------- Analysis Cases -------------------------------
switch(what)
    case 'SEARCH:define' % defines searchlights for 120 voxels in gray matter surface
        glm=8;sn=1;rad=12;vox=100;surf='32';
        vararginoptions(varargin,{'sn','glm','rad','vox','surf'});
        
        mask=fullfile(sprintf(glmDir,glm),sprintf('s%02d',sn),'mask.nii');
        Vmask=spm_vol(mask);
        Vmask.data=spm_read_vols(Vmask);
        
        % directory for pial and white
        surfDir=fullfile(wbDir,sprintf('s%02d',sn));        
        white={fullfile(surfDir,sprintf('%s.L.white.%sk.surf.gii',sprintf('s%02d',sn),surf)),...
            fullfile(surfDir,sprintf('%s.R.white.%sk.surf.gii',sprintf('s%02d',sn),surf))};
        pial={fullfile(surfDir,sprintf('%s.L.pial.%sk.surf.gii',sprintf('s%02d',sn),surf)),...
            fullfile(surfDir,sprintf('%s.R.pial.%sk.surf.gii',sprintf('s%02d',sn),surf))};
        
        S=rsa.readSurf(white,pial);
        
        L=rsa.defineSearchlight_surface(S,Vmask,'sphere',[rad vox]);
        save(fullfile(anatomicalDir,sprintf('s%02d',sn),sprintf('s%02d_searchlight_%d.mat',sn,vox)),'-struct','L');
        varargout={L};
    case 'SEARCH:run_LDC' % runs LDC searchlight using defined searchlights (above)
        glm=8;sn=1;rad=12;vox=100;
        vararginoptions(varargin,{'sn','glm','rad','vox'});
        cwd=pwd;
        
        spmDir=fullfile(sprintf(glmDir,glm),sprintf('s%02d',sn));
        cd(spmDir)
        D=load(fullfile(sprintf(glmDir,glm),sprintf('s%02d',sn),'SPM_info.mat')); % load subject's trial structure
        
        % load their searchlight definitions and SPM file
        L=load(fullfile(anatomicalDir,sprintf('s%02d',sn),sprintf('s%02d_searchlight_%d.mat',sn,vox)));
        
        % load files
        load(fullfile(sprintf(glmDir,glm),sprintf('s%02d',sn),'SPM.mat'));  % load subject's SPM data structure (SPM struct)
        
        % update the naming of the directory
        if ~strcmp(fullfile(sprintf(glmDir,glm),sprintf('s%02d',sn)),SPM.swd) % need to rename SPM
            SPM=spmj_move_rawdata(SPM,fullfile(imagingDir,sprintf('s%02d',sn)));
        end
        
        name=sprintf('s%02d_glm%d_dist',sn,glm);
        
        % make index vectors
        conditionVec=D.cond;
        partition=D.run;
        
        % run the searchlight
        tic;
        rsa.runSearchlightLDC(L,SPM,'conditionVec',conditionVec,'partition',partition,'spmDir',spmDir,'analysisName',name);
        toc;
        
        cd(cwd);
        varargout={SPM,vox,rad};
    case 'SEARCH:contrast' % averaged LDC values for specified contrasts
        sn=1;glm=8;
        vararginoptions(varargin,{'sn','glm'});
        cwd=pwd;
        
        % go to subject's directory
        cd(fullfile(sprintf(glmDir,glm),sprintf('s%02d',sn)));
        
        load SPM;
        D=load('SPM_info.mat'); % load subject's trial structure
        
        % define the contrasts
        con_name={'single','chord','all'};
        
        indMat=indicatorMatrix('allpairs',unique(D.cond(D.cond>0))');
        single=[];
        chord=[];
        all=[];
        
        for i=1:length(indMat)
            if sum(indMat(i,1:5)~=0)==2 % chord: first 5 is the chord
                chord=[chord;i];
                all=[all;i];
            elseif sum(indMat(i,6:10)~=0)==2 % single: second 5 is the sinlge
                single=[single;i];
                all=[all;i];
            else
                continue
            end
        end
        
        con_idx{1}=single;
        con_idx{2}=chord;
        con_idx{3}=all;
        
        % Load subject surface searchlight results (1 vol per paired conds)
        LDC_file=fullfile(sprintf(glmDir,glm),sprintf('s%02d',sn),sprintf('s%02d_glm%d_dist_LDC.nii',sn,glm)); % searchlight nifti
        [subjDir,fname,ext]=fileparts(LDC_file);
        cd(subjDir);
        
        % Repeat for each of the defined contrasts
        for c=1:length(con_name)
            vol=spm_vol([fname ext]);
            vdat=spm_read_vols(vol); % is searchlight data
            
            % average across all paired dists
            %Y.LDC       = nanmean(vdat, 4);
            
            % prep output file
            Y.dim=vol(1).dim;
            Y.dt=vol(1).dt;
            Y.mat=vol(1).mat;
            
            % select contrast
            vdat=vdat(:,:,:,con_idx{c}); % is searchlight data
            
            % average across paired dists if the contrast
            Y.LDC=nanmean(vdat,4);
            
            % save output
            Y.fname=sprintf('s%02d_glm%d_dist_%s.nii',sn,glm,con_name{c});
            Y.descrip=sprintf('exp: ''op2'' \nglm: ''FAST'' \ncontrast: ''%s''',con_name{c});
            spm_write_vol(Y,Y.LDC);fprintf('Done s%02d_glm%d_dist_%s.nii \n',sn,glm,con_name{c})
            % clear variables
            clear vol vdat LDC Y
        end
        % save output and return to working directory
        SPM.multiv_con=con_name;
        save('SPM.mat','SPM');
        cd(cwd);
        varargout={SPM};
    case 'SEARCH:vol2surf' % map indiv vol distance (.nii) onto surface (.gifti)
        sn=subj_vec;
        glm=8;
        surf='32k';
        hemis=[1 2];
        vararginoptions(varargin,{'sn','glm','hemis','surf'});
        % define the contrasts
        con_name={'single','chord','all'};
        
        for s=sn
            images=cell(1,numel(con_name));
            for h=hemis
                surfDir=fullfile(wbDir,sprintf('s%02d',s));
                white=fullfile(surfDir,sprintf('%s.%s.white.%s.surf.gii',sprintf('s%02d', s),hem{h},surf));
                pial=fullfile(surfDir,sprintf('%s.%s.pial.%s.surf.gii',sprintf('s%02d',s),hem{h},surf));
                C1=gifti(white);
                C2=gifti(pial);
                
                for i=1:numel(con_name)
                    images{i}=fullfile(sprintf(glmDir,glm),sprintf('s%02d',s),sprintf('s%02d_glm%d_dist_%s.nii',s,glm,con_name{i}));
                end
                
                outfile=fullfile(surfDir,sprintf('%s.%s.glm%d.dist.func.gii',sprintf('s%02d',s),hem{h},glm));
                G=surf_vol2surf(C1.vertices,C2.vertices,images,'column_names',con_name,'anatomicalStruct',hname{h},'exclude_thres',0.75,'faces',C1.faces,'ignore_zeros',0);
                save(G,outfile);
                fprintf('mapped %s %s glm%d \n',sprintf('s%02d',s),hemi{h},glm);
            end
        end
    case 'SEARCH:group_maps' % map group contrasts on surface (.gifti)
        sn=subj_vec;
        glm=8;
        hemis=[1 2];
        map='dist';
        surf='32k';
        vararginoptions(varargin,{'sn','glm','hemis','map'});
        groupDir=fullfile(wbDir,sprintf('group%s',surf));
        dircheck(groupDir);
        
        for h=hemis
            fprintf(1,'%s ...',hemi{h});
            inputFiles={};
            columnName={};
            for s=sn
                inputFiles{end+1}=fullfile(wbDir,sprintf('s%02d',s),sprintf('%s.%s.glm%d.%s.func.gii',sprintf('s%02d',s),hem{h},glm,map));
                columnName{end+1}=sprintf('s%02d',s);
            end
            groupfiles=cell(1);
            con_name={'single','chord','all'};
            nc=numel(con_name);
            for ic=1:nc
                groupfiles{ic}=fullfile(groupDir,sprintf('group.%s.%s.glm%d.%s.func.gii',map,hem{h},glm,con_name{ic}));
            end
            
            surf_groupGiftis(inputFiles,'outcolnames',columnName,'outfilenames',groupfiles);
            fprintf(1,'Done.\n');
        end
    case 'SEARCH:group_stats' % perform group stats on surface (.gifti)
        glm=8;
        hemis=[1 2];
        map='dist';
        sm=0; % smoothing kernel in mm (optional)
        surf='32'; % 164k or 32k vertices
        vararginoptions(varargin,{'glm','hemis','map','sm','surf'});
        groupDir=fullfile(wbDir,sprintf('group%sk',surf));
        dircheck(groupDir);
        % loop over the metric files and calculate the cSPM of each
        for h=hemis
            fprintf(1,'%s ...\n',hemi{h});
            
            groupfiles=cell(1);
            con_name={'single','chord','all'};
            nc=numel(con_name);
            
            % perform stats
            for ic=1:nc
                groupfiles{ic}=fullfile(groupDir,sprintf('group.%s.%s.glm%d.%s.func.gii',map,hem{h},glm,con_name{ic}));
                
                % Perform smoothing (optional)
                if sm>0
                    surface=fullfile(atlasDir, sprintf('FS_LR_%s/fs_LR.%sk.%s.flat.surf.gii',surf,surf,hem{h}));
                    groupfiles{ic}=surf_smooth(groupfiles{ic},'surf',surface,'kernel',sm);
                end
                metric=gifti(groupfiles{ic});
                
                cSPM=surf_getcSPM('onesample_t','data',metric.cdata,'maskthreshold',0.7);%,'maskthreshold',0.5); % set maskthreshold to 0.5 = calculate stats at location if 50% of subjects have data at this point
                C.data(:,ic)=cSPM.con.con; % mean
                C.c_name{ic}=['mean_' con_name{ic}];
                C.data(:,ic+nc)=cSPM.con.Z; % t (confusing)
                C.c_name{ic+nc}=['t_' con_name{ic}];
            end
            % Save output
            O = surf_makeFuncGifti(C.data,'columnNames',C.c_name,'anatomicalStruct',hname{h});
            summaryfile=fullfile(groupDir,sprintf('summary.%s.glm%d.%s.sm%d.func.gii',hem{h},glm,map,sm));
            save(O,summaryfile);
            fprintf(1,'Done.\n');
        end
    case 'SEARCH:group_contrast' % chord, single dissimilarity difference
        glm=8;
        hemis=[1 2];
        map='dist';
        sm=0; % smoothing kernel in mm (optional)
        surf='32'; % 164k or 32k vertices
        vararginoptions(varargin,{'glm','hemis','map','sm','surf'});
        groupDir=fullfile(wbDir,sprintf('group%sk',surf));
        dircheck(groupDir);
        % loop over the metric files and calculate the cSPM of each
        for h=hemis
            fprintf(1,'%s ...\n',hemi{h});
            
            con_name={'chord','single'};
            nc=numel(con_name);
            
            % perform stats
            data=0;
            c=[1 -1];
            for ic=1:nc
                groupfiles=fullfile(groupDir,sprintf('group.%s.%s.glm%d.%s.func.gii',map,hem{h},glm,con_name{ic}));
                
                % Perform smoothing (optional)
                if sm>0
                    surface=fullfile(atlasDir,sprintf('FS_LR_%s/fs_LR.%sk.%s.flat.surf.gii',surf,surf,hem{h}));
                    groupfiles=surf_smooth(groupfiles,'surf',surface,'kernel',sm);
                end
                metric=gifti(groupfiles);
                
                data=data+c(ic)*double(metric.cdata);
            end
            
            cSPM=surf_getcSPM('onesample_t','data',data,'maskthreshold',0.75);%,'maskthreshold',0.5); % set maskthreshold to 0.5 = calculate stats at location if 50% of subjects have data at this point
            C.data(:,1)=cSPM.con.con; % mean
            C.c_name{1}=['mean_' [con_name{1} '_' con_name{2}]];
            C.data(:,2)=cSPM.con.Z; % t
            C.c_name{2}=['t_' [con_name{1} '_' con_name{2}]];
            
            % Save output
            O = surf_makeFuncGifti(C.data,'columnNames',C.c_name,'anatomicalStruct',hname{h});
            summaryfile=fullfile(groupDir,sprintf('%s.%s.glm%d.%s.sm%d.func.gii',[con_name{1} '_' con_name{2}],hem{h},glm,map,sm));
            save(O,summaryfile);
            fprintf(1,'Done.\n');
        end
    case 'SEARCH:rdm_surface' % project rdms into the surface
        sn=subj_vec;glm=8;surf='32k';hemis=[1 2];
        vararginoptions(varargin,{'sn','glm','surf','hemis'});
        cwd=pwd;
        
        % loop over all subject
        for s=sn
            % go to subject's directory
            cd(fullfile(sprintf(glmDir,glm),sprintf('s%02d',s)));

            load SPM;
            D=load('SPM_info.mat'); % load subject's trial structure

            % Load subject surface searchlight results (1 vol per paired conds)
            LDC_file=fullfile(sprintf(glmDir,glm),sprintf('s%02d',s),sprintf('s%02d_glm%d_dist_LDC.nii',s,glm)); % searchlight nifti
            [subjDir,fname,ext]=fileparts(LDC_file);
            cd(subjDir);

            vol=spm_vol([fname ext]);
            vdat=spm_read_vols(vol); % is searchlight data
            name=sprintf('s%02d_glm%d_dist',s,glm);

            % loop over all distances and save all distance file separately
            for d=1:size(vdat,4)
                % prep output file
                Y.dim=vol(1).dim;
                Y.dt=vol(1).dt;
                Y.mat=vol(1).mat;

                % dth distance
                Y.LDC=vdat(:,:,:,d);

                % save output
                Y.fname=sprintf([name '_%.2d.nii'],d);
                Y.descrip=sprintf('exp: ''op2'' \nglm: ''FAST'' \ndist: ''%d''',d);
                spm_write_vol(Y,Y.LDC);fprintf('Done s%02d glm%d dist_%.2d \n',s,glm,d)
                % clear variables
                clear Y
            end
            % project into surface
            images=cell(1,size(vdat,4));
            for h=hemis
                surfDir=fullfile(wbDir,sprintf('s%02d',s));
                white=fullfile(surfDir,sprintf('%s.%s.white.%s.surf.gii',sprintf('s%02d', s),hem{h},surf));
                pial=fullfile(surfDir,sprintf('%s.%s.pial.%s.surf.gii',sprintf('s%02d',s),hem{h},surf));
                C1=gifti(white);
                C2=gifti(pial);

                col_name=cell(1,size(vdat,4));
                % loop over all distances
                for d=1:size(vdat,4)
                    images{d}=fullfile(sprintf(glmDir,glm),sprintf('s%02d',s),sprintf([name '_%.2d.nii'],d));
                    col_name{d}=num2str(d);
                end

                outfile=fullfile(surfDir,sprintf('%s.%s.glm%d.distall.func.gii',sprintf('s%02d',s),hem{h},glm));
                G=surf_vol2surf(C1.vertices,C2.vertices,images,'column_names',col_name,'anatomicalStruct',hname{h},'exclude_thres',0.75,'faces',C1.faces,'ignore_zeros',0);
                save(G,outfile);
                fprintf('mapped %s %s glm%d \n',sprintf('s%02d',s),hemi{h},glm);
            end
            cd(cwd);
        end
    case 'SEARCH:reliability' % calculate the reliability of RDMs at each location
        sn=subj_vec;glm=8;surf='32k';hemis=[1 2];
        vararginoptions(varargin,{'sn','glm','surf'});
        cwd=pwd;
        
        for h=hemis
            % loop over all subject and load the data
            for s=1:numel(sn)
                surfDir=fullfile(wbDir,sprintf('s%02d',sn(s)));
                file=fullfile(surfDir,sprintf('%s.%s.glm%d.distall.func.gii',sprintf('s%02d',sn(s)),hem{h},glm));
                metric=gifti(file);
                data{s}=double(metric.cdata);
            end
            %cond_multi=[135,145,235,245,345,1,2,3,4,5];
            press_tmap=zeros(size(data{1},1),1);
            cue_tmap=zeros(size(data{1},1),1);
            press_cue_tmap=zeros(size(data{1},1),1);

            % loop over all indices
            loadbar=waitbar(0,'Good things come with patience...');
            for i=1:size(data{1},1)
                waitbar(i/size(data{1},1))
                % check which subject has value at this index
                subj=[];
                for s=1:numel(sn)
                    if var(data{s}(i,:))>0
                        subj=[subj,s];
                    end
                end

                % if we have data for more than 12 subjects
                if length(subj)>12
                    press_reliability=[];
                    cue_reliability=[];
                    press_cue=[];
                    for s=subj

                        % test and train subjects
                        test_subj=s;
                        train_subj=subj;
                        train_subj(train_subj==test_subj)=[];

                        % odd and even group
                        group_test=mod(sn(s),2);
                        group_train=mod(sn(train_subj),2);

                        % vetorize RDMs
                        test_dist=data{test_subj}(i,:);

                        train_dist=zeros(length(train_subj),length(test_dist));
                        for t=1:numel(train_subj)
                            train_dist(t,:)=data{train_subj(t)}(i,:);
                        end

                        % matrix RDMs
                        test_RDM=rsa.rdm.squareRDM(test_dist);
                        train_RDM=zeros(size(test_RDM,1),size(test_RDM,2),length(train_subj));
                        for t=1:numel(train_subj)
                            train_RDM(:,:,t)=rsa.rdm.squareRDM(train_dist(t,:));
                        end

                        % press
                        mean_press_RDM=mean(train_RDM(:,:,group_train~=group_test),3);
                        % cue
                        mean_cue_RDM=mean(train_RDM(:,:,group_train~=group_test),3);

                        % press reliability
                        val_press_reliability=(rsa_covWeightCosine(rsa.rdm.vectorizeRDM(test_RDM(1:5,1:5)),rsa.rdm.vectorizeRDM(mean_press_RDM(1:5,1:5)))+...
                            rsa_covWeightCosine(rsa.rdm.vectorizeRDM(test_RDM(6:10,6:10)),rsa.rdm.vectorizeRDM(mean_press_RDM(6:10,6:10))))/2;

                        % cue reliability
                        val_cue_reliability=(rsa_covWeightCosine(rsa.rdm.vectorizeRDM(test_RDM(1:5,1:5)),rsa.rdm.vectorizeRDM(mean_cue_RDM(6:10,6:10)))+...
                            rsa_covWeightCosine(rsa.rdm.vectorizeRDM(test_RDM(6:10,6:10)),rsa.rdm.vectorizeRDM(mean_cue_RDM(1:5,1:5))))/2;

                        % append the data from different folds
                        press_reliability=[press_reliability,val_press_reliability];
                        cue_reliability=[cue_reliability,val_cue_reliability];
                    end
                    %[press_tmap(i,1),~]=ttest_2(press_reliability,0,2,'onesample');
                    press_tmap(i,1)=mean(press_reliability);
                    cue_tmap(i,1)=mean(cue_reliability);
                    press_cue_tmap(i,1)=mean(press_reliability-cue_reliability);
                    %[cue_tmap(i,1),~]=ttest_2(cue_reliability,0,2,'onesample');
                end
            end
            close(loadbar)

            % Save output
            groupDir=fullfile(wbDir,sprintf('group%s',surf));
            O=surf_makeFuncGifti([press_tmap cue_tmap press_cue_tmap],'columnNames',{'press','cue','diff'},'anatomicalStruct',hname{h});
            summaryfile=fullfile(groupDir,sprintf('reliability.%s.glm%d.func.gii',hem{h},glm));
            save(O,summaryfile);
            fprintf(1,'Done.\n');
        end
    case 'SEARCH:model_selection' % check if press wins or cue?
        sn=subj_vec;glm=8;surf='32k';h=1;
        vararginoptions(varargin,{'sn','glm','h','surf'});
        cwd=pwd;
        % loop over all subject and load the data
        for s=1:numel(sn)
            surfDir=fullfile(wbDir,sprintf('s%02d',sn(s)));
            file=fullfile(surfDir,sprintf('%s.%s.glm%d.distall.func.gii',sprintf('s%02d',sn(s)),hem{h},glm));
            metric=gifti(file);
            data{s}=double(metric.cdata);
        end
        %cond_multi=[135,145,235,245,345,1,2,3,4,5];
        % press-cue map...
        press_cue_tmap=zeros(size(data{1},1),1);
        
        % loop over all indices
        h=waitbar(0,'Good things come with patience...');
        for i=1:size(data{1},1)
            waitbar(i/size(data{1},1))
            % check which subject has value at this index
            subj=[];
            for s=1:numel(sn)
                if var(data{s}(i,:))>0
                    subj=[subj,s];
                end
            end
            
            % if we have data for more than 12 subjects
            if length(subj)>12
                press_cue=[];
                for s=subj
                    test_subj=s;
                    train_subj=[1:s-1 s+1:length(sn)];

                    % odd and even group
                    group_test=mod(sn(s),2);
                    group_train=mod(sn(train_subj),2);

                    % vetorize RDMs
                    test_dist=data{test_subj}(i,:);

                    train_dist=zeros(length(train_subj),length(test_dist));
                    for t=1:numel(train_subj)
                        train_dist(t,:)=data{train_subj(t)}(i,:);
                    end

                    % matrix RDMs
                    test_RDM=rsa.rdm.squareRDM(test_dist);
                    train_RDM=zeros(size(test_RDM,1),size(test_RDM,2),length(train_subj));
                    for t=1:numel(train_subj)
                        train_RDM(:,:,t)=rsa.rdm.squareRDM(train_dist(t,:));
                    end

                    % press
                    mean_press_RDM=mean(train_RDM,3);

                    % cue
                    mean_cue_RDM=train_RDM;
                    mean_cue_RDM(:,:,group_train~=group_test)=mean_cue_RDM([6:10 1:5 11],[6:10 1:5 11],group_train~=group_test);
                    mean_cue_RDM=mean(mean_cue_RDM,3);

                    % press-cue:
                    val=rsa_covWeightCosine(rsa.rdm.vectorizeRDM(test_RDM(1:5,1:5)),rsa.rdm.vectorizeRDM(mean_press_RDM(1:5,1:5)))+...
                        rsa_covWeightCosine(rsa.rdm.vectorizeRDM(test_RDM(6:10,6:10)),rsa.rdm.vectorizeRDM(mean_press_RDM(6:10,6:10)))-...
                        rsa_covWeightCosine(rsa.rdm.vectorizeRDM(test_RDM(1:5,1:5)),rsa.rdm.vectorizeRDM(mean_cue_RDM(1:5,1:5)))-...
                        rsa_covWeightCosine(rsa.rdm.vectorizeRDM(test_RDM(6:10,6:10)),rsa.rdm.vectorizeRDM(mean_cue_RDM(6:10,6:10)));

                    press_cue=[press_cue,val];
                end
                [press_cue_tmap(i,1),~]=ttest_2(press_cue,0,2,'onesample');
            end
        end
        close(h)
        
        % Save output
        groupDir=fullfile(wbDir,sprintf('group%s',surf));
        O=surf_makeFuncGifti(press_cue_tmap,'columnNames',{'press_cue'},'anatomicalStruct',hname{1});
        summaryfile=fullfile(groupDir,sprintf('presscue.%s.glm%d.func.gii',hem{1},glm));
        save(O,summaryfile);
        fprintf(1,'Done.\n');
        
    case 'TESSEL:select' % select the tessel with significant decodability
        hemi = 1;
        n_tessel = 362;
        surf = '32';
        glm = 8;
        vararginoptions(varargin,{'hemi','n_tessel','surf','glm'});
        
        % load in distances and icosahedron
        I = gifti(fullfile(atlasDir,sprintf('FS_LR_%s',surf),sprintf('Icosahedron-%d.%sk.%s.label.gii',n_tessel,surf,hem{hemi})));
        G = gifti(fullfile(wbDir,sprintf('group%dk',32),sprintf('summary.%s.glm%d.dist.sm0.func.gii',hem{hemi},glm)));
        
                
        %maskDist = G.cdata(:,3)>.005; % all vertices where distances (all) > .004
        %maskDist = G.cdata(:,6)>3.4; % all vertices where distances (all) significant with p<.001
        maskDist = G.cdata(:,3)>.001 & G.cdata(:,6)>3.819;
        tessels = unique(I.cdata)';
        tessels = tessels(tessels~=0); % exclude 0 - medial wall
        choice = [];
        for i=tessels
            numAll = sum(I.cdata==i);
            distPres = sum(maskDist(I.cdata==i)); % how many times significant distance (dist presence)
            if distPres>(numAll*.4)
                choice = [choice,i];
            end
        end
        varargout={double(choice)};
    case 'calc_overlap-nonoverlapcontrast'
        hemi = 1;
        surf = '32';
        glm = 3;
        sm = 0;
        vararginoptions(varargin,{'hemi','n_tessel','surf','glm'});
        
                
        single = gifti(fullfile(wbDir,sprintf('group%sk',surf),sprintf('group.psc.%s.glm%d.overlap-long_single.func.gii',hem{hemi},glm)));
        chord = gifti(fullfile(wbDir,sprintf('group%sk',surf),sprintf('group.psc.%s.glm%d.overlap-long_chord.func.gii',hem{hemi},glm)));

        both = (single.cdata+chord.cdata)/2;
        name_both = fullfile(wbDir,sprintf('group%sk',surf),sprintf('group.psc.%s.glm%d.overlap-long_both.func.gii',hem{hemi},glm));
        
        
        O_both = surf_makeFuncGifti(both,'columnNames',subj_name, 'anatomicalStruct',hname{hemi});
        save(O_both,name_both);
        
        % Apply smoothing to individuals (optional)
        if sm>0
            surface = fullfile(atlasDir,sprintf('FS_LR_%s/fs_LR.%sk.%s.flat.surf.gii',surf,surf,hem{hemi}));
            name_both = surf_smooth(name_both,'surf',surface,'kernel',sm);            
        end
        
        metric = gifti(name_both);
        cSPM = surf_getcSPM('onesample_t','data',metric.cdata);
        C.data(:,1) = cSPM.con.con; % mean
        C.c_name{1} = 'mean_psc';
        C.data(:,2) = cSPM.con.Z; % t
        C.c_name{2} = 't_psc';
        
        % Save output
        O = surf_makeFuncGifti(C.data,'columnNames',C.c_name,'anatomicalStruct',hname{hemi});
        summaryfile = fullfile(wbDir,sprintf('group%sk',surf),sprintf('summary.%s.%s.func.gii',hem{hemi},'overlap-long'));
        save(O,summaryfile);        
    case 'TESSEL:online_pre' % among the selected tessel which ones shows higher activity for overlap condition
        hemi = 1;
        n_tessel = 362;
        surf = '32';
        glm = 3;
        sm = 0;
        vararginoptions(varargin,{'hemi','n_tessel','surf','glm'});
        
        
        % load icosahedron
        I = rsa.gifti(fullfile(atlasDir,sprintf('FS_LR_%s',surf),sprintf('Icosahedron-%d.%sk.%s.label.gii',n_tessel,surf,hem{hemi})));
        
        % load the summary for the contrast
        summaryfile = fullfile(wbDir,sprintf('group%sk',surf),sprintf('summary.%s.%s.func.gii',hem{hemi},'overlap-long'));
        G = gifti(summaryfile);
        
        initial_tessels = op2multi_imana('TESSEL:select','hemi',hemi);
        mask = G.cdata(:,2)>2.831;%2.831;%3.4;
        
        choice = [];
        for i=initial_tessels
            numAll = sum(I.cdata==i);
            conPres = sum(mask(I.cdata==i)); % how many times significant distance (con presence)
            if conPres>(numAll*.4)
                choice = [choice,i];
            end
        end
        %varargout={double(choice)};
        % creat the mask
        
        pathtosurf = fullfile(atlasDir,sprintf('FS_LR_%s',surf));
        % Brodmann ROI file
        label_brodmann = fullfile(pathtosurf,sprintf('ROI.%sk.%s.label.gii',surf,hem{hemi}));
        P1 = gifti(label_brodmann);
        name1=P1.labels.name;
        color=[0,0,0,0;1,1,1,1];
        
        mask = double(ismember(I.cdata,choice));
        G = surf_makeLabelGifti(mask,'labelNames',{name1{1},'overlap_long'},'labelRGBA',color,'anatomicalStruct',hname{hemi});
        file = fullfile(wbDir,sprintf('group%sk',surf),sprintf('%s.%s.label.gii','overlap-long',hem{hemi}));
        save(G,file);
        
        varargout = {choice};
    case 'label_to_border'
        surf = '32';
        h = 1;
        
        surface = fullfile(atlasDir, sprintf('FS_LR_%s/fs_LR.%sk.%s.flat.surf.gii', surf, surf, hem{h}));
        label = fullfile(wbDir,sprintf('group%sk',surf),sprintf('%s.%s.label.gii','overlap-long',hem{h}));
        
        border = fullfile('python','notebooks',sprintf('%s.%s.border','overlap-long',hem{h}));
        
        comm=sprintf('wb_command -label-to-border %s %s %s -placement %f',...
                surface,label,border,0.001);
            
        fprintf('%s\n',comm) 
        [err,out]=system(comm);
        disp(err)
        
    case 'label_to_text'
        
        
        
        vertices = readmatrix('python/notebooks/overlap-long.L.txt','Delimiter',' ');
        vertices = vertices(:,1);
        vertices = unique(vertices);
        
        h = 1;
        surf = '32';
        
        pathtosurf = fullfile(atlasDir, sprintf('FS_LR_%s', surf));
        
        surface = fullfile(pathtosurf, sprintf('fs_LR.%sk.%s.flat.surf.gii', surf,hem{h}));
        Flat = gifti(surface);
        
        
        border = Flat.vertices(vertices,:);
        
        dist = squareform(pdist(border));
        dist = 100*eye(size(dist))+dist;
        border(find(min(dist,[],2)>4),:) = [];
        
        writematrix(border,'python/notebooks/border_op.txt','Delimiter',',') 
        
    case 'ROI:define' % currecntly used for definition of tesselation
        sn = subj_vec;
        glm = 8;
        hemis = [1,2];
        surf = '32';
        n_node = 362;
        vararginoptions(varargin,{'sn','glm','hemis','surf','n_node'});        
                        
        % loop over subjects
        for s = sn
            fprintf('%s...\n',sprintf('s%02d',s));
            R = cell(1,1);
            surfDir = fullfile(wbDir, sprintf('s%02d', s));
            
            idx = 0;
            for h = hemis
                file = fullfile(sprintf(glmDir,glm),sprintf('s%02d',s),'mask.nii');
                pathtosurf = fullfile(atlasDir,sprintf('FS_LR_%s',surf));
                surface = fullfile(pathtosurf, sprintf('fs_LR.%sk.%s.flat.surf.gii',surf,hem{h}));
                
                % ROI file
                P = gifti(fullfile(pathtosurf,sprintf('Icosahedron-%d.%sk.%s.label.gii',n_node,surf,hem{h})));
                selected_roi = op2multi_imana('TESSEL:select','hemi',h);
                for r = selected_roi
                    idx = idx+1;
                    R{idx}.type     = 'surf_nodes';
                    R{idx}.white    = fullfile(surfDir, sprintf('%s.%s.white.%sk.surf.gii', sprintf('s%02d', s), hem{h}, surf));
                    R{idx}.pial     = fullfile(surfDir, sprintf('%s.%s.pial.%sk.surf.gii', sprintf('s%02d', s), hem{h}, surf));
                    R{idx}.flat     = surface;
                    R{idx}.linedef  = [5,0,1]; % take 5 steps along node between white and pial
                    R{idx}.image    = file;
                    R{idx}.name     = [sprintf('s%02d', s) '_' num2str(r) '_' hem{h}];
                    R{idx}.location = find(P.cdata==r);
                end
            end
            R = region_calcregions(R);
            
            savename = fullfile(roiDir,sprintf('%s_tessel_%d.mat',sprintf('s%02d',s),n_node));
            save(savename,'R');
            %varargout = {R}; % return output structure
            fprintf('\nROIs have been defined for %s \n', sprintf('s%02d',s));
            clear R
        end
    case 'ROI:betas' % get the rois and make them ready for pcm
        glm = 8;
        sn  = subj_vec;
        old_num = 0;
        type = 'new';
        n_node =362;
        vararginoptions(varargin, {'sn','glm','type','n_node','old_num'});
        
        switch(type)
            case 'new'
                T = [];
            case 'add'
                T = load(fullfile(roiDir,sprintf('betas_glm%d_tessel_%d_N=%d.mat',glm,n_node,old_num)));
        end
        
        % harvest
        for s = sn % for each subj
            fprintf('\nSubject: s%02d\n', s) % output to user
            
            % load SPM
            load(fullfile(sprintf(glmDir,glm),sprintf('s%02d',s),'SPM.mat'));  % load subject's SPM data structure (SPM struct)
            % spm info - condition information
            spm_info = load(fullfile(sprintf(glmDir,glm),subj_name{s},'SPM_info.mat'));
            
            % update the naming of the directory
            if ~strcmp(fullfile(sprintf(glmDir,glm),sprintf('s%02d',s)),SPM.swd) % need to rename SPM
                SPM = spmj_move_rawdata(SPM,fullfile(imagingDir,sprintf('s%02d',s)));
            end
            
            % load ROI
            load(fullfile(roiDir,sprintf('%s_tessel_%d.mat',sprintf('s%02d',s),n_node)))
            
            % TR img info
            V = SPM.xY.VY;
            
            % get raw data for voxels in region
            for r = 1:length(R) % for each region
                
                % # number of the ROI according to the one we defined in
                % ROI things also the hemisphere
                roi_info = split(R{r}.name,'_')';
                roi = str2double(roi_info{2});
                hemi = find(strcmp(hem, roi_info{3}));
                
                % estimate region betas
                Y = region_getdata(V, R{r});  % Data Y is N x P (P is in order of transpose of R{r}.depth)
                [betaW,resMS,~,beta,~] = rsa.spm.noiseNormalizeBeta(Y,SPM,'normmode','overall','shrinkage',[]);
                
                mask = zeros(size(betaW,1),1);
                mask(1:length(spm_info.cond)) = spm_info.cond;
                mask = mask>0 & mask<11; % ignore intruction and intercept
                
                S.betaW                                 = {betaW(mask,:)};                               % multivariate pw: cells for voxel data b/c diff numVoxels
                S.betaUW                                = {bsxfun(@rdivide, beta(mask,:), sqrt(resMS))}; % univariate pw
                S.betaRAW                               = {beta(mask,:)};
                S.resMS                                 = {resMS};
                
                % voxel position
                S.volcoord = {R{r}.data'};
                S.flatcoord = {R{r}.flatcoord'};
                S.depth = {R{r}.depth'};
                S.ROI_side = hemi;
                S.ROI_type = roi;
                S.region = r;
                S.SN = s;

                T = addstruct(T, S);
                fprintf('%d.', r)
            end
        end
        % save T
        savename = fullfile(roiDir,sprintf('betas_glm%d_tessel_%d_N=%d.mat', glm,n_node,old_num+numel(sn)) );
        save(savename,'-struct','T');
        fprintf('\n')
        
        
    case 'modelfamily_on_surface' % after running model family in python, run this...
        hemi = 1;
        surf = '32';
        n_node = 362;
        sm = 3;
        vararginoptions(varargin, {'hemi','surf','n_node','sm'});
        
        h=hemi;
        
        % load the csv file
        R = readmatrix(fullfile('tessel_summary_cv.csv'));
        D = [];
        D.press=R(:,1); D.cue=R(:,2);
        D.numPresses=R(:,3); D.odd=R(:,4);
        D.roi=R(:,5); D.hemi=R(:,6); D.SN=R(:,7);
        clear R
        
        % combine the result from single finger and chord data
        T=tapply(D,{'SN','roi','hemi','odd','numPresses'},...
            {D.press,'sum','name','press'},...
            {D.cue,'sum','name','cue'});%'subset',D.numPresses==3
        
        % project back on surface
        % load the surface
        pathtosurf = fullfile(atlasDir,sprintf('FS_LR_%s',surf));
        P = gifti(fullfile(pathtosurf,sprintf('Icosahedron-%d.%sk.%s.label.gii',n_node,surf,hem{h})));
        label = P.cdata;
        
        press_single = zeros(length(label),length(subj_name));
        press_chord = zeros(length(label),length(subj_name));
        press_both = zeros(length(label),length(subj_name));
        
        cue_single = zeros(length(label),length(subj_name));
        cue_chord = zeros(length(label),length(subj_name));
        cue_both = zeros(length(label),length(subj_name));
        
        for s=unique(T.SN)'
            for r=unique(T.roi(T.hemi==h))'
                bf_press_single = T.press(T.roi==r&T.hemi==h&T.SN==s&T.numPresses==1);
                bf_press_chord = T.press(T.roi==r&T.hemi==h&T.SN==s&T.numPresses==3);
                bf_press_both = (bf_press_single+bf_press_chord)/2;
                
                bf_cue_single = T.cue(T.roi==r&T.hemi==h&T.SN==s&T.numPresses==1);
                bf_cue_chord = T.cue(T.roi==r&T.hemi==h&T.SN==s&T.numPresses==3);
                bf_cue_both = (bf_cue_single+bf_cue_chord)/2;

                press_single(label==r,s)=bf_press_single;
                press_chord(label==r,s)=bf_press_chord;
                press_both(label==r,s)=bf_press_both;
                
                cue_single(label==r,s)=bf_cue_single;
                cue_chord(label==r,s)=bf_cue_chord;
                cue_both(label==r,s)=bf_cue_both;
                
            end
        end
        groupDir = fullfile(wbDir,sprintf('group%sk',surf));
        
        O_press_single = surf_makeFuncGifti(press_single,'columnNames',subj_name, 'anatomicalStruct',hname{h});
        O_press_chord = surf_makeFuncGifti(press_chord,'columnNames',subj_name, 'anatomicalStruct',hname{h});
        O_press_both = surf_makeFuncGifti(press_both,'columnNames',subj_name, 'anatomicalStruct',hname{h});
        
        name_press_single = fullfile(groupDir,sprintf('group.%s.%s.func.gii',hem{h},'press_single'));
        name_press_chord = fullfile(groupDir,sprintf('group.%s.%s.func.gii',hem{h},'press_chord'));
        name_press_both = fullfile(groupDir,sprintf('group.%s.%s.func.gii',hem{h},'press_both'));
        
        save(O_press_single,name_press_single);
        save(O_press_chord,name_press_chord);
        save(O_press_both,name_press_both);
        
        
        O_cue_single = surf_makeFuncGifti(cue_single,'columnNames',subj_name, 'anatomicalStruct',hname{h});
        O_cue_chord = surf_makeFuncGifti(cue_chord,'columnNames',subj_name, 'anatomicalStruct',hname{h});
        O_cue_both = surf_makeFuncGifti(cue_both,'columnNames',subj_name, 'anatomicalStruct',hname{h});
        
        name_cue_single = fullfile(groupDir,sprintf('group.%s.%s.func.gii',hem{h},'cue_single'));
        name_cue_chord = fullfile(groupDir,sprintf('group.%s.%s.func.gii',hem{h},'cue_chord'));
        name_cue_both = fullfile(groupDir,sprintf('group.%s.%s.func.gii',hem{h},'cue_both'));
        
        save(O_cue_single,name_cue_single);
        save(O_cue_chord,name_cue_chord);
        save(O_cue_both,name_cue_both);
        
        % Apply smoothing to individuals (optional)
        if sm>0
            surface = fullfile(atlasDir, sprintf('FS_LR_%s/fs_LR.%sk.%s.flat.surf.gii', surf, surf, hem{h}));
            
            name_press_single = surf_smooth(name_press_single,'surf',surface,'kernel',sm);
            name_press_chord = surf_smooth(name_press_chord,'surf',surface,'kernel',sm);
            name_press_both = surf_smooth(name_press_both,'surf',surface,'kernel',sm);
            
            name_cue_single = surf_smooth(name_cue_single,'surf',surface,'kernel',sm);
            name_cue_chord = surf_smooth(name_cue_chord,'surf',surface,'kernel',sm);
            name_cue_both = surf_smooth(name_cue_both,'surf',surface,'kernel',sm);
            
        end
        
        % Calculating summary (mean and t) maps
        metric_press = gifti(name_press_single);
        cSPM_press = surf_getcSPM('onesample_t','data',metric_press.cdata);
        C.data(:,1) = cSPM_press.con.con; % mean
        C.c_name{1} = 'press_single';
        C.data(:,2) = cSPM_press.con.Z; % t
        C.c_name{2} = 't_press_single';
        
        metric_press = gifti(name_press_chord);
        cSPM_press = surf_getcSPM('onesample_t','data',metric_press.cdata);
        C.data(:,3) = cSPM_press.con.con; % mean
        C.c_name{3} = 'press_chord';
        C.data(:,4) = cSPM_press.con.Z; % t
        C.c_name{4} = 't_press_chord';
       
        metric_press = gifti(name_press_both);
        cSPM_press = surf_getcSPM('onesample_t','data',metric_press.cdata);
        C.data(:,5) = cSPM_press.con.con; % mean
        C.c_name{5} = 'press_both';
        C.data(:,6) = cSPM_press.con.Z; % t
        C.c_name{6} = 't_press_both';
        
        % cue
        metric_cue = gifti(name_cue_single);
        cSPM_press = surf_getcSPM('onesample_t','data',metric_cue.cdata);
        C.data(:,7) = cSPM_press.con.con; % mean
        C.c_name{7} = 'cue_single';
        C.data(:,8) = cSPM_press.con.Z; % t
        C.c_name{8} = 't_cue_single';
        
        metric_cue = gifti(name_cue_chord);
        cSPM_press = surf_getcSPM('onesample_t','data',metric_cue.cdata);
        C.data(:,9) = cSPM_press.con.con; % mean
        C.c_name{9} = 'cue_chord';
        C.data(:,10) = cSPM_press.con.Z; % t
        C.c_name{10} = 't_cue_chord';
       
        metric_cue = gifti(name_cue_both);
        cSPM_press = surf_getcSPM('onesample_t','data',metric_cue.cdata);
        C.data(:,11) = cSPM_press.con.con; % mean
        C.c_name{11} = 'cue_both';
        C.data(:,12) = cSPM_press.con.Z; % t
        C.c_name{12} = 't_cue_both';

        % ----------------------------------------------------------------
        % Save output
        O = surf_makeFuncGifti(C.data(:,1:2:12),'columnNames',C.c_name(1:2:12),'anatomicalStruct',hname{h});
        summaryfile = fullfile(groupDir,sprintf('summary.%s.%s.func.gii',hem{h},'modelfamilyBF'));
        save(O,summaryfile);
        
        O = surf_makeFuncGifti(C.data(:,2:2:12),'columnNames',C.c_name(2:2:12),'anatomicalStruct',hname{h});
        summaryfile = fullfile(groupDir,sprintf('summary.%s.%s.func.gii',hem{h},'modelfamilyT'));
        save(O,summaryfile);
        
        fprintf(1, 'Done.\n');
    case 'modelfamily_2comp' % use this one
        hemi = 1;
        surf = '32';
        n_node = 362;
        vararginoptions(varargin, {'hemi','surf','n_node'});
        
        h=hemi;
        
        % load the csv file
        R = readmatrix(fullfile('../python/notebooks/tessel_summary_cv.csv'));
        D = [];
        D.press=R(:,1); D.cue=R(:,2);
        D.numPresses=R(:,3); D.odd=R(:,4);
        D.roi=R(:,5); D.hemi=R(:,6); D.SN=R(:,7);
        clear R
        
        % decodability
        choice = op2multi_imana('TESSEL:select');
        
        % combine the result from single finger and chord data
        T=tapply(D,{'SN','roi','hemi','odd'},...
            {D.press,'sum','name','press'},... I think this should be sum
            {D.cue,'sum','name','cue'},'subset',D.hemi==hemi&ismember(D.roi,choice));
        
        T.pxp_press=zeros(size(T.press));
        T.pxp_cue=zeros(size(T.press));
        
        % calculate pxp
        for r=choice
            lme_press = [T.press(T.roi==r) zeros(22,1)];
            [~,~,~,pxp_press] = spm_BMS(lme_press);
            T.pxp_press(T.roi==r)=repmat(pxp_press(1), size(lme_press,1),1);
            
            lme_cue = [T.cue(T.roi==r) zeros(22,1)];
            [~,~,~,pxp_cue] = spm_BMS(lme_cue);
            T.pxp_cue(T.roi==r)=repmat(pxp_cue(1), size(lme_cue,1),1);
        end
        
        
        L=tapply(T,{'roi','hemi'},...
            {T.press,'myttest','name','t_press'},...
            {T.cue,'myttest','name','t_cue'},...
            {T.press,'mean','name','press'},...
            {T.cue,'mean','name','cue'},...
            {T.pxp_press,'mean','name','pxp_press'},...
            {T.pxp_cue,'mean','name','pxp_cue'});
        
        L.selected_press=L.press>=1&L.pxp_press>=0.75;
        L.selected_cue=L.cue>=1&L.pxp_cue>=0.75;
        
        
        
        %{
        pMap_vec = 1-tcdf([L.t_press],length(subj_vec)-1);
        pMap_vec_sorted = sort(pMap_vec);
        pMap_vec_sorted = pMap_vec_sorted(:);
        pThreshold = pMap_vec_sorted(find(cumsum(pMap_vec_sorted)>0.05,1));
        u_press=tinv(1-pThreshold,length(subj_vec)-1);
        
        pMap_vec = 1-tcdf(L.t_cue,length(subj_vec)-1);
        pMap_vec_sorted = sort(pMap_vec);
        pMap_vec_sorted = pMap_vec_sorted(:);
        pThreshold = pMap_vec_sorted(find(cumsum(pMap_vec_sorted)>0.05,1));
        u_cue=tinv(1-pThreshold,length(subj_vec)-1);
        %}
%         
        
        
        % project back on surface
        % load the surface
        pathtosurf = fullfile(atlasDir,sprintf('FS_LR_%s',surf));
        P = gifti(fullfile(pathtosurf,sprintf('Icosahedron-%d.%sk.%s.label.gii',n_node,surf,hem{h})));
        label = P.cdata;
        
        press = zeros(length(label),length(subj_name));
        cue = zeros(length(label),length(subj_name));
        
        
        for s=unique(T.SN)'
            for r=unique(T.roi(T.hemi==h))'

                %{
                if L.t_press(L.roi==r&L.hemi==h)>=u_press
                    bf_press = T.press(T.roi==r&T.hemi==h&T.SN==s);
                else
                    bf_press = 0;
                end
                
                if L.t_cue(L.roi==r&L.hemi==h)>=u_cue
                    bf_cue = T.cue(T.roi==r&T.hemi==h&T.SN==s);
                else
                    bf_cue = 0;
                end
                %}
                if L.selected_press(L.roi==r&L.hemi==h)
                    bf_press = T.press(T.roi==r&T.hemi==h&T.SN==s);
                else
                    bf_press = 0;
                end
                
                if L.selected_cue(L.roi==r&L.hemi==h)
                    bf_cue = T.cue(T.roi==r&T.hemi==h&T.SN==s);
                else
                    bf_cue = 0;
                end
                
                press(label==r,s)=bf_press;
                cue(label==r,s)=bf_cue;
                
            end
        end
        groupDir = fullfile(wbDir,sprintf('group%sk',surf));
        
        
        O_press = surf_makeFuncGifti(press,'columnNames',subj_name, 'anatomicalStruct',hname{h});
        O_cue = surf_makeFuncGifti(cue,'columnNames',subj_name, 'anatomicalStruct',hname{h});
        
        name_press = fullfile(groupDir,sprintf('group.%s.%s.func.gii',hem{h},'press'));
        name_cue = fullfile(groupDir,sprintf('group.%s.%s.func.gii',hem{h},'cue'));
        
        save(O_press,name_press);
        save(O_cue,name_cue);
        
        
        
        % Calculating summary (mean and t) maps
        metric = gifti(name_press);
        cSPM = surf_getcSPM('onesample_t','data',metric.cdata);
        C.data(:,1) = cSPM.con.con; % mean
        C.c_name{1} = 'press';
        C.data(:,2) = cSPM.con.Z; % t
        C.c_name{2} = 't_press';

        
        % cue
        metric = gifti(name_cue);
        cSPM = surf_getcSPM('onesample_t','data',metric.cdata);
        C.data(:,3) = cSPM.con.con; % mean
        C.c_name{3} = 'cue';
        C.data(:,4) = cSPM.con.Z; % t
        C.c_name{4} = 't_cue';
        

        % ----------------------------------------------------------------
        % Save output
        O = surf_makeFuncGifti(C.data,'columnNames',C.c_name,'anatomicalStruct',hname{h});
        summaryfile = fullfile(groupDir,sprintf('summary.%s.%s.func.gii',hem{h},'modelfamily_2compnew'));
        save(O,summaryfile);
        
        
        fprintf(1, 'Done.\n');
        
        
        
        % also save a label file
        p=C.data(:,1);
        c=C.data(:,3);
        label=zeros(size(p));
        label(p>0&isnan(c))=1;
        label(c>0&isnan(p))=2;
        label(c>0&p>0)=3;
        
        pathtosurf = fullfile(atlasDir,sprintf('FS_LR_%s',surf));
        % Brodmann ROI file
        label_brodmann = fullfile(pathtosurf,sprintf('ROI.%sk.%s.label.gii',surf,hem{hemi}));
        P1 = gifti(label_brodmann);
        name1=P1.labels.name;
        
        if sum(label==3)>0
            name={name1{1},'press','cue','both'};
            color=[...
                0,0,0,0;...
                0.403921568627451, 0.0, 0.05098039215686274,1;...
                0.03137254901960784, 0.18823529411764706, 0.4196078431372549,1;...
                0.4353    0.1882    0.4706,1];
        else
            name={name1{1},'press','cue'};
            color=[...
                0,0,0,0;...
                0.403921568627451, 0.0, 0.05098039215686274,1;...
                0.03137254901960784, 0.18823529411764706, 0.4196078431372549,1];
        end
        
        
        G = surf_makeLabelGifti(label,'labelNames',name,'labelRGBA',color,'anatomicalStruct',hname{hemi});
        file = fullfile(wbDir,sprintf('group%sk',surf),sprintf('%s.%s.label.gii','pressANDcue',hem{hemi}));
        save(G,file);
    case 'modelfamily_morecomp' % also added the independent model
        hemi = 1;
        surf = '32';
        n_node = 362;
        vararginoptions(varargin, {'hemi','surf','n_node'});
        
        h=hemi;
        
        % load the csv file
        R = readmatrix(fullfile('tessel_summary_morecomp.csv'));
        D = [];
        D.press=R(:,1); D.cue=R(:,2); D.indep=R(:,3);
        D.numPresses=R(:,4); D.odd=R(:,5);
        D.roi=R(:,6); D.hemi=R(:,7); D.SN=R(:,8);
        clear R
        
        % combine the result from single finger and chord data
        T=tapply(D,{'SN','roi','hemi','odd'},...
            {D.press,'sum','name','press'},...
            {D.cue,'sum','name','cue'},...
            {D.indep,'sum','name','indep'});
        
        L=tapply(T,{'roi','hemi'},...
            {T.press,'myttest','name','t_press'},...
            {T.cue,'myttest','name','t_cue'},...
            {T.indep,'myttest','name','t_indep'});
        
        L=tapply(T,{'roi','hemi'},...
            {T.press,'myttest','name','t_press'},...
            {T.cue,'myttest','name','t_cue'},...
            {T.indep,'myttest','name','t_indep'},...
            {T.press,'mean','name','press'},...
            {T.cue,'mean','name','cue'},...
            {T.indep,'mean','name','indep'});
        
        % project back on surface
        % load the surface
        pathtosurf = fullfile(atlasDir,sprintf('FS_LR_%s',surf));
        P = gifti(fullfile(pathtosurf,sprintf('Icosahedron-%d.%sk.%s.label.gii',n_node,surf,hem{h})));
        label = P.cdata;
        
        press = zeros(length(label),length(subj_name));
        cue = zeros(length(label),length(subj_name));
        indep = zeros(length(label),length(subj_name));
                
        for s=unique(T.SN)'
            for r=unique(T.roi(T.hemi==h))'
                %bf_press = T.press(T.roi==r&T.hemi==h&T.SN==s);
                %bf_cue = T.cue(T.roi==r&T.hemi==h&T.SN==s);
                %bf_indep = T.indep(T.roi==r&T.hemi==h&T.SN==s);
               
                %press(label==r,s)=bf_press;
                %cue(label==r,s)=bf_cue;
                %indep(label==r,s)=bf_indep;
                
                
                if L.press(L.roi==r&L.hemi==h)>=0.5
                    bf_press = T.press(T.roi==r&T.hemi==h&T.SN==s);
                else
                    bf_press = 0;
                end
                
                if L.cue(L.roi==r&L.hemi==h)>=0.5
                    bf_cue = T.cue(T.roi==r&T.hemi==h&T.SN==s);
                else
                    bf_cue = 0;
                end
                
                if L.indep(L.roi==r&L.hemi==h)>=0.5
                    bf_indep = T.cue(T.roi==r&T.hemi==h&T.SN==s);
                else
                    bf_indep = 0;
                end
                
                press(label==r,s)=bf_press;
                cue(label==r,s)=bf_cue;
                indep(label==r,s)=bf_indep;
                
            end
        end
        groupDir = fullfile(wbDir,sprintf('group%sk',surf));
        
        
        O_press = surf_makeFuncGifti(press,'columnNames',subj_name, 'anatomicalStruct',hname{h});
        O_cue = surf_makeFuncGifti(cue,'columnNames',subj_name, 'anatomicalStruct',hname{h});
        O_indep = surf_makeFuncGifti(indep,'columnNames',subj_name, 'anatomicalStruct',hname{h});
        
        name_press = fullfile(groupDir,sprintf('group.%s.%s.func.gii',hem{h},'press'));
        name_cue = fullfile(groupDir,sprintf('group.%s.%s.func.gii',hem{h},'cue'));
        name_indep = fullfile(groupDir,sprintf('group.%s.%s.func.gii',hem{h},'indep'));
        
        save(O_press,name_press);
        save(O_cue,name_cue);
        save(O_indep,name_indep);
        
        
        
        % Calculating summary (mean and t) maps
        metric = gifti(name_press);
        cSPM = surf_getcSPM('onesample_t','data',metric.cdata);
        C.data(:,1) = cSPM.con.con; % mean
        C.c_name{1} = 'press';
        C.data(:,2) = cSPM.con.Z; % t
        C.c_name{2} = 't_press';

        
        % cue
        metric = gifti(name_cue);
        cSPM = surf_getcSPM('onesample_t','data',metric.cdata);
        C.data(:,3) = cSPM.con.con; % mean
        C.c_name{3} = 'cue';
        C.data(:,4) = cSPM.con.Z; % t
        C.c_name{4} = 't_cue';
        
        % indep
        metric = gifti(name_indep);
        cSPM = surf_getcSPM('onesample_t','data',metric.cdata);
        C.data(:,5) = cSPM.con.con; % mean
        C.c_name{5} = 'cue_single';
        C.data(:,6) = cSPM.con.Z; % t
        C.c_name{6} = 't_indep';

        % ----------------------------------------------------------------
        % Save output
        O = surf_makeFuncGifti(C.data,'columnNames',C.c_name,'anatomicalStruct',hname{h});
        summaryfile = fullfile(groupDir,sprintf('summary.%s.%s.func.gii',hem{h},'modelfamily_morecomp'));
        save(O,summaryfile);
        
        
        fprintf(1, 'Done.\n');
        
        
        % also save a label file
        p=C.data(:,1);
        c=C.data(:,5);
        label=zeros(size(p));
        label(p>0&isnan(c))=1;
        label(c>0&isnan(p))=2;
        label(c>0&p>0)=3;
        
        pathtosurf = fullfile(atlasDir,sprintf('FS_LR_%s',surf));
        % Brodmann ROI file
        label_brodmann = fullfile(pathtosurf,sprintf('ROI.%sk.%s.label.gii',surf,hem{hemi}));
        P1 = gifti(label_brodmann);
        name1=P1.labels.name;
        
        if sum(label==3)>0
            name={name1{1},'press','indep','both'};
            color=[0,0,0,0;1,0,0,1;0,0,1,1;1,0,1,1];
        else
            name={name1{1},'press','indep'};
            color=[0,0,0,0;1,0,0,1;0,0,1,1];
        end
        
        
        G = surf_makeLabelGifti(label,'labelNames',name,'labelRGBA',color,'anatomicalStruct',hname{hemi});
        file = fullfile(wbDir,sprintf('group%sk',surf),sprintf('%s.%s.label.gii','pressANDindep',hem{hemi}));
        save(G,file);
    case 'reliabilityType1_on_surface' % cross subject reliability
        hemi = 1;
        surf = '32';
        n_node = 362;
        vararginoptions(varargin, {'hemi','surf','n_node','sm'});
        
        h=hemi;
        
        % load the csv file
        R = readmatrix(fullfile('tessel_reliability_type1.csv'));
        D = [];
        D.numPresses=R(:,1); D.corr=R(:,2);
        D.roi=R(:,3); D.hemi=R(:,4);
        clear R
        
        % project back on surface
        % load the surface
        pathtosurf = fullfile(atlasDir,sprintf('FS_LR_%s',surf));
        P = gifti(fullfile(pathtosurf,sprintf('Icosahedron-%d.%sk.%s.label.gii',n_node,surf,hem{h})));
        label = P.cdata;
        
        single = zeros(length(label),1);
        chord = zeros(length(label),1);
        both = zeros(length(label),1);
        
        for r=unique(D.roi(D.hemi==h))'
            corr_single = D.corr(D.roi==r&D.hemi==h&D.numPresses==1);
            corr_chord = D.corr(D.roi==r&D.hemi==h&D.numPresses==3);
            corr_both = (corr_single+corr_chord)/2;

            single(label==r,1)=corr_single;
            chord(label==r,1)=corr_chord;
            both(label==r,1)=corr_both;
        end
        
        groupDir = fullfile(wbDir, sprintf('group%sk', surf));
        

        C.data(:,1) = single; 
        C.c_name{1} = 'single';
        C.data(:,2) = chord;
        C.c_name{2} = 'chord';
        C.data(:,3) = both;
        C.c_name{3} = 'both';

        % Save output
        groupDir = fullfile(wbDir, sprintf('group%sk',surf));
        O = surf_makeFuncGifti(C.data,'columnNames',C.c_name,'anatomicalStruct',hname{h});
        summaryfile = fullfile(groupDir,sprintf('summary.%s.%s.func.gii',hem{h},'reliabilityType1'));
        
        save(O,summaryfile);
        fprintf(1,'Done.\n');
    case 'reliabilityType2_on_surface' % cross subject reliability
        hemi = 1;
        surf = '32';
        n_node = 362;
        vararginoptions(varargin, {'hemi','surf','n_node','sm'});
        
        h=hemi;
        
        % load the csv file
        R = readmatrix(fullfile('tessel_reliability_type2.csv'));
        D = [];
        D.odd=R(:,1); D.corr=R(:,2);
        D.roi=R(:,3); D.hemi=R(:,4);
        clear R
        
        % project back on surface
        % load the surface
        pathtosurf = fullfile(atlasDir,sprintf('FS_LR_%s',surf));
        P = gifti(fullfile(pathtosurf,sprintf('Icosahedron-%d.%sk.%s.label.gii',n_node,surf,hem{h})));
        label = P.cdata;
        
        odd = zeros(length(label),1);
        even = zeros(length(label),1);
        both = zeros(length(label),1);
        
        for r=unique(D.roi(D.hemi==h))'
            corr_odd = D.corr(D.roi==r&D.hemi==h&D.odd==1);
            corr_even = D.corr(D.roi==r&D.hemi==h&D.odd==0);
            corr_both = (corr_odd+corr_even)/2;

            odd(label==r,1)=corr_odd;
            even(label==r,1)=corr_even;
            both(label==r,1)=corr_both;
        end
        
        groupDir = fullfile(wbDir, sprintf('group%sk', surf));
        

        C.data(:,1) = odd; 
        C.c_name{1} = 'odd';
        C.data(:,2) = even;
        C.c_name{2} = 'even';
        C.data(:,3) = both;
        C.c_name{3} = 'both';

        % Save output
        groupDir = fullfile(wbDir, sprintf('group%sk',surf));
        O = surf_makeFuncGifti(C.data,'columnNames',C.c_name,'anatomicalStruct',hname{h});
        summaryfile = fullfile(groupDir,sprintf('summary.%s.%s.func.gii',hem{h},'reliabilityType2'));
        
        save(O,summaryfile);
        fprintf(1,'Done.\n');
        
        
    case 'modelFamily_tests'
        hemi = 1;
        surf = '32';
        n_node = 362;
        vararginoptions(varargin, {'hemi','surf','n_node','sm'});
        
        h=hemi;
        
        % load the csv file
        R = readmatrix(fullfile('tessel_summary_cv.csv'));
        %R = readmatrix(fullfile('tessel_var_cv.csv'));
        %R = readmatrix(fullfile('tessel_summary_morecomp.csv'));
        
        D = [];
        D.press=R(:,1); D.cue=R(:,2);
        D.numPresses=R(:,3); D.odd=R(:,4);
        D.roi=R(:,5); D.hemi=R(:,6); D.SN=R(:,7);
        %D.numPresses=R(:,4); D.odd=R(:,5);
        %D.roi=R(:,6); D.hemi=R(:,7); D.SN=R(:,8);
        clear R
        
        % combine the result from single finger and chord data
        T=tapply(D,{'SN','roi','hemi'},...
            {D.press,'sum','name','press'},...
            {D.cue,'sum','name','cue'});
        
        % roi for overlap-nonoverlap
        selected_roi = op2multi_imana('TESSEL:online_pre');
        
        T=getrow(T,T.hemi==h&ismember(T.roi,selected_roi));
        
        S=tapply(T,{'roi'},...
            {T.press,'myttest','name','press'},...
            {T.cue,'myttest','name','cue'},...
            {T.press,'mean','name','pressM'},...
            {T.cue,'mean','name','cueM'},...
            {T.press-T.cue,'myttest','name','diff'});
        
        
        % threshold
        u=tinv(1-0.05/length(selected_roi),length(subj_vec)-1);
        
        
        % STEP 1: order the p values

        pMap_vec = 1-tcdf([S.press;S.cue],21);
        pMap_vec_sorted=sort(pMap_vec);
        pMap_vec_sorted = pMap_vec_sorted(:);

        q=0.05;
        % STEP 2: find the critical p value that yields an average FDR smaller than q
        pThreshold = pMap_vec_sorted(find(cumsum(pMap_vec_sorted) > q, 1));

        u=tinv(1-pThreshold,length(subj_vec)-1);
        
        %Punc=1-tcdf(u,con.df(2));
        
        % project back on the surface
        
        S.pressM(S.press>u)
        keyboard()
        %
        
        % project back on surface
        % load the surface
        pathtosurf = fullfile(atlasDir,sprintf('FS_LR_%s',surf));
        P = gifti(fullfile(pathtosurf,sprintf('Icosahedron-%d.%sk.%s.label.gii',n_node,surf,hem{h})));
        label = P.cdata;
        
        press = zeros(length(label),1);
        cue = zeros(length(label),1);
        diff = zeros(length(label),1);
        
        for r=unique(S.roi)'
%             press(label==r,1)=S.press(S.roi==r);
%             cue(label==r,1)=S.cue(S.roi==r);
%             diff(label==r,1)=S.diff(S.roi==r);
            if S.press(S.roi==r)>=u
                press(label==r,1)=S.press(S.roi==r);
            end
            if S.cue(S.roi==r)>=u
                cue(label==r,1)=S.cue(S.roi==r);
            end
            if S.diff(S.roi==r)>=u
                diff(label==r,1)=S.diff(S.roi==r);
            end

%             if S.press(S.roi==r)>=0
%                 press(label==r,1)=S.press(S.roi==r);
%             end
%             if S.cue(S.roi==r)>=0
%                 cue(label==r,1)=S.cue(S.roi==r);
%             end
%             if S.diff(S.roi==r)>=0
%                 diff(label==r,1)=S.diff(S.roi==r);
%             end
        end
                

        C.data(:,1) = press; 
        C.c_name{1} = 'press';
        C.data(:,2) = cue;
        C.c_name{2} = 'cue';
        C.data(:,3) = diff;
        C.c_name{3} = 'diff';

        % Save output
        groupDir = fullfile(wbDir, sprintf('group%sk',surf));
        O = surf_makeFuncGifti(C.data,'columnNames',C.c_name,'anatomicalStruct',hname{h});
        summaryfile = fullfile(groupDir,sprintf('summary.%s.%s.func.gii',hem{h},'overlapRegBF3'));
        
        save(O,summaryfile);
        fprintf(1,'Done.\n');
    case 'modelFamily_test2' % use this one
        hemi = 1;
        surf = '32';
        n_node = 362;
        vararginoptions(varargin, {'hemi','surf','n_node'});
        
        h=hemi;
        
        % load the csv file
        R = readmatrix(fullfile('tessel_summary_cv.csv'));
        D = [];
        D.press=R(:,1); D.cue=R(:,2);
        D.numPresses=R(:,3); D.odd=R(:,4);
        D.roi=R(:,5); D.hemi=R(:,6); D.SN=R(:,7);
        clear R
        

        % roi for overlap-nonoverlap
        %choice = op2multi_imana('TESSEL:online_pre');
        
        % combine the result from single finger and chord data
        T=tapply(D,{'SN','roi','hemi','odd'},...
            {D.press,'sum','name','press'},... I think this should be sum
            {D.cue,'sum','name','cue'},'subset',D.hemi==hemi); % ismember(D.roi,choice)
        
        T.pxp_press=zeros(size(T.press));
        T.pxp_cue=zeros(size(T.press));
        
        % calculate pxp
        for r=unique(T.roi)'
            lme_press = [T.press(T.roi==r) zeros(22,1)];
            [~,~,~,pxp_press] = spm_BMS(lme_press);
            T.pxp_press(T.roi==r)=repmat(pxp_press(1), size(lme_press,1),1);
            
            lme_cue = [T.cue(T.roi==r) zeros(22,1)];
            [~,~,~,pxp_cue] = spm_BMS(lme_cue);
            T.pxp_cue(T.roi==r)=repmat(pxp_cue(1), size(lme_cue,1),1);
        end
        % T.roi(T.press==max(T.press))
        
        
        L=tapply(T,{'roi','hemi'},...
            {T.press,'myttest','name','t_press'},...
            {T.cue,'myttest','name','t_cue'},...
            {T.press,'mean','name','press'},...
            {T.cue,'mean','name','cue'},...
            {T.pxp_press,'mean','name','pxp_press'},...
            {T.pxp_cue,'mean','name','pxp_cue'});
        
        L.selected_press=L.press>=1&L.pxp_press>=0.75;
        L.selected_cue=L.cue>=1&L.pxp_cue>=0.75;
        
        keyboard()
        
end
