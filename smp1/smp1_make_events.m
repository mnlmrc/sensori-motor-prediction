clc
close all
clear

%%
if isfolder("/Volumes/diedrichsen_data$//data/SensoriMotorPrediction/smp1")
    workdir = "/Volumes/diedrichsen_data$//data/SensoriMotorPrediction/smp1";
else
    fprintf('Workdir not found. Mount or connect to server and try again.');
end
    

baseDir = (sprintf('%s/',workdir));                                        % Base directory of the project
behavDir = 'behavioural';                                                  % Behavioural directory
targetDir = 'target';
session = 'sess1';
subj_id = 'subj100';
glmDir = 'glm1';

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
            cue = '100%';
        case 21
            cue = '75%';
        case 44
            cue = '50%';
        case 12
            cue = '25%';
        case 93
            cue = '0%';
    end

    switch exec.stimFinger(ntrial)
        case 91999
            stimFinger = 'index';
        case 99919
            stimFinger = 'ring';
    end
    
    exec.eventtype{ntrial, 1} = [cue '_' stimFinger '_exec'];
    
end

%% plan go
planGo.BN = D.BN(go);
planGo.TN = D.TN(go);
planGo.cue = D.cue(go);
planGo.stimFinger = D.stimFinger(go);
planGo.Onset = D.startTimeReal(go) + D.baselineWait(go);
planGo.Duration = D.planTime(go);

for ntrial = 1:length(exec.BN)

    switch planGo.cue(ntrial)
        case 39
            cue = '100%';
        case 21
            cue = '75%';
        case 44
            cue = '50%';
        case 12
            cue = '25%';
        case 93
            cue = '0%';
    end

    switch planGo.stimFinger(ntrial)
        case 91999
            stimFinger = 'index';
        case 99919
            stimFinger = 'ring';
    end
    
    planGo.eventtype{ntrial, 1} = [cue '_' stimFinger '_planGo'];
    
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
            cue = '100%';
        case 21
            cue = '75%';
        case 44
            cue = '50%';
        case 12
            cue = '25%';
        case 93
            cue = '0%';
    end
    
    planNoGo.eventtype{ntrial, 1} = [cue '_planNoGo'];
    
end

%% rest
rest = [];
rest.BN = [];
rest.TN = [];
rest.cue = [];
rest.stimFinger = [];
rest.Onset = [];
rest.Duration = [];
rest.eventtype = [];
for run = 1:max(D.BN)

    % retrieve trial log
    TN = D.TN(D.BN == run);
    startTimeReal = D.startTimeReal(D.BN == run);
    baselineWait = D.baselineWait(D.BN == run);
    planTime = D.planTime(D.BN == run);
    execMaxTime  = D.execMaxTime(D.BN == run);
    iti = D.iti(D.BN == run);
    
    % look target
    Dt = dload(fullfile(baseDir, targetDir, subj_id, ['smp1_' subj_id(5:end) sprintf('_%02d_scanning.tgt', run)]));
    startTime = Dt.startTime;
    startTimeDiff = diff(startTime);
    idx = find(startTimeDiff > 10000);
    onset = startTimeReal(idx) + baselineWait(idx) + planTime(idx) + execMaxTime(idx) + iti(idx);
    
    % add to rest
    rest.BN = [rest.BN; run; run; run];
    rest.TN = [rest.TN; TN(idx)];
    rest.cue = [rest.cue; NaN; NaN; NaN;];
    rest.stimFinger = [rest.stimFinger;NaN; NaN; NaN;];
    rest.Onset = [rest.Onset; onset];
    rest.Duration = [rest.Duration;12000; 12000; 12000;];
    rest.eventtype = [rest.eventtype;{'rest'}; {'rest'}; {'rest'}];

end

%% make table

exec = struct2table(exec);
planGo = struct2table(planGo);
planNoGo = struct2table(planNoGo);
rest = struct2table(rest);
events = [exec; planGo; planNoGo;rest];

%% convert to secs
events.Onset = events.Onset ./ 1000;
events.Duration = events.Duration ./ 1000;

%% export
 output_folder = fullfile(baseDir, glmDir, subj_id, session);
 if ~exist(output_folder, "dir")
    mkdir(output_folder);
 end
writetable(events, fullfile(output_folder,  'events.tsv'), 'FileType', 'text', 'Delimiter','\t')





