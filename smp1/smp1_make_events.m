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
subj_id = 'subj101';
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

for ntrial = 1:length(exec.BN)

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
    planNoGo.stimFinger_id{ntrial, 1} = stimFinger;
    planNoGo.epoch{ntrial, 1} = 'plan';
    planNoGo.instruction{ntrial, 1} = 'nogo';
    
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
rest.cue_id = [];
rest.stimFinger_id = [];
rest.epoch = [];
rest.instruction = [];
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
    rest.cue_id = [rest.cue_id;{'rest'}; {'rest'}; {'rest'}];
    rest.stimFinger_id = [rest.stimFinger_id;{'rest'}; {'rest'}; {'rest'}];
    rest.epoch = [rest.epoch;{'rest'}; {'rest'}; {'rest'}];
    rest.instruction = [rest.instruction;{'rest'}; {'rest'}; {'rest'}];

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
 output_folder = fullfile(baseDir, behavDir, subj_id);
 if ~exist(output_folder, "dir")
    mkdir(output_folder);
 end
writetable(events, fullfile(output_folder,  'events.tsv'), 'FileType', 'text', 'Delimiter','\t')





