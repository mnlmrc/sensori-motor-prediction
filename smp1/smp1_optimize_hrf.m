function p1 = smp1_optimize_hrf(p0, varargin)

    sn = [];
    vararginoptions(varargin, {'sn'})
    
    if isfolder('/cifs/diedrichsen/data/SensoriMotorPrediction/smp1/')
        resFile = sprintf('/cifs/diedrichsen/data/SensoriMotorPrediction/smp1/glm5/subj%d/ResMS.nii', sn);
    elseif isfolder('/Volumes/diedrichsen_data$/data/SensoriMotorPrediction/smp1/')
        resFile = sprintf('/Volumes/diedrichsen_data$/data/SensoriMotorPrediction/smp1/glm5/subj%d/ResMS.nii', sn);
    end
    
    fun = @(p0) smp1_calc_avg_res(p0, 'sn', sn); 

    p1 = fminsearch(fun, p0);
    