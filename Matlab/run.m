try
    Main_2
    Main_3
catch ME
    errorMessage = sprintf('Error in function %s() at line %d.\n\nError Message:\n%s', ...
                    ME.stack(1).name, ME.stack(1).line, ME.message);
    fprintf(1, '%s\n', errorMessage); % To command window.
    
    fullFileName = 'Error Log.txt';
    fid = fopen(fullFileName, 'at');
    fprintf(fid, '%s\n', errorMessage); % To file
    fclose(fid);
end

%% get NLE for every patient
% clear;
% close all;
% warning('off','all');
% addpath(genpath('MedicalImageProcessingToolbox'));
% addpath(genpath('ReadData3D'));
% XlsxPath = ['.' filesep 'data xlsx' filesep];
% load('fibrosis.mat'); 
% SIZE = max(size(Fibrosis));
% NLE = zeros(SIZE,1);
% for i = 1:SIZE
%     pid = Fibrosis(i).PID;
%     if isfile([XlsxPath pid '.xlsx'])
%         Statistic = readtable([XlsxPath pid '.xlsx'], 'ReadRowNames',true, 'Sheet', 'S');
%     else
%         disp(['file ' pid '.xlsx not exist']);
%         return;
%     end
%     
%     %% Calculate NLE (pass to function later)
%     try
%         LiverSI_20min_normalize = Statistic.x20(1:7);
%     catch 
%         LiverSI_20min_normalize = Statistic.x10(1:7);
%     end
%     clear Statistic
%     %SpleenSI_20min_normalize = Statistic.x20(8:10);
%     LiverSI_20min_normalize = mean(LiverSI_20min_normalize);
%     %SpleenSI_20min_normalize = mean(SpleenSI_20min_normalize);
%     NLE(i) = LiverSI_20min_normalize - 1;
% end