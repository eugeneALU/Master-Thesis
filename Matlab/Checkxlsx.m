clear;
close all;
warning('off','all');
XlsxPath = ['..' filesep 'data xlsx' filesep];
ImagePath = ['..' filesep 'Image and Mask' filesep 'NILB' filesep];

load('fibrosis.mat');
SIZE = max(size(Fibrosis));
for i = 1:SIZE
    pid = Fibrosis(i).PID;
    Statistic = readtable([XlsxPath pid '.xlsx'], 'ReadRowNames',true, 'Sheet', 'S');
    try
        X = Statistic.x20(1:7);
    catch
        disp([pid ' NO statistic 20 min data']);
    end
end