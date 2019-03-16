clear;
close all;
mkdir Image
DirPath = ['..' filesep 'Image'];
load('fibrosis.mat');
SIZE = max(size(Fibrosis));
for i = 1:SIZE
    pid = Fibrosis(i).PID;
    mkdir (DirPath, pid);
end