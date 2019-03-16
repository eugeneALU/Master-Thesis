% Script for testing dicom files 

wd = pwd;
cd('/Volumes/data_radiofys/Data/MR/Anders/')
[FileName,PathName] = uigetfile('*.*','Choose a DICOM file');
cd(wd)
dicomfile = fullfile(PathName,FileName);
info = dicominfo(dicomfile);

% fileDir = uigetdir('Choose dicom directory');
% 
% dicomStruct = dir(fileDir);
% nFiles = 0;
% info = [];
% 
% for fileIDX = 1:length(dicomStruct)
%     try
%         
%         fileName = fullfile(fileDir,dicomStruct(fileIDX).name);
%         info{nFiles + 1} = dicominfo(fileName);
%         nFiles = nFiles + 1;
%     catch
%         
%         warning(fileName)
%         
%     end
% end


