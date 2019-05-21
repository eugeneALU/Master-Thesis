clear;
close all;
XlsxPath = ['..' filesep 'data xlsx' filesep];
ImagePath = ['..' filesep 'Image and Mask' filesep 'HIFI' filesep];
addpath(genpath('../MedicalImageProcessingToolbox'));
addpath(genpath('../ReadData3D'));

load('HIFIfibrosis.mat');
SIZE = max(size(Fibrosis));
SLICE = 0;
AREA_AVG = 0;
SLICE_AVG = 0;
for i = 1:SIZE
    pid = Fibrosis(i).PID;
    disp (['Check ' pid])
    % try Szesze mask
    if isfile([ImagePath pid filesep pid '_Liver_SL.vtk'])
        info = vtk_read_header([ImagePath pid filesep pid '_Liver_SL.vtk']);
        temp = vtk_read_volume(info);
        mask = temp > 0.5;
        disp('SL')
    % try  Ola mask
    elseif isfile([ImagePath pid filesep pid '_Liver.vtk'])
        info = vtk_read_header([ImagePath pid filesep pid '_Liver.vtk']);
        info.BitDepth = 32;
        temp = vtk_read_volume(info);
        mask = (temp) > 0.5; %for HIFI dataset we don't need to /255 (it's already done)
    else 
        disp([pid ' mask is not exist']);
    end
    
    if (max(mask(:)) < 1)
        disp([pid ' mask is useless']);
    end
%     slicenum = size(mask);
%     slicenum = slicenum(3);
%     for j = 1: slicenum
%         SUM = sum(mask(:,:,slicenum),[1,2]);
%         if (SUM >= 2500)
%             AREA_AVG = AREA_AVG + SUM;
%             SLICE = SLICE  + 1;
%         end
%     end
%    
%     SLICE_AVG = SLICE_AVG + slicenum;
end
% AREA_AVG = AREA_AVG / SLICE;
% SLICE_AVG = SLICE_AVG / SIZE;

%total average slice = 96.5934
%THRESHOLD = 2500 WE GET 678 samples 
% average area = 5202.5