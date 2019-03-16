%% get image of certain slice of certain patient
% indicate in pid, stage, slice
%%
clear;
close all;
warning('off','all');
addpath(genpath('MedicalImageProcessingToolbox'));
addpath(genpath('ReadData3D'));

XlsxPath = ['..' filesep 'data xlsx' filesep];
ImagePath = ['..' filesep 'Image and Mask' filesep 'NILB' filesep];
StoreImagePath = ['..' filesep 'Image_train' filesep];
StoreMaskedImagePath = ['..' filesep 'MaskedImage' filesep];
StoreLiverMaskPath = ['..' filesep 'LiverMask' filesep];
DataPath = ['..' filesep 'Data' filesep];

pid = 'BE63';
stage = 1;
stage_str = '1';
slice = 27;
slice_str = '27';


%% get image file
if isfile([ImagePath pid filesep pid '.mhd'])
    im = read_mhd([ImagePath pid filesep pid '.mhd']);
    LiverImage = im.data;
else
    disp(['NO ' pid ' image file']);
    return;
end  
clear im;

%% load mask
% try Szesze mask
if isfile([ImagePath pid filesep pid '_Liver_SL.vtk'])
    info = vtk_read_header([ImagePath pid filesep pid '_Liver_SL.vtk']);
    temp = vtk_read_volume(info);
    mask = temp > 0.5;
% try  Ola mask
elseif isfile([ImagePath pid filesep pid '_Liver.vtk'])
    info = vtk_read_header([ImagePath pid filesep pid '_Liver.vtk']);
    info.BitDepth = 32;
    temp = vtk_read_volume(info);
    mask = (temp/255) > 0.5;
else 
    disp(['NO Live mask file. Something error in ' pid]);
    return;
end
clear temp info;

%% get according slice mask
LiverMask = mask(:,:,slice);
%% get according slice image
Image = LiverImage(:,:,slice);
%% Get mask enhance image
MaskedImage = Image;
MaskedImage(~LiverMask) = 0;

%%
imwrite(mat2gray(Image), [StoreImagePath filesep stage_str filesep pid '_' slice_str '_image.jpg']);
imwrite(mat2gray(MaskedImage),[StoreMaskedImagePath filesep stage_str filesep pid '_' slice_str '_maskedimage.jpg']);
imwrite(LiverMask,[StoreLiverMaskPath filesep stage_str filesep pid '_' slice_str '_livermask.jpg']);
            