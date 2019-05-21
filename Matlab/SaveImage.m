clear;
close all;
warning('off','all');
addpath(genpath('../MedicalImageProcessingToolbox'));
addpath(genpath('../ReadData3D'));

StoreImagePath = ['..' filesep 'Image_HIFI' filesep];
StoreMaskedImagePath = ['..' filesep 'MaskedImage_HIFI' filesep];
StoreLiverMaskPath = ['..' filesep 'LiverMask_HIFI' filesep];

ImagePath = ['..' filesep 'Image and Mask' filesep 'HIFI' filesep];
load('HIFIfibrosis.mat');   % load in Fibrosis struct
SIZE = max(size(Fibrosis));
AreaThreshold = 2500;

for i = 1:SIZE
    pid = Fibrosis(i).PID;
    stage = Fibrosis(i).fibrosis;
    %% get image file
    if isfile([ImagePath pid filesep pid '.mhd'])
        im = read_mhd([ImagePath pid filesep pid '.mhd']);
        LiverImage = im.data;
    else
        disp(['NO ' pid ' image file']);
        return;
    end  
    clear im;
    %imshow3D(LiverImage);
    
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

    %% get total slice number
    [~,~,SliceNum] = size(LiverImage);
    GlobalMAX = max(LiverImage, [], 'all');

    for slice = 1:SliceNum 
        %% get according slice mask
        LiverMask = mask(:,:,slice);
        if (sum(LiverMask(:)) >= AreaThreshold)
            %% get according slice image
            Image = LiverImage(:,:,slice);
            LocalMAX = max(Image, [], 'all');
            %% Get mask enhance image
            MaskedImage = Image;
            %% check  CR73_47
            MaskedImage = MaskedImage .* LiverMask;
            %% Save image and mask
            stage_str = num2str(stage);
            slice_str = num2str(slice);
            imwrite(mat2gray(Image,[0,GlobalMAX]), [StoreImagePath filesep stage_str filesep pid '_' slice_str '_image.jpg'],'jpg','Quality',100,'BitDepth',8);
            imwrite(mat2gray(MaskedImage,[0,GlobalMAX]),[StoreMaskedImagePath filesep stage_str filesep pid '_' slice_str '_maskedimage.jpg'],'jpg','Quality',100,'BitDepth',8);
            imwrite(LiverMask,[StoreLiverMaskPath filesep stage_str filesep pid '_' slice_str '_livermask.jpg'],'jpg','Quality',100,'BitDepth',8);
            clear stage_str;
        end
    end
end
            

