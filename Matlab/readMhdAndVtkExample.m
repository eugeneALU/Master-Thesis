%% example file of how to use read_mhd and vtk_read_header
clear
close all

%% Loading stuff

PID = 'AD67';
folder = PID;

% Load image
im = read_mhd([folder filesep PID '.mhd']);
liverImage = im.data;

% Load Szesze mask
info = vtk_read_header([folder filesep PID '_Liver_SL.vtk']);
temp = vtk_read_volume(info);
maskSL = temp > 0.5;

% Load Ola mask
info = vtk_read_header([folder filesep PID '_Liver.vtk']);
info.BitDepth = 32;
temp = vtk_read_volume(info);
maskOla = (temp/255) > 0.5;


%% Inspect Szesze Mask
maskedImage = liverImage;
maskedImage(maskSL) = maskedImage(maskSL)*3;
imshow3D(maskedImage);


%% Inspect Ola Mask
maskedImage = liverImage;
maskedImage(maskOla) = maskedImage(maskOla)*3;
imshow3D(maskedImage);


%% Calc and show diference image
diffMask = zeros(size(maskOla));
diffMask(maskSL) = 1;
diffMask(maskOla) = 2;
diffMask(maskOla & maskSL) = 3;
% imshow3D(diffMask,[0 3],'jet');


nSlices = size(liverImage,3);
final = zeros(size(liverImage,1), size(liverImage,2), 3, size(liverImage,3));
range = [1 3];

for slice = 1:nSlices

    imSlice = liverImage(:,:,slice);
    diffMaskSlice = diffMask(:,:,slice);
    maskLiver = diffMaskSlice>0;
    
    grayMap = mat2gray(diffMaskSlice,range)*255;
    map = jet(255);
    rgbMap = ind2rgb(uint16(grayMap),map);
    merged = imfuse(imSlice,rgbMap,'blend');
    
    %im(~maskLiver) = 1;
    grayImage = mat2gray(imSlice,[0 max(imSlice(:))]);
    grayImage = grayImage(:,:,[1 1 1]);
    
    finalSlice = [];
    for i = 1:3
        temp1 = grayImage(:,:,i);
        temp2 = double(merged(:,:,i))./255;
        temp1(maskLiver) = temp2(maskLiver);
        finalSlice(:,:,i) = fliplr(rot90(temp1,-1));
    end
    
    final(:,:,:,slice) = finalSlice; 

end

final = squeeze(final);
implay(final)