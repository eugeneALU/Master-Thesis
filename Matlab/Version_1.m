clear;
close all;
%% Parameter and path
Radius = 7; 
Window_size = (Radius*2 + 1)^2;
Bin_size = 64;
addpath(genpath('readData'));

%% select interesting file
[file,path,indx] = uigetfile({'*.PAR'}, 'Select file');
if isequal(file,0)
   disp('User selected Cancel');
   return;
else
   disp(['User selected ', fullfile(path, file)]);
end

file = fullfile(path, file);
clear path
data = readparrec(file);

%% select interesting image
[Slice] = imshow3D(data.image);
image = data.image(:,:,Slice);
[row, col] = size(image);
img_max = max(image(:));
img_min = min(image(:));
%image = (image - img_min)/(img_max - img_min); % convert image into 0-1 range

%% show image and select ROI of Liver
figure('NumberTitle', 'off', 'Name', 'Draw ROI');
imshow(image, []);
%liver = drawfreehand('LineWidth',2,'Color','red');
liver = drawpolygon('LineWidth',2,'Color','red');
ROI = int32(liver.Position);
% create ROI mask image
mask_liver = liver.createMask();

%% calculate mean/SD for nomralize // later 'mean' should from spleen & SD from liver
MEAN = mean(image(mask_liver));
SD = std(image(mask_liver));
image_normalize = (image - MEAN)/SD;
max_normalize = max(image_normalize(:));
min_normalize = min(image_normalize(:));
% rescale the image into 0~1 range
image_normalize = (image_normalize - min_normalize)/(max_normalize - min_normalize); 
%image_normalize = mat2gray(image_normalize);

%% create mask image
image_mask = zeros([row, col]);
for i = 1:row
    for j = 1: col
        if (mask_liver(i,j))
            image_mask(i,j) = image_normalize(i,j);
        else
            image_mask(i,j) = 0;  
        end
    end
end

%figure(2)
%imshow(image_mask, []);
%figure(3)
%imshow(image_normalize, []);

%% crop image into several pieces (size = 15)
% first crop to smaller reangle image 
X_max = max(ROI(:,1));
X_min = min(ROI(:,1));
Y_max = max(ROI(:,2));
Y_min = min(ROI(:,2));
%image_crop = imcrop(image_normalize, [X_min, Y_min, X_max-X_min, Y_max-Y_min]);
%image_crop_ROI = int32([ROI(:,1)-X_min+1, ROI(:,2)-Y_min+1]); % +1 since index is start from 0
%image_crop_Mask = mask_liver(Y_min:Y_max, X_min:X_max); % image are in row*col format and col=x row=y; Just transposed
%figure(4)
%imshow(image_crop);
%hold on;
%pgon = polyshape(image_crop_ROI(:,1),image_crop_ROI(:,2));
%plot(pgon);

%% CHECK NLE!!!!!
NLE = 1;

%% GLCM for each window  / window size = 15
disp("Start computing...");
%image_RGB = repmat(image_normalize, [1 1 3]); % convert to 3 channel image
image_RGB = zeros([row, col]);
%examine_window = [];
RFI = zeros([row,col]);
for i = Y_min:Y_max
    for j = X_min:X_max
        if (mask_liver(i,j))
            up = max(i - Radius, 1);
            down = min(i + Radius, row);
            left = max(j - Radius, 1);
            right = min(j + Radius, col);
            crop_Mask = mask_liver(up:down, left:right);
            if (sum(crop_Mask(:)) == Window_size) % whether the window contain all the pixel in the ROI / later might change to other boundry handle method
               %fprintf('Processing:[%d, %d]\n', i, j);
               %examine_window = [examine_window; [i,j]];
               image_crop = image_normalize(up:down, left:right);
               [SRE,LRE,GLN,RP,RLN,LRLGLE,LRHGLE] = GLRLM(image_crop, Bin_size);
               [E, SUME, MAXP] = GLCM(image_crop, Bin_size);
               a = -4.3-0.24*NLE-0.93*LRLGLE-3.15*MAXP+1.21*SUME;
               tmp = exp(a);
               RFI = tmp/(1+tmp);
               image_RGB(i,j) = RFI; 
            end
        end
    end
end
figure('NumberTitle', 'off', 'Name', 'Result') 
ax1 = axes;
imagesc(ax1, image_normalize);
colormap(ax1,'gray');
hold on;
pgon = polyshape(ROI(:,1),ROI(:,2));
plot(ax1, pgon);
ax2 = axes;
imagesc(ax2, image_RGB, 'alphadata', image_RGB>0);
colormap(ax2, jet(256));
caxis(ax2, [0 1]);
colorbar;
%set(gcf,'position',[100,100,1000,500]);   

         
