function [RFI_Avg, Slice, AREA,...
          GLRLM_SRE_Avg, GLRLM_LRE_Avg, GLRLM_GLN_Avg, GLRLM_RP_Avg,...
          GLRLM_RLN_Avg, GLRLM_LRLGLE_Avg, GLRLM_LRHGLE_Avg, GLRLM_SRLGLE_Avg, GLRLM_SRHGLE_Avg,...
          GLRLM_HGRE_Avg, GLRLM_LGRE_Avg,...
          GLCM_E_Avg, GLCM_SUME_Avg, GLCM_MAXP_Avg, GLCM_ASM_Avg,...
          GLCM_COR_Avg, GLCM_CON_Avg, GLCM_HOMO_Avg,...
          GLCM_AUTO_Avg, GLCM_CSHAD_Avg, GLCM_CPROM_Avg, GLCM_DIFE_Avg,...
          GLCM_DIFAV_Avg, GLCM_SUMAV_Avg, GLCM_DIFVAR_Avg, GLCM_SUMVAR_Avg,...
          GLCM_IMC1_Avg, GLCM_IMC2_Avg, GLCM_SOS_Avg] = Version_2(PID)
% tic;
%% Feature we want
COUNT = 0;
RFI_Avg = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
GLRLM_SRE_Avg = 0;
GLRLM_LRE_Avg = 0;
GLRLM_GLN_Avg = 0;
GLRLM_RP_Avg = 0;
GLRLM_RLN_Avg = 0;
GLRLM_LRLGLE_Avg = 0;
GLRLM_LRHGLE_Avg = 0; 
GLRLM_SRLGLE_Avg = 0;
GLRLM_SRHGLE_Avg = 0;
GLRLM_HGRE_Avg = 0;
GLRLM_LGRE_Avg = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
GLCM_E_Avg = 0;
GLCM_SUME_Avg = 0;
GLCM_MAXP_Avg = 0;
GLCM_ASM_Avg = 0;
GLCM_COR_Avg = 0;
GLCM_CON_Avg = 0;
GLCM_HOMO_Avg = 0;
GLCM_AUTO_Avg = 0;
GLCM_CSHAD_Avg = 0;
GLCM_CPROM_Avg = 0;
GLCM_DIFE_Avg = 0;
GLCM_DIFAV_Avg = 0;
GLCM_SUMAV_Avg = 0;
GLCM_DIFVAR_Avg = 0;
GLCM_SUMVAR_Avg = 0;
GLCM_IMC1_Avg = 0;
GLCM_IMC2_Avg = 0;
GLCM_SOS_Avg = 0;

%% Parameter and path
Radius = 7; 
Window_size = 225;
Bin_size = 64;
XlsxPath = ['..' filesep 'data xlsx' filesep];
ImagePath = ['..' filesep 'Image and Mask' filesep 'NILB' filesep];

%% get statitic file
% check if file exist
if isfile([XlsxPath PID '.xlsx'])
    Statistic = readtable([XlsxPath PID '.xlsx'], 'ReadRowNames',true, 'Sheet', 'S');
else
    disp(['file ' PID '.xlsx not exist']);
    return
end

%% get image file
if isfile([ImagePath PID filesep PID '.mhd'])
    im = read_mhd([ImagePath PID filesep PID '.mhd']);
    LiverImage = im.data;
else
    disp(['NO ' PID ' image file']);
    return
end
clear im;

%% load mask
% try Szesze mask
if isfile([ImagePath PID filesep PID '_Liver_SL.vtk'])
    info = vtk_read_header([ImagePath PID filesep PID '_Liver_SL.vtk']);
    temp = vtk_read_volume(info);
    mask = temp > 0.5;
% try  Ola mask
elseif isfile([ImagePath PID filesep PID '_Liver.vtk'])
    info = vtk_read_header([ImagePath PID filesep PID '_Liver.vtk']);
    info.BitDepth = 32;
    temp = vtk_read_volume(info);
    mask = (temp/255) > 0.5;
else 
    disp('NO Live mask file. Try to draw by ourselves');
end
clear temp info;

%% select interesting  slice
% MaskedImage = LiverImage;
% if (exist('mask', 'var'))
%     MaskedImage(mask) = MaskedImage(mask)*3;
%     [Slice] = imshow3D(MaskedImage);
%     LiverMask = mask(:,:,Slice);  % load mask as ROI / can change later
% else 
%     [Slice] = imshow3D(LiverImage);
% end
%% automaticlly select interesting  slice
[~,~,SliceNum] = size(LiverImage);
Slice = round(SliceNum/2);
LiverMask = mask(:,:,Slice);
Image = LiverImage(:,:,Slice);

% if mask not exist we need to create ROI by ourselves
if (~exist('mask', 'var'))
    %% show image and select ROI of Liver
    f = figure('NumberTitle', 'off', 'Name', 'Draw ROI');
    imshow(Image,[]);
    %liver = drawfreehand('LineWidth',2,'Color','red');
    mask = drawpolygon('LineWidth',2,'Color','red');
    %ROI = int32(mask.Position);
    % create ROI mask image
    LiverMask = mask.createMask();
    close(f);
end
clear mask LiverImage;

%% transform data into 0-1 range based on 'local' max/min
AREA = sum(LiverMask(:)); 
[row, col] = size(Image);
%MaskedImage = Image;   
%MaskedImage(LiverMask) = MaskedImage(LiverMask)*3;
%Image_max = max(MaskedImage(LiverMask));
%Image_min = min(MaskedImage(LiverMask)); 
% convert image into 0-1 range // Base on the Liver region
%Masked_01 = (MaskedImage - Image_min)/(Image_max - Image_min);

%% calculate mean/SD for nomralize // later 'mean' should from spleen & SD from liver
LiverMEAN = mean(Image(LiverMask));
LiverSD = std(Image(LiverMask));
Liver_normalize = (Image - LiverMEAN)/LiverSD;
%Liver_normalize_MAX = max(Liver_normalize(LiverMask));
%Liver_normalize_MIN = min(Liver_normalize(LiverMask));
% rescale the image into 0~1 range
%Liver_01 = (Liver_normalize - Liver_normalize_MIN)/(Liver_normalize_MAX - Liver_normalize_MIN); 

%% create mask image
image_mask = zeros([row, col]);
for i = 1:row
    for j = 1: col
        if (LiverMask(i,j))
            image_mask(i,j) = Liver_normalize(i,j);
        else
            image_mask(i,j) = 0;  
        end
    end
end

%% crop image 
% first crop to smaller reangle image 
%X_max = max(ROI(:,1));
%X_min = min(ROI(:,1));
%Y_max = max(ROI(:,2));
%Y_min = min(ROI(:,2));
%image_crop = imcrop(image_normalize, [X_min, Y_min, X_max-X_min, Y_max-Y_min]);
%image_crop_ROI = int32([ROI(:,1)-X_min+1, ROI(:,2)-Y_min+1]); % +1 since index is start from 0
%image_crop_Mask = mask_liver(Y_min:Y_max, X_min:X_max); % image are in row*col format and col=x row=y; Just transposed
%figure(4)
%imshow(image_crop);
%hold on;
%pgon = polyshape(image_crop_ROI(:,1),image_crop_ROI(:,2));
%plot(pgon);

%% Calculate NLE
try
    LiverSI_20min_normalize = Statistic.x20(1:7);
catch 
    LiverSI_20min_normalize = Statistic.x10(1:7);
end
%SpleenSI_20min_normalize = Statistic.x20(8:10);
LiverSI_20min_normalize = mean(LiverSI_20min_normalize);
%SpleenSI_20min_normalize = mean(SpleenSI_20min_normalize);
NLE = LiverSI_20min_normalize - 1;
% % FOR LP39
% NLE = 0.735288105;

%% GLCM for each window  / window size = 15
disp("Start computing...");

RFI = zeros([row,col]);
for i = 1:row
    for j = 1:col
        if (LiverMask(i,j))
            up = max(i - Radius, 1);
            down = min(i + Radius, row);
            left = max(j - Radius, 1);
            right = min(j + Radius, col);
            crop_Mask = LiverMask(up:down, left:right);
            % whether the window contain all the pixel in the ROI / later might change to other boundry handle method
            if (sum(crop_Mask(:)) == Window_size)
                image_crop = Liver_normalize(up:down, left:right);
                [SRE,LRE,GLN,RP,RLN,LRLGLE,LRHGLE,HGRE,LGRE,SRHGLE,SRLGLE] = GLRLM(image_crop, Bin_size);
                [E, SUME, MAXP, ASM, COR, CON, HOMO, AUTO, CSHAD,...
                 CPROM, DIFE, DIFAV, SUMAV, DIFVAR, SUMVAR, IMC1, IMC2, SOS] = GLCM(image_crop, Bin_size);
                a = -4.3-(0.24*NLE)-(0.93*LRLGLE)-(3.15*MAXP)+(1.21*SUME);
                tmp = exp(a);
                RFI(i,j) = tmp/(1+tmp);
                %% accumulate feature
                COUNT  = COUNT + 1; 
                GLRLM_SRE_Avg = GLRLM_SRE_Avg + SRE; 
                GLRLM_LRE_Avg = GLRLM_LRE_Avg + LRE;
                GLRLM_GLN_Avg = GLRLM_GLN_Avg + GLN;
                GLRLM_RP_Avg = GLRLM_RP_Avg + RP;
                GLRLM_RLN_Avg = GLRLM_RLN_Avg + RLN;
                GLRLM_LRLGLE_Avg = GLRLM_LRLGLE_Avg + LRLGLE;
                GLRLM_LRHGLE_Avg = GLRLM_LRHGLE_Avg + LRHGLE;
                GLRLM_SRLGLE_Avg = GLRLM_SRLGLE_Avg + SRLGLE;
                GLRLM_SRHGLE_Avg = GLRLM_SRHGLE_Avg + SRHGLE;
                GLRLM_HGRE_Avg = GLRLM_HGRE_Avg + HGRE;
                GLRLM_LGRE_Avg = GLRLM_LGRE_Avg + LGRE;
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                GLCM_E_Avg = GLCM_E_Avg + E;
                GLCM_SUME_Avg = GLCM_SUME_Avg + SUME;
                GLCM_MAXP_Avg = GLCM_MAXP_Avg + MAXP;
                GLCM_ASM_Avg = GLCM_ASM_Avg + ASM;
                GLCM_COR_Avg = GLCM_COR_Avg + COR;
                GLCM_CON_Avg = GLCM_CON_Avg + CON;
                GLCM_HOMO_Avg = GLCM_HOMO_Avg + HOMO;
                GLCM_AUTO_Avg = GLCM_AUTO_Avg + AUTO;
                GLCM_CSHAD_Avg = GLCM_CSHAD_Avg + CSHAD;
                GLCM_CPROM_Avg = GLCM_CPROM_Avg + CPROM;
                GLCM_DIFE_Avg = GLCM_DIFE_Avg + DIFE;
                GLCM_DIFAV_Avg = GLCM_DIFAV_Avg + DIFAV;
                GLCM_SUMAV_Avg = GLCM_SUMAV_Avg + SUMAV;
                GLCM_DIFVAR_Avg = GLCM_DIFVAR_Avg + DIFVAR;
                GLCM_SUMVAR_Avg = GLCM_SUMVAR_Avg + SUMVAR;
                GLCM_IMC1_Avg = GLCM_IMC1_Avg + IMC1;
                GLCM_IMC2_Avg = GLCM_IMC2_Avg + IMC2;
                GLCM_SOS_Avg = GLCM_SOS_Avg + SOS;
            end
        end
    end
end

%% show result image
% figure('NumberTitle', 'off', 'Name', 'Result') 
% ax1 = axes;
% imagesc(Liver_01);
% colormap(ax1,'gray');
% ax2 = axes;
% imagesc(ax2, RFI, 'alphadata', RFI>0);
% colormap(ax2, jet(256));
% caxis(ax2, [0,1]);
% ax2.Visible = 'off';
% linkprop([ax1 ax2],'Position');
% colorbar;  
% set(gcf,'position',[100,100,650,500]);   

%% Average feature
RFI_Avg = sum(RFI(:)) / COUNT;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
GLRLM_SRE_Avg = GLRLM_SRE_Avg / COUNT;
GLRLM_LRE_Avg = GLRLM_LRE_Avg / COUNT;
GLRLM_GLN_Avg = GLRLM_GLN_Avg / COUNT;
GLRLM_RP_Avg = GLRLM_RP_Avg / COUNT;
GLRLM_RLN_Avg = GLRLM_RLN_Avg / COUNT;
GLRLM_LRLGLE_Avg = GLRLM_LRLGLE_Avg / COUNT;
GLRLM_LRHGLE_Avg = GLRLM_LRHGLE_Avg / COUNT; 
GLRLM_SRLGLE_Avg = GLRLM_SRLGLE_Avg / COUNT;
GLRLM_SRHGLE_Avg = GLRLM_SRHGLE_Avg / COUNT; 
GLRLM_HGRE_Avg = GLRLM_HGRE_Avg / COUNT;
GLRLM_LGRE_Avg = GLRLM_LGRE_Avg / COUNT;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
GLCM_E_Avg = GLCM_E_Avg / COUNT;
GLCM_SUME_Avg = GLCM_SUME_Avg / COUNT;
GLCM_MAXP_Avg = GLCM_MAXP_Avg / COUNT;
GLCM_ASM_Avg = GLCM_ASM_Avg / COUNT;
GLCM_COR_Avg = GLCM_COR_Avg / COUNT;
GLCM_CON_Avg = GLCM_CON_Avg / COUNT;
GLCM_HOMO_Avg = GLCM_HOMO_Avg / COUNT;
GLCM_AUTO_Avg = GLCM_AUTO_Avg / COUNT;
GLCM_CSHAD_Avg = GLCM_CSHAD_Avg / COUNT;
GLCM_CPROM_Avg = GLCM_CPROM_Avg / COUNT;
GLCM_DIFE_Avg = GLCM_DIFE_Avg / COUNT;
GLCM_DIFAV_Avg = GLCM_DIFAV_Avg / COUNT;
GLCM_SUMAV_Avg = GLCM_SUMAV_Avg / COUNT;
GLCM_DIFVAR_Avg = GLCM_DIFVAR_Avg / COUNT;
GLCM_SUMVAR_Avg = GLCM_SUMVAR_Avg / COUNT;
GLCM_IMC1_Avg = GLCM_IMC1_Avg / COUNT;
GLCM_IMC2_Avg = GLCM_IMC2_Avg / COUNT;
GLCM_SOS_Avg = GLCM_SOS_Avg / COUNT;
end

%% NER VERSION RFI AVG
% AD67 0 0.3623
% AE50 4 0.3290
