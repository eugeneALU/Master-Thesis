clear;
close all;
warning('off','all');
addpath(genpath('MedicalImageProcessingToolbox'));
addpath(genpath('ReadData3D'));

XlsxPath = ['..' filesep 'data xlsx' filesep];
ImagePath = ['..' filesep 'Image and Mask' filesep 'NILB' filesep];
% output path
StoreImagePath = ['..' filesep 'Image' filesep];
StoreMaskedImagePath = ['..' filesep 'MaskedImage' filesep];
StoreLiverMaskPath = ['..' filesep 'LiverMask' filesep];
DataPath = ['..' filesep 'Data' filesep];
AreaThreshold = 2500;   %50*50
%% Read in Label data 
load('fibrosis.mat');   % load in Fibrosis struct
%Fibrosis = (Fibrosis(1:2));   %only for test
SIZE = max(size(Fibrosis));

%% Feature Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
STAGE = 1;
SLICE = 2;
RFI_AVG = 3;
AREA = 4;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
GLRLM_SRE = 5;
GLRLM_LRE = 6;
GLRLM_GLN = 7;
GLRLM_RP = 8;
GLRLM_RLN = 9;
GLRLM_LRLGLE = 10;
GLRLM_LRHGLE = 11;
GLRLM_SRLGLE = 12;
GLRLM_SRHGLE = 13;
GLRLM_HGRE = 14;
GLRLM_LGRE = 15;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
GLCM_E = 16;
GLCM_SUME = 17;
GLCM_MAXP = 18;
GLCM_ASM = 19;
GLCM_COR = 20;
GLCM_CON = 21;
GLCM_HOMO = 22;
GLCM_AUTO = 23;
GLCM_CSHAD = 24;
GLCM_CPROM = 25;
GLCM_DIFE = 26;
GLCM_DIFAV = 27;
GLCM_SUMAV = 28;
GLCM_DIFVAR = 29;
GLCM_SUMVAR = 30;
GLCM_IMC1 = 31;
GLCM_IMC2 = 32;
GLCM_SOS = 33;

for i = 10:SIZE
    pid = Fibrosis(i).PID;
    stage = Fibrosis(i).fibrosis;
    
    %% get statistic file
    % check if file exist
    if isfile([XlsxPath pid '.xlsx'])
        Statistic = readtable([XlsxPath pid '.xlsx'], 'ReadRowNames',true, 'Sheet', 'S');
    else
        disp(['file ' pid '.xlsx not exist']);
        return;
    end
    
    %% Calculate NLE (pass to function later)
    try
        LiverSI_20min_normalize = Statistic.x20(1:7);
    catch 
        LiverSI_20min_normalize = Statistic.x10(1:7);
    end
    %SpleenSI_20min_normalize = Statistic.x20(8:10);
    LiverSI_20min_normalize = mean(LiverSI_20min_normalize);
    %SpleenSI_20min_normalize = mean(SpleenSI_20min_normalize);
    NLE = LiverSI_20min_normalize - 1;
    clear Statistic LiverSI_20min_normalize;
    
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
    
    %% get total slice number
    [~,~,SliceNum] = size(LiverImage);
    
    %% Table for Append Result
    PID = strings(SliceNum,1);
    DATA = zeros(SliceNum, 33);
    
    %% Loop through each slice who's Liver area is big enough
    COUNT = 0;
    for slice = 1:SliceNum   
       try
       %% get according slice mask
       LiverMask = mask(:,:,slice);
       if (sum(LiverMask(:)) >= AreaThreshold)
            COUNT = COUNT + 1;
            %% get according slice image
            Image = LiverImage(:,:,slice);
            %% Get mask enhance image
            MaskedImage = Image;
            MaskedImage(~LiverMask) = 0;
            %% Save image and mask
            stage_str = num2str(stage);
            slice_str = num2str(slice);
            imwrite(mat2gray(Image), [StoreImagePath filesep stage_str filesep pid '_' slice_str '_image.jpg']);
            imwrite(mat2gray(MaskedImage),[StoreMaskedImagePath filesep stage_str filesep pid '_' slice_str '_maskedimage.jpg']);
            imwrite(LiverMask,[StoreLiverMaskPath filesep stage_str filesep pid '_' slice_str '_livermask.jpg']);
            clear stage_str;
            
            %% call function
            disp(['Start computing...' pid ' Slice:' slice_str]);
            clear slice_str;
            
            [rfi,...
             GLRLM_sre, GLRLM_lre, GLRLM_gln, GLRLM_rp,...
             GLRLM_rln, GLRLM_lrlgle, GLRLM_lrhgle, GLRLM_srlgle, GLRLM_srhgle,...
             GLRLM_hgre, GLRLM_lgre,...
             GLCM_e, GLCM_sume, GLCM_maxp, GLCM_asm,...
             GLCM_cor, GLCM_con, GLCM_homo,...
             GLCM_auto, GLCM_cshad, GLCM_cprom, GLCM_dife,...
             GLCM_difav, GLCM_sumav, GLCM_difvar, GLCM_sumvar,...
             GLCM_imc1, GLCM_imc2, GLCM_sos] = Version_3(Image, LiverMask, NLE);

            %% Append Result
           	PID(COUNT) = pid;
            DATA(COUNT,STAGE) = stage;
            DATA(COUNT,SLICE) = slice; 
            DATA(COUNT,RFI_AVG) = rfi; 
            DATA(COUNT,AREA) = sum(LiverMask(:));
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            DATA(COUNT,GLRLM_SRE) = GLRLM_sre;
            DATA(COUNT,GLRLM_LRE) = GLRLM_lre;
            DATA(COUNT,GLRLM_GLN) = GLRLM_gln;
            DATA(COUNT,GLRLM_RP) = GLRLM_rp;
            DATA(COUNT,GLRLM_RLN) = GLRLM_rln;
            DATA(COUNT,GLRLM_LRLGLE) = GLRLM_lrlgle;
            DATA(COUNT,GLRLM_LRHGLE) = GLRLM_lrhgle;
            DATA(COUNT,GLRLM_SRLGLE) = GLRLM_srlgle;
            DATA(COUNT,GLRLM_SRHGLE) = GLRLM_srhgle;
            DATA(COUNT,GLRLM_HGRE) = GLRLM_hgre;
            DATA(COUNT,GLRLM_LGRE) = GLRLM_lgre;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            DATA(COUNT,GLCM_E) = GLCM_e;
            DATA(COUNT,GLCM_SUME) = GLCM_sume;
            DATA(COUNT,GLCM_MAXP) = GLCM_maxp;
            DATA(COUNT,GLCM_ASM) = GLCM_asm;
            DATA(COUNT,GLCM_COR) = GLCM_cor;
            DATA(COUNT,GLCM_CON) = GLCM_con;
            DATA(COUNT,GLCM_HOMO) = GLCM_homo;
            DATA(COUNT,GLCM_AUTO) = GLCM_auto;
            DATA(COUNT,GLCM_CSHAD) = GLCM_cshad;
            DATA(COUNT,GLCM_CPROM) = GLCM_cprom;
            DATA(COUNT,GLCM_DIFE) = GLCM_dife;
            DATA(COUNT,GLCM_DIFAV) = GLCM_difav;
            DATA(COUNT,GLCM_SUMAV) = GLCM_sumav;
            DATA(COUNT,GLCM_DIFVAR) = GLCM_difvar;
            DATA(COUNT,GLCM_SUMVAR) = GLCM_sumvar;
            DATA(COUNT,GLCM_IMC1) = GLCM_imc1;
            DATA(COUNT,GLCM_IMC2) = GLCM_imc2;
            DATA(COUNT,GLCM_SOS) = GLCM_sos;
       end
       catch ME
            errorMessage = sprintf('Error in function %s() at line %d.\nIteration: %d\nError Message:\n%s', ...
                            ME.stack(1).name, ME.stack(1).line, i, ME.message);
            fprintf(1, '%s\n', errorMessage); % To command window.

            fullFileName = 'Error Log.txt';
            fid = fopen(fullFileName, 'at');
            fprintf(fid, '%s\n', errorMessage); % To file
            fclose(fid); 
            continue
        end
    end
    % take only 1 ~ COUNT;
    PID = PID(1:COUNT);
    DATA = DATA(1:COUNT,:);
    % Create Table
    RESULT = table(PID, DATA);
    % Split the DATA column for adding the column name
    RESULT = splitvars(RESULT);
    % Set column name
    RESULT.Properties.VariableNames = {'PID','STAGE','SLICE','RFI_AVG','AREA',...
                                       'GLRLM_SRE', 'GLRLM_LRE', 'GLRLM_GLN', 'GLRLM_RP',...
                                       'GLRLM_RLN', 'GLRLM_LRLGLE', 'GLRLM_LRHGLE', 'GLRLM_SRLGLE', 'GLRLM_SRHGLE',...
                                       'GLRLM_HGRE', 'GLRLM_LGRE',...
                                       'GLCM_E', 'GLCM_SUME', 'GLCM_MAXP', 'GLCM_ASM',...
                                       'GLCM_COR', 'GLCM_CON', 'GLCM_HOMO',...
                                       'GLCM_AUTO', 'GLCM_CSHAD', 'GLCM_CPROM', 'GLCM_DIFE',...
                                       'GLCM_DIFAV', 'GLCM_SUMAV', 'GLCM_DIFVAR', 'GLCM_SUMVAR',...
                                       'GLCM_IMC1', 'GLCM_IMC2', 'GLCM_SOS'};
    writetable(RESULT,[DataPath pid '_Features.xlsx']);
    clear RESULT DATA PID mask;
end   