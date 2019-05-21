clear;
close all;
warning('off','all');
addpath(genpath('../MedicalImageProcessingToolbox'));
addpath(genpath('../ReadData3D'));
%% Read in Label data 
load('fibrosis.mat');   % load in Fibrosis struct

SIZE = max(size(Fibrosis));
PID = {Fibrosis.PID}';
STAGE = [Fibrosis.fibrosis]';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SliceNum = zeros(SIZE,1);
RFI_Avg = zeros(SIZE,1);
AREA = zeros(SIZE,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
GLRLM_SRE = zeros(SIZE,1);
GLRLM_LRE = zeros(SIZE,1);
GLRLM_GLN = zeros(SIZE,1);
GLRLM_RP = zeros(SIZE,1);
GLRLM_RLN = zeros(SIZE,1);
GLRLM_LRLGLE = zeros(SIZE,1);
GLRLM_LRHGLE = zeros(SIZE,1);
GLRLM_SRLGLE = zeros(SIZE,1);
GLRLM_SRHGLE = zeros(SIZE,1);
GLRLM_HGRE = zeros(SIZE,1);
GLRLM_LGRE = zeros(SIZE,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
GLCM_E = zeros(SIZE,1);
GLCM_SUME = zeros(SIZE,1);
GLCM_MAXP = zeros(SIZE,1);
GLCM_ASM = zeros(SIZE,1);
GLCM_COR = zeros(SIZE,1);
GLCM_CON = zeros(SIZE,1);
GLCM_HOMO = zeros(SIZE,1);
GLCM_AUTO = zeros(SIZE,1);
GLCM_CSHAD = zeros(SIZE,1);
GLCM_CPROM = zeros(SIZE,1);
GLCM_DIFE = zeros(SIZE,1);
GLCM_DIFAV = zeros(SIZE,1);
GLCM_SUMAV = zeros(SIZE,1);
GLCM_DIFVAR = zeros(SIZE,1);
GLCM_SUMVAR = zeros(SIZE,1);
GLCM_IMC1 = zeros(SIZE,1);
GLCM_IMC2 = zeros(SIZE,1);
GLCM_SOS = zeros(SIZE,1);


for i = 1:SIZE
    try
    pid = Fibrosis(i).PID;
    stage = Fibrosis(i).fibrosis;
    display (['Now Processing: ', pid]);
    %% call function
    [rfi, slice, area,...
     GLRLM_sre, GLRLM_lre, GLRLM_gln, GLRLM_rp,...
     GLRLM_rln, GLRLM_lrlgle, GLRLM_lrhgle, GLRLM_srlgle, GLRLM_srhgle,...
     GLRLM_hgre, GLRLM_lgre,...
     GLCM_e, GLCM_sume, GLCM_maxp, GLCM_asm,...
     GLCM_cor, GLCM_con, GLCM_homo,...
     GLCM_auto, GLCM_cshad, GLCM_cprom, GLCM_dife,...
     GLCM_difav, GLCM_sumav, GLCM_difvar, GLCM_sumvar,...
     GLCM_imc1, GLCM_imc2, GLCM_sos] = Version_2(pid);
    
    %% Append Result
    RFI_Avg(i) = rfi;
    SliceNum(i) = slice; 
    AREA(i) = area;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    GLRLM_SRE(i) = GLRLM_sre;
    GLRLM_LRE(i) = GLRLM_lre;
    GLRLM_GLN(i) = GLRLM_gln;
    GLRLM_RP(i) = GLRLM_rp;
    GLRLM_RLN(i) = GLRLM_rln;
    GLRLM_LRLGLE(i) = GLRLM_lrlgle;
    GLRLM_LRHGLE(i) = GLRLM_lrhgle;
    GLRLM_SRLGLE(i) = GLRLM_srlgle;
    GLRLM_SRHGLE(i) = GLRLM_srhgle;
    GLRLM_HGRE(i) = GLRLM_hgre;
    GLRLM_LGRE(i) = GLRLM_lgre;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    GLCM_E(i) = GLCM_e;
    GLCM_SUME(i) = GLCM_sume;
    GLCM_MAXP(i) = GLCM_maxp;
    GLCM_ASM(i) = GLCM_asm;
    GLCM_COR(i) = GLCM_cor;
    GLCM_CON(i) = GLCM_con;
    GLCM_HOMO(i) = GLCM_homo;
    GLCM_AUTO(i) = GLCM_auto;
    GLCM_CSHAD(i) = GLCM_cshad;
    GLCM_CPROM(i) = GLCM_cprom;
    GLCM_DIFE(i) = GLCM_dife;
    GLCM_DIFAV(i) = GLCM_difav;
    GLCM_SUMAV(i) = GLCM_sumav;
    GLCM_DIFVAR(i) = GLCM_difvar;
    GLCM_SUMVAR(i) = GLCM_sumvar;
    GLCM_IMC1(i) = GLCM_imc1;
    GLCM_IMC2(i) = GLCM_imc2;
    GLCM_SOS(i) = GLCM_sos;
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
% 34 element
RESULT = table(PID, STAGE, SliceNum, RFI_Avg, AREA,...
               GLRLM_SRE, GLRLM_LRE, GLRLM_GLN, GLRLM_RP,...
               GLRLM_RLN, GLRLM_LRLGLE, GLRLM_LRHGLE, GLRLM_SRLGLE, GLRLM_SRHGLE,...
               GLRLM_HGRE, GLRLM_LGRE,...
               GLCM_E, GLCM_SUME, GLCM_MAXP, GLCM_ASM,...
               GLCM_COR, GLCM_CON, GLCM_HOMO,...
               GLCM_AUTO, GLCM_CSHAD, GLCM_CPROM, GLCM_DIFE,...
               GLCM_DIFAV, GLCM_SUMAV, GLCM_DIFVAR, GLCM_SUMVAR,...
               GLCM_IMC1, GLCM_IMC2, GLCM_SOS);
writetable(RESULT,'RESULT.xlsx');

% print average RFI scatter image of every fibrosis stage
%scatter(STAGE, RFI_Avg);

%% read/write table
% readtable('RESULT2.xlsx','ReadRowNames',true );
% writetable(RESULT,'RESULT2.xlsx','WriteRowNames',true)
%% select table 
% RESULT({'RowNAME'}, {'ColNAME'})
% RESULT(RowINDEX, ColINDEX)
% RESULT.ColNAME
%% create table
% RESULT = table(STAGE, SliceNum, RFI_Avg, 'RowNames', PID); // with rowname
%% concat two table 
% RESULT = [RESULT; T0]  // append to RESULT
