% Script for ploting data exported in differet file format 
close all
clear all
addpath /Volumes/andti/Matlab/Library/readData/readSinLab/;
addpath /Volumes/andti/Matlab/Library/readData/readSpar/;
addpath /Volumes/andti/Matlab/Library/plotData/;

% % In vivo thalamus 
% rawFile = '/Volumes/data_radiofys/Data/MR/Anders/140806-DBS-dicom/transfer/1446_PRESS_THA_sin.raw';
% sparFileWater = '/Volumes/data_radiofys/Data/MR/Anders/140806-DBS-dicom/140805-Parkinson06-TF53-Achieva/parkinson06-TF53-MRS-th-sin_6_2_raw_ref.SPAR';
% sparFileMetabolite = '/Volumes/data_radiofys/Data/MR/Anders/140806-DBS-dicom/140805-Parkinson06-TF53-Achieva/parkinson06-TF53-MRS-th-sin_6_2_raw_act.SPAR';

% % SpherA 
% rawFile = '/Volumes/data_radiofys/Data/MR/Anders/140811-SphereA-MRSorientation-Achieva/transfer/1632_SV_PRESS_35_cor.raw';
% sparFileWater = '/Volumes/data_radiofys/Data/MR/Anders/140811-SphereA-MRSorientation-Achieva/MRS_spar/SphereA_SV_PRESS_35_cor_9_2_raw_ref.SPAR';
% sparFileMetabolite = '/Volumes/data_radiofys/Data/MR/Anders/140811-SphereA-MRSorientation-Achieva/MRS_spar/SphereA_SV_PRESS_35_cor_9_2_raw_act.SPAR';

wd = pwd;
cd '/Volumes/data_radiofys/Data/MR/Anders/';
[rawFileName, rawDir] = uigetfile('*.raw','Select raw file ');
rawFile = fullfile(rawDir,rawFileName);
cd(rawDir)
cd ..
[metaboliteSparFileName, sparDir] = uigetfile('*.spar',['Select act file  to compar with ' rawFileName]);
sparFileMetabolite = fullfile(sparDir,metaboliteSparFileName);
sparFileWater = strrep(sparFileMetabolite,'raw_act','raw_ref');
cd(wd);
labraw = readLABRAWspectro( rawFile );
sparWater = readsparsdat( sparFileWater );
sparMetabolite = readsparsdat( sparFileMetabolite );

NrawFID = length(labraw.ReferenceFID{1});
NsparFID = length(sparWater.data{1});
NFID = length(labraw.ReferenceFID);

[TimeRaw, PPMRaw] = getTimeAndPPM(NrawFID, labraw.SampleFrequency, sparWater.Synthesizer_frequency);
[TimeSpar, PPMSpar] = getTimeAndPPM(NsparFID, sparWater.Sample_frequency, sparWater.Synthesizer_frequency);

sparFIDwater = zeros(NFID,NsparFID);
sparFIDmetabolite = zeros(NFID,NsparFID);

rawFIDwater =  zeros(NFID,NrawFID);
rawFIDmetabolite =  zeros(NFID,NrawFID);
rawFIDwaterECCdownSampled  = zeros(NFID,NsparFID);
rawFIDmetaboliteECCdownSampled  = zeros(NFID,NsparFID);

for FrameIDX = 1:NFID
    
    % Calculate complex FIDs of the spar data
    sparFIDwater(FrameIDX,:) = sparWater.data{FrameIDX}(1,:) + 1i .* sparWater.data{FrameIDX}(2,:);
    sparFIDmetabolite(FrameIDX,:) = sparMetabolite.data{FrameIDX}(1,:) + 1i .* sparMetabolite.data{FrameIDX}(2,:);
    
    % Read complex FID of the raw data
    rawFIDwater(FrameIDX,:) = labraw.ReferenceFID{FrameIDX};
    rawFIDmetabolite(FrameIDX,:) = labraw.MetaboliteFID{FrameIDX};  
       
end

% FILTER SPAR
% Scale first point with 

sparFIDwater2 = sparFIDwater;
sparFIDmetabolite2 = sparFIDmetabolite;
sparFIDwater2(:,1) = 2 .* sparFIDwater(:,1);
sparFIDmetabolite2(:,1) = 2 .* sparFIDmetabolite(:,1);

% Do ECC on the water spar data
sparFIDwaterECC = sparFIDwater2 .* exp(-1i .* angle(sparFIDwater2)); % Klose corrected unsupressed FID 
sparFIDmetaboliteECC = sparFIDmetabolite2 .* exp(-1i .* angle(sparFIDwater2));

% FILTER RAW
% Do ECC on the raw data 
rawFIDwaterECC = rawFIDwater .* exp( -1i .* angle(rawFIDwater));
rawFIDmetaboliteECC = rawFIDmetabolite .* exp( -1i .* angle(rawFIDwater));
     
scaleRaw2Spar = mean(abs(sparFIDwaterECC(1,:))) / mean(abs(rawFIDwaterECC(1,:)));
scaleRaw2SparMet = mean(abs(sparFIDmetaboliteECC(1,:))) / mean(abs(rawFIDmetaboliteECC(1,:)));
rawFIDwaterScaled = scaleRaw2Spar * rawFIDwaterECC;
rawFIDmetaboliteScaled = scaleRaw2Spar * rawFIDmetaboliteECC;

% Down sample the 32 kHz data to spar sample freq. 
for FrameIDX = 1:NFID
 
   rawFIDwaterECCdownSampled(FrameIDX,:) = mean(reshape(rawFIDwaterScaled(FrameIDX,:), labraw.SampleFrequency/sparWater.Sample_frequency,[]));  
   rawFIDmetaboliteECCdownSampled(FrameIDX,:) = mean(reshape(rawFIDmetaboliteScaled(FrameIDX,:), labraw.SampleFrequency/sparWater.Sample_frequency,[]));

end

plotCompare(rawFIDwaterScaled,rawFIDmetaboliteScaled,TimeRaw,PPMRaw,sparFIDwaterECC,sparFIDmetabolite2,TimeSpar,PPMSpar,'Raw signal scaled');
plotLog(rawFIDwaterScaled,TimeRaw,sparFIDwaterECC,TimeSpar,'Log plot of FIDs')

plot2by2(rawFIDwaterScaled,rawFIDmetaboliteScaled,TimeRaw,PPMRaw,sparFIDwaterECC,sparFIDmetabolite2,TimeSpar,PPMSpar,'Raw signal scaled');

