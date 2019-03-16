function [dicom, info] = readDicomMRS_Philips(varargin)
%READDICOMMRS_PHILIPS Reads MRS files in dicom format
%   [ dicom, info ] = readDicomMRS_Philips
%   [ dicom, info ] = readDicomMRS_Philips( dicomFileName )
%
% Writen 20140813 Anders Tisell

%
% ------------Initialization----------------
% Select the dicom file


if nargin == 0
    wd = pwd;
    
    cd('/Volumes/andti/Temporary/DICOM/')
    [FileName,PathName] = uigetfile('*','Choose a MRS file');
    cd(wd);
    if ~isempty(FileName)
    Dicomfile = fullfile(PathName,FileName);
    else
        error('Shit for brains behind the keybord!')
    end

elseif nargin == 1 % A cell aray with filenames including path
    Dicomfile = varargin{1};
else
    close(waitbar_h)
    error('Could not read dicom file, wrong number of arguments')
end

waitbar_h = waitbar(0,'Read dicomfile');

info = dicominfo(Dicomfile);

% Examination info

dicom.FileName = info.Filename;
dicom.SeriesDescription = info.SeriesDescription;
dicom.AccessionNumber = info.AccessionNumber;
dicom.StudyDescription = info.StudyDescription;
dicom.StudyInstanceUID = info.StudyInstanceUID;
dicom.SeriesInstanceUID = info.SeriesInstanceUID;
dicom.StudyID = info.StudyID; 

% Patient data
dicom.PatientName = info.PatientName;
dicom.PatientID = info.PatientID;
dicom.PatientBirthDate = info.PatientBirthDate;
dicom.PatientSex = info.PatientSex;

% System description 
dicom.MagneticFieldStrength = info.MagneticFieldStrength;
dicom.Modality = info.Modality;
dicom.Manufacturer = info.Manufacturer;
dicom.InstitutionName = info.InstitutionName;
dicom.ManufacturerModelName = info.ManufacturerModelName;
dicom.DeviceSerialNumber = info.DeviceSerialNumber;
dicom.SoftwareVersion = info.SoftwareVersion;

% Acqusition parameters

dicom.ProtocolName = info.ProtocolName;
dicom.SeriesDescription = info.SeriesDescription;
dicom.ChemicalShiftReference = info.ChemicalShiftReference;

dicom.RepetitionTime = info.Private_2005_1030;
%dicom.EchoTime = [];
%dicom.InversionTime = [];
%dicom.MixingTime = []; 

dicom.Samples = info.SpectroscopyAcquisitionDataColumns;
dicom.EchoTimeDisplay = info.Private_2005_1025; % The diplayed echo time

% dicom.NrOfEchoes

% Geometry parameters
dicom.NumberOfFrames = info.NumberOfFrames;
dicom.Rows = info.Rows;
dicom.Columns = info.Columns;

dicom.FOV_ap = info.Private_2005_1085.Item_1.Private_2005_1057; % ap_size
dicom.FOV_fh = info.Private_2005_1085.Item_1.Private_2005_1058; % fh_size
dicom.FOV_rl = info.Private_2005_1085.Item_1.Private_2005_1059; % lr_size

dicom.off_center_ap = info.Private_2005_1085.Item_1.Private_2005_105a; % ap_pff_center
dicom.off_center_fh = info.Private_2005_1085.Item_1.Private_2005_105b; % cc_off_center
dicom.off_cneter_rl = info.Private_2005_1085.Item_1.Private_2005_105c; % lr_off_center

dicom.ap_angulation = info.Private_2005_1085.Item_1.Private_2005_1054; % ap_angulation
dicom.fh_angulation = info.Private_2005_1085.Item_1.Private_2005_1055; % cc_angulation
dicom.lr_angulation = info.Private_2005_1085.Item_1.Private_2005_1056; % lr_angulation

nFid = double(dicom.NumberOfFrames * dicom.Rows * dicom.Columns);
idxFID = 0;
dicom.Slices = 1; warning('Add function for calculation of slice idx')
%dicom.data = zeros(dicom.NumberOfFrames,dicom.Rows,dicom.Columns,dicom.Slices);
sliceIDX = dicom.Slices;
for frameIDX=1:dicom.NumberOfFrames
    for rowIDX=1:dicom.Rows
        for columnIDX = 1:dicom.Columns
            
            idxFID = 1+idxFID;
            
            % Write the data as
            dicom.data{frameIDX,rowIDX,columnIDX,sliceIDX} = info.SpectroscopyData( ...
                (idxFID-1) * 2 * info.SpectroscopyAcquisitionDataColumns + 1: ...
                2:(idxFID) * 2 * info.SpectroscopyAcquisitionDataColumns) ...
                + 1i * info.SpectroscopyData( ...
                (idxFID-1) * 2 * info.SpectroscopyAcquisitionDataColumns + 2: ...
                2:(idxFID) * 2 * info.SpectroscopyAcquisitionDataColumns);
            
            % Time parameters, for dynamic serise time paramers are saved as vectors 
            % with size Nframes*1
            dicom.RepetitionTimeVector{frameIDX,rowIDX,columnIDX,sliceIDX} = info.Private_2005_1030;
            dicom.EchoTime{frameIDX,rowIDX,columnIDX,sliceIDX} = eval(['info.PerFrameFunctionalGroupsSequence.Item_' num2str(1) '.MREchoSequence.Item_1.EffectiveEchoTime']);
            %dicom.InversionTime{frameIDX,rowIDX,columnIDX} = [];
            %dicom.MixingTime{frameIDX,rowIDX,columnIDX} = [];
            
            waitbar(0.5+0.5*idxFID/nFid,waitbar_h);
            
        end
    end
end

close(waitbar_h)

% % *** snip ***
% 
% MRSlabAngulationAP	MRSeries	2005	1069
% MRSlabAngulationFH	MRSeries	2005	106A
% MRSlabAngulationRL	MRSeries	2005	106B
% MRSlabFovAP	MRSeries	2005	106C
% MRSlabFovFH	MRSeries	2005	106D
% MRSlabFovRL	MRSeries	2005	104E
% MRSlabOffcentreAP	MRSeries	2005	104F
% MRSlabOffcentreFH	MRSeries	2005	1050
% MRSlabOffcentreRL	MRSeries	2005	1051
% 
% MRSpectrumEchoTime	Spectrum	2005	1310
% MRSpectrumExtraNumber	Spectrum	2005	1300
% MRSpectrumInversionTime	Spectrum	2005	1312
% MRSpectrumKxCoordinate	Spectrum	2005	1301
% MRSpectrumKyCoordinate	Spectrum	2005	1302
% MRSpectrumLocationNumber	Spectrum	2005	1303
% 
% MRVolumeAngulationAP	MRSeries	2005	1054
% MRVolumeAngulationFH	MRSeries	2005	1055
% MRVolumeAngulationRL	MRSeries	2005	1056
% MRVolumeFovAP	MRSeries	2005	1057
% MRVolumeFovFH	MRSeries	2005	1058
% MRVolumeFovRL	MRSeries	2005	1059
% MRVolumeOffcentreAP	MRSeries	2005	105A
% MRVolumeOffcentreFH	MRSeries	2005	105B
% MRVolumeOffcentreRL	MRSeries	2005	105C
% MRVolumeSelection	MRSeries	2005	1364

