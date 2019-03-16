function [ dicom, info ] = readDicomMRI_Philips( varargin )
%READDICOMMRI_PHILIPS Reads MRI images in dicom format
%   [ dicom, info ] = readDicomMRI_Philips
%   [ dicom, info ] = readDicomMRI_Philips( dicomFileName )
%
%   Writen 20140813 Anders Tisell


% ------------Initialization----------------
% Select the dicom file


if nargin == 0
    wd = pwd;
    
    cd('/Volumes/andti/Temporary/DICOM/')
    [FileName,PathName] = uigetfile('*.*','Choose a MRS*.dcm file');
    cd(wd)
    Dicomfile = fullfile(PathName,FileName);
    
%     if ~isDICOMmrs(Dicomfile)  
%         error([Dicomfile ' is not a dicom MRS file']) 
%     end

elseif nargin == 1 % A cell aray with filenames including path
    Dicomfile = varargin{1};
    %     if ~isDICOMmrs(Dicomfile), error([Dicomfile ' is not a dicom MRS file']), end
else
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

% Acqusition paramets 

dicom.ProtocolName = info.ProtocolName;
dicom.SeriesDescription = info.SeriesDescription;
dicom.ChemicalShiftReference = info.ChemicalShiftReference;

% Time parameters, for dynamic serise time paramers are saved as vectors 
% with size Nframes*1
dicom.RepetitionTime = info.Private_2005_1030;
%dicom.EchoTime = [];
%dicom.InversionTime = [];
%dicom.MixingTime = []; 

% Geometry parameters
dicom.NumberOfFrames = info.NumberOfFrames;
dicom.Rows = info.Rows;
dicom.Columns = info.Columns;

dicom.FOV_ap = info.Private_2001_105f.Item_1.Private_2005_1074;
dicom.FOV_fh = info.Private_2001_105f.Item_1.Private_2005_1075;
dicom.FOV_rl = info.Private_2001_105f.Item_1.Private_2005_1076;

dicom.off_center_ap = info.Private_2001_105f.Item_1.Private_2005_1078;
dicom.off_center_fh = info.Private_2001_105f.Item_1.Private_2005_1079;
dicom.off_cneter_rl = info.Private_2001_105f.Item_1.Private_2005_107a;

dicom.angulation_ap = info.Private_2001_105f.Item_1.Private_2005_1071;
dicom.angulation_fh = info.Private_2001_105f.Item_1.Private_2005_1072;
dicom.angulation_rl = info.Private_2001_105f.Item_1.Private_2005_1073;

%if length(info.Private_2001_105f)>1, warning('Multiple Items in geometry tag Private_2001_105f'), end


% Mesurment data Data

waitbar_h = waitbar(0.5,waitbar_h);
DicomReadImage = dicomread(info);
dicom.data= DicomReadImage;

close(waitbar_h)
end
% % *** Snip ***
% ImageAcqDeviceProcessingCode      Image	0018	1401
% ImageAcquisitionDate              Image   0008	0022
% ImageAcquisitionTime              Image   0008	0032
% ImageBluePaletColorLUTDescrptr	MRImage	0028	1103
% ImageBluePaletteColorLUTData      MRImage	0028	1203
% ImageColumns                  Spectrum	0028	0011
% ImageDerivationDescription        Image	0008	2111
% ImageDisplayDirection	PresentationState	2005	1385
% ImageGreenPaletColorLUTDescrptr	MRImage	0028	1102
% ImageGreenPaletteColorLUTData     MRImage	0028	1202
% ImageLargestImagePixelValue       Image	0028	0107
% ImageLossyImageCompression	SC  Image	0028	2110
% ImagePatientOrientation	SC      Image	0020	0020
% ImagePhotometricInterpretation	Image	0028	0004
% ImagePixelAspectRatio             Image	0028	0034
% ImagePlanarConfiguration          Image	0028	0006
% ImagePlaneNumber                  MRImage	2001	100A
% ImagePlaneOrientation             MRImage	2001	100B
% ImagePlaneOrientationPatient      MRImage	0020	0037
% ImagePlanePositionPatient         MRImage	0020	0032
% ImagePresentationStateUID         Image	2001	1052
% ImageReconstructionDiameter       MRImage	0018	1100
% ImageRedPaletColorLUTDescrptr     MRImage	0028	1101
% ImageRedPaletteColorLUTData       MRImage	0028	1201
% ImageReference                    PresentationState	0008	1140
% ImageRows                         Spectrum	0028	0010
% ImageSamplesPerPixel              Image	0028	0002
% ImageScanOptions                  MRImage	0018	0022
% ImageSmallestImagePixelValue      Image	0028	0106
% ImageType                         MRImage	0008	0008
% ImagesPerSeriesReference          PresentationState	0008	1115