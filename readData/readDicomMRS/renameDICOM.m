function renameDICOM( varargin )
%renameDICOM copy spectro dicom files listed in dicomdir
%   renameDICOM(dicomdir) dicomdir is the path to wre the dicomdir file
%   Looks at the dicom files in dicom dir and
%
%   Code addpeted form COPYSERIESFROMDICOMDIR Written 2010-05-25 by Jadrian Miles
%
%   Se also: oarsDicomdirAT 
%
%   Written 2014-08-06 by Anders Tisell 

if nargin == 0
    wd = pwd;
    
    cd('/Volumes/data_radiofys/Data/MR/Anders/')
    [FileName,PathName] = uigetfile('*.*','Choose a DICOMDIR file');
    cd(wd)
    dicomdir = fullfile(PathName,FileName);
    
%     if ~isDICOMmrs(Dicomfile)  
%         error([Dicomfile ' is not a dicom MRS file']) 
%     end

elseif nargin == 1 % A cell aray with filenames including path
    dicomdir = varargin{1};
    %     if ~isDICOMmrs(Dicomfile), error([Dicomfile ' is not a dicom MRS file']), end
else
    error('Could not read dicom file, wrong number of arguments')
end

dicomdirPath = dicomdir(1:end-8);
patient = parseDicomdirAT(dicomdir);

waitbar_h = waitbar(0,'Copying dicom MRS files');

for i=1:length(patient)
    for j=1:length(patient{i}.study)
        warning(['Cheek if '  patient{i}.study{j}.info.StudyID ' is the SERLIN number'])
        
        fileDir = ['/Volumes/andti/Temporary/DICOM/',patient{i}.study{j}.info.StudyID];
        
        mkdir(fileDir);
        
        for k=1:length(patient{1}.study{1}.series)
            
            s = patient{i}.study{j}.series{k};
            
            try
                for l=1:length(s.image) % Will the length ever be more than 1?
                    
                    switch lower(s.image{l}.info.DirectoryRecordType)
                        
                        case 'spectroscopy'
                            sourceFile = fullfile(dicomdirPath,strrep(s.image{l}.info.ReferencedFileID,'\','/'));
                            destFile = fullfile(fileDir,['MRS_' s.image{l}.info.ReferencedFileID(10:13) '_f' num2str(s.image{l}.info.NumberOfFrames) 'r' num2str(s.image{l}.info.Rows) 'c' num2str(s.image{l}.info.Columns)]);
                           % disp(['Copy ' sourceFile ' to ' destFile])
                            copyfile(sourceFile,destFile)
                        
                        case 'image'
                            sourceFile = fullfile(dicomdirPath,strrep(s.image{l}.info.ReferencedFileID,'\','/'));
                            try
                            destFile = fullfile(fileDir,['Image_' s.image{l}.info.ReferencedFileID(10:13) '_f' num2str(s.image{l}.info.NumberOfFrames) 'r' num2str(s.image{l}.info.Rows) 'c' num2str(s.image{l}.info.Columns)]);
                            catch
                            destFile = fullfile(fileDir,['Image_' s.image{l}.info.ReferencedFileID(10:13) '_fXr' num2str(s.image{l}.info.Rows) 'c' num2str(s.image{l}.info.Columns)]);    
                            end
                            
                            %disp(['Copy ' sourceFile ' to ' destFile])
                            copyfile(sourceFile,destFile)
                         
                        case 'private'
                            sourceFile = fullfile(dicomdirPath,strrep(s.image{l}.info.ReferencedFileID,'\','/'));
                            destFile = fullfile(fileDir,['Private_' s.image{l}.info.ReferencedFileID(10:13) '_f' num2str(s.image{l}.info.NumberOfFrames) 'r' num2str(s.image{l}.info.Rows) 'c' num2str(s.image{l}.info.Columns)]);
                            disp(['Copy ' sourceFile ' to ' destFile])
                            copyfile(sourceFile,destFile)    
                        
                        %otherwise
                        %    disp(['Did not find record type:' s.image{l}.info.DirectoryRecordType])
                    end
                end
                
            catch err
                err.identifier
            end
            
            waitbar(k/length(patient{i}.study{j}.series),waitbar_h);
        
        end
    end
end
close(waitbar_h)
end

