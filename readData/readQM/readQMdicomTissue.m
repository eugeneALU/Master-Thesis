function [WM GM CSF NON IC info] = readQMdicomTissue(varargin)

% function [R1 R2 PD B1 E info] = readQMdicom
% function [R1 R2 PD B1 E info] = readQMdicom(qrapDir)

if nargin==0
    wd=pwd;
    cd('~/Project/')
    qrapDir = uigetdir('Choose directory with dicom files');
    cd(wd)
else 
    qrapDir = varargin{1}; 
end

qrapStruct=dir(qrapDir);

NumberOfSlices=(length(qrapStruct)-2)/5;

%fullfile(qrapDir,qrapStruct(3).name)



info = dicominfo(fullfile(qrapDir,qrapStruct(3).name));
WM = zeros(info.Width,info.Height,NumberOfSlices);
GM = zeros(info.Width,info.Height,NumberOfSlices);
CSF = zeros(info.Width,info.Height,NumberOfSlices);
NON = zeros(info.Width,info.Height,NumberOfSlices);
IC = zeros(info.Width,info.Height,NumberOfSlices);

for k=3:(length(qrapStruct))
   
    image_type = qrapStruct(k).name(1:2);
    
    
    slice = str2double(qrapStruct(k).name(strfind(qrapStruct(k).name,'_')+1:strfind(qrapStruct(k).name,'.dcm')-1));
    
    if strcmp(image_type,'WM') 
        WM(:,:,slice)=flipud(rot90(single(dicomread(fullfile(qrapDir,qrapStruct(k).name)))))*info.RescaleSlope/100;
        %imagesc(R1(:,:,slice),[500 1000]), colorbar, title(image_type), pause(0.2) 
        
    elseif strcmp(image_type,'GM')
        GM(:,:,slice)=flipud(rot90(single(dicomread(fullfile(qrapDir,qrapStruct(k).name)))))*info.RescaleSlope/100;
        %imagesc(R2(:,:,slice),[50 120]), colorbar, title(image_type), pause(0.2)
    elseif strcmp(image_type,'CS')
        CSF(:,:,slice)=flipud(rot90(single(dicomread(fullfile(qrapDir,qrapStruct(k).name)))))*info.RescaleSlope/100;
        %imagesc(PD(:,:,slice),[60 100]), colorbar, title(image_type), pause(2)
    elseif strcmp(image_type,'NO')
        NON(:,:,slice)=flipud(rot90(single(dicomread(fullfile(qrapDir,qrapStruct(k).name)))))*info.RescaleSlope/100;
        %imagesc(E(:,:,slice),[0 1]), colorbar, title(image_type), pause(0.2)
    elseif strcmp(image_type,'IC')
        IC(:,:,slice)=flipud(rot90(single(dicomread(fullfile(qrapDir,qrapStruct(k).name)))))*info.RescaleSlope/100;
        
    else
        error(['Do not know what to do with ' qrapStruct(k).name])
    end
    
   
end


% 
% 
% %Test for errors
% 
% R1slice = R1(:,:,15);
% R2slice = R2(:,:,15);
% PDslice = PD(:,:,15);
% B1slice = B1(:,:,15);
% 
% tissue = find(PDslice > 50 & R2slice < 20 &  R2slice > 2 & R1slice < 5 & R1slice > 0.2);
% 
% length(tissue)
% R1vec = R1slice(tissue);
% R2vec = R2slice(tissue);
% PDvec = PDslice(tissue);
% B1vec = B1slice(tissue);
% 
% figure('Name','R1'), hist(R1vec(R1vec > 0.2 & R1vec < 5),0:0.1:15)
% figure('Name','R2'), hist(R2vec(R2vec > 2 & R2vec < 20 ),0:0.1:15)
% figure('Name','PD'), hist(PDvec(PDvec > 40 & PDvec < 120),40:1:120)
% figure('Name','B1'), hist(B1vec(B1vec > 80 & B1vec < 120),80:1:120)
%
