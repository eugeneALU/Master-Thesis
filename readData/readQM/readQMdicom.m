function [R1 R2 PD B1 E info] = readQMdicom(varargin)

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
T1 = zeros(info.Width,info.Height,NumberOfSlices);
T2 = zeros(info.Width,info.Height,NumberOfSlices);
PD = zeros(info.Width,info.Height,NumberOfSlices);
B1 = zeros(info.Width,info.Height,NumberOfSlices);
E = zeros(info.Width,info.Height,NumberOfSlices);

for k=3:(length(qrapStruct))
   
    %image_type = qrapStruct(k).name(16:17);
    %slice = str2double(qrapStruct(k).name(12:14));
    
    image_type = qrapStruct(k).name(10:11);
    slice = str2double(qrapStruct(k).name(7:8));
    if strcmp(image_type,'T1') 
        T1(:,:,slice)=flipud(rot90(single(dicomread(fullfile(qrapDir,qrapStruct(k).name)))/1000))*info.RescaleSlope;
        %imagesc(R1(:,:,slice),[500 1000]), colorbar, title(image_type), pause(0.2) 
        
    elseif strcmp(image_type,'T2')
        T2(:,:,slice)=flipud(rot90(single(dicomread(fullfile(qrapDir,qrapStruct(k).name)))/1000))*info.RescaleSlope;
        %imagesc(R2(:,:,slice),[50 120]), colorbar, title(image_type), pause(0.2)
    elseif strcmp(image_type,'B1')
        B1(:,:,slice)=flipud(rot90(single(dicomread(fullfile(qrapDir,qrapStruct(k).name)))))*info.RescaleSlope;
        %imagesc(B1(:,:,slice),[80 120]), colorbar, title(image_type), pause(0.2)
    elseif strcmp(image_type,'PR')
        PD(:,:,slice)=flipud(rot90(single(dicomread(fullfile(qrapDir,qrapStruct(k).name)))))*info.RescaleSlope;
        %imagesc(PD(:,:,slice),[60 100]), colorbar, title(image_type), pause(2)
    elseif strcmp(image_type,'RE')
        E(:,:,slice)=flipud(rot90(single(dicomread(fullfile(qrapDir,qrapStruct(k).name)))))*info.RescaleSlope;
        %imagesc(E(:,:,slice),[0 1]), colorbar, title(image_type), pause(0.2)
 
    else
        error(['Do not know what to do with ' qrapStruct(k).name])
    end
    
   
end


R1 = 1./T1;
R2 = 1./T2;
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
