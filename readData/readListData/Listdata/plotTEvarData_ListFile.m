function plotTEvarData_ListFile( varargin )
%plotTEvarData_ListFile Plots the FID of a TE varied experiment reding the
%data from a .list/.data file
%   Detailed explanation goes here

if nargin == 0
    wd = pwd;
    dataDir = '/Data/Ingenia_V_Achieva_NMRdata/140224-Braino-Ingenia-1/';
    if isdir(dataDir)        
        cd(dataDir);
    end
    [FileName,PathName] = uigetfile('*.list','Select the List-file');
    cd(wd);
    FullFileName =  fullfile(PathName, FileName);
    
else
    
    FullFileName = varargin{1};
    
end

[data,info] = readListData(FullFileName);


coilData = squeeze(data(:,1,1,1,1,1,1,1,1,:,1,1));



data = mean(data,10);
data = squeeze(data);

time = (0:0.0005:2047*0.0005)';
time = repmat(time,1,7);


TE = [0.035 0.061 0.092 0.132 0.187 0.273 0.5];

for col=1:size(time,2)

    time(:,col) = time(:,col) + TE(col);

end

figure
plot(time,abs(data),'.'), hold on
%plot(time,real(data),'+'),
%plot(time,imag(data),'o')

figure
plot(repmat(time(:,1),1,13),abs(coilData),'.'), hold on


end

