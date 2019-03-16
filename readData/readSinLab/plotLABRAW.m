function [ FID, data, info ] = plotLABRAW(filename, varargin )
%%plotSINLAB Load and plots data of philips LABRAW file
%   freadSINLAB(SinLABfile)
%
%   Function underdevelopment now readSINLAB.m for windows is used to
%   produce a mat file with data and info struct.
%   Anders Tisell 140730
%
%   See also:  PLOTSPAR PLOTLIST
%
%   Dependencies: loadSINLAB.m


% Allow user to select a file if input FILENAME is not provided or is empty
if nargin < 1 || isempty(filename)
    wd=pwd;
    cd('/Volumes/data_radiofys/Data/MR/')
    [fn, pn] = uigetfile({'*.raw'},'Select a RAW file');
    cd(wd);
    if fn ~= 0
        filename = sprintf('%s%s',pn,fn);
    else
        disp('readSINLAB cancelled');
        return;
    end
end

[data,info] = loadLABRAW(filename);

if info.dims.nKy == 1 && info.dims.nKz == 1
    disp('Read SVS raw data')
    for rowIDX = 1:info.dims.nRows
        
        if info.dims.nMixes == 2
            
            DataAverage = mean(mean(data,1),12);
            RawMet = squeeze(DataAverage(1,:,1,1,1,1,1,1,1,rowIDX,1,1));
            RawWater = squeeze(DataAverage(1,:,1,1,1,1,1,1,1,rowIDX,2,1));
            
            PhasedWater = RawWater.*exp(-i*angle(RawWater));
            PhasedMet = RawMet.*exp(-i*angle(RawWater));
            
            FID{rowIDX,1} = PhasedMet;
            FID{rowIDX,2} = PhasedWater;
        else
            % If no water referens exist set FID{:,2} to all zeros
            disp(['No water referenec in ' filename])
            DataAverage = mean(mean(data,1),12);
            FID{rowIDX,1} = squeeze(DataAverage(1,:,1,1,1,1,1,1,1,rowIDX,1,1));
            FID{rowIDX,2} = zeros(size(FID{rowIDX,1}));
            
        end
    end
else
    disp('Read CSI raw data')
    CSIidx = 0;
    for KyIDX = 1:info.dims.nKy
        for KzIDX = 1:info.dims.nKz
            for rowIDX = 1:info.dims.nRows
                CSIidx = CSIidx + 1;
                if info.dims.nMixes == 2
                    
                    DataAverage = mean(mean(data,1),12);
                    RawMet = squeeze(DataAverage(1,:,KyIDX,KzIDX,1,1,1,1,1,rowIDX,1,1));
                    RawWater = squeeze(DataAverage(1,:,KyIDX,KzIDX,1,1,1,1,1,rowIDX,2,1));
                    
                    PhasedWater = RawWater.*exp(-i*angle(RawWater));
                    PhasedMet = RawMet.*exp(-i*angle(RawWater));
                    
                    FID{CSIidx,1} = PhasedMet;
                    FID{CSIidx,2} = PhasedWater;
                    
                else
                    
                    DataAverage = mean(mean(data,1),12);
                    FID{CSIidx,1} = squeeze(DataAverage(1,:,KyIDX,KzIDX,1,1,1,1,1,rowIDX,1,1));
                    FID{CSIidx,2} = zeros(size(FID{CSIidx,1}));
                                           
                end
            end
        end
    end
    
    
    
end


% Calculate FFT frequencys and PPM scale, assume 1.5 T Syntesister freq 
% 63895839 since Syntesiter frequence not exported in teh RAW file
% Set Sample frequecy to 32 kHz since it seems like the RAW file is allways
% sampled with 32 kHz

SyFq = 63895839;
SampFq = 32*10^3;
dFq = SampFq/double(info.dims.nKx);
Fq = SyFq-(SampFq/2):dFq:SyFq+(SampFq/2)-dFq; % FFTshift puts zero fq at idx/2 + 1 
refFq = SyFq / (4.67*10^-6 + 1);
PPM = (Fq - refFq)/refFq * 10^6;
Time = double(0:1:info.dims.nKx-1)/SampFq;

for rowIDX = 1:size(FID,1)

FFT{rowIDX,1} = fftshift(fft(FID{rowIDX,1},[],2),2);
FFT{rowIDX,2} = fftshift(fft(FID{rowIDX,2},[],2),2);

figure('Name',['Plot FID and Spectrum from .RAW data row:' num2str(rowIDX)])
subplot(2,2,1), % Metabolite FID 
plot(Time,real(FID{rowIDX,1}),'k.'), %hold on, plot(Time,imag(FID{rowIDX,1}),'r.'), plot(Time,abs(FID{rowIDX,1}),'b.')
subplot(2,2,2), % Water FID
plot(Time,real(FID{rowIDX,2}),'k.'), %hold on, plot(Time,imag(FID{rowIDX,2}),'r.'), plot(Time,abs(FID{rowIDX,2}),'b.')
subplot(2,2,3), % Metabolt spectrum
plot(PPM,real(FFT{rowIDX,1}),'k-'), xlim([0 4]) %hold on, plot(PPM,imag(FFT{rowIDX,2}),'r-'), plot(PPM,abs(FFT{rowIDX,1}),'b-')
subplot(2,2,4), % Water Spectrum 
plot(PPM,real(FFT{rowIDX,2}),'k-'), xlim([0 9.34]) %hold on, plot(PPM,imag(FFT{rowIDX,2}),'r-'), plot(PPM,abs(FFT{rowIDX,2}),'b-')

if mod(rowIDX,10) == 0, close all, end % Ugly function to stop java memory crasches

end
end

