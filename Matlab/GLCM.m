function [E, SUME, MAXP, ASM, COR, CON, HOMO, AUTO, CSHAD,...
          CPROM, DIFE, DIFAV, SUMAV, DIFVAR, SUMVAR, IMC1, IMC2, SOS] = GLCM(matrix, Bin_size)
%GLCM Summary
%   input:
%       matrix: target, only one channel, not quantize
%       Bin_size: number of quantize level
%   output:
%       MAXP: maximum probability
%       E: Entropy
%       ASM: Angular Second Moment (Energy)
%       SUME: Sum Entropy
%       COR: Correlation
%       CON: Contrast
%       HOMO: Homogenuity
%       AUTO: Autocorrelation
%       CSHAD: Cluster Shade
%       CPROM: Cluster Prominence
%       DIFE: Difference entropy
%       DIFAV: Difference Variance
%       SUMAV: Sum Average
%       DIFVAR: Difference Variance
%       SUMVAR: Sum Variance
%       IMC1: Informational Measure of Correlation (IMC) 1
%       IMC2: Informational Measure of Correlation (IMC) 2
%       SOS: Sum of Squares

Epsilon = 1e-17; % in case log(0) happen
%% compute GLCM for 0, 45, 90, 135 degree // quantize matrix locally!!!
G0 = graycomatrix(matrix,'NumLevels',Bin_size, 'GrayLimits',[], 'offset', [0 1], 'Symmetric', true);
G45 = graycomatrix(matrix,'NumLevels',Bin_size, 'GrayLimits',[], 'offset', [-1 1], 'Symmetric', true);
G90 = graycomatrix(matrix,'NumLevels',Bin_size, 'GrayLimits',[], 'offset', [-1 0], 'Symmetric', true);
G135 = graycomatrix(matrix,'NumLevels',Bin_size, 'GrayLimits',[], 'offset', [-1 -1], 'Symmetric', true);

%% sum all of four direction
G0 = G0 / sum(G0(:));
G45 = G45 / sum(G45(:));
G90 = G90 / sum(G90(:));
G135 = G135 / sum(G135(:));
% G = G0 + G45 + G90 + G135;
% G = G ./ 4;
G_all = cat(3,G0,G45,G90,G135);
I = 1:Bin_size;
J = 1:Bin_size;
KDIF = 0:Bin_size-1;
KSUM = 2:2*Bin_size;    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
E = zeros(1,4);
SUME = zeros(1,4);
MAXP = zeros(1,4);
ASM = zeros(1,4);
COR = zeros(1,4);
CON = zeros(1,4);
HOMO = zeros(1,4);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
AUTO = zeros(1,4);
CSHAD = zeros(1,4);
CPROM = zeros(1,4);
DIFE= zeros(1,4);
DIFAV = zeros(1,4);
SUMAV = zeros(1,4);
DIFVAR = zeros(1,4);
SUMVAR = zeros(1,4);
IMC1 = zeros(1,4);
IMC2 = zeros(1,4);
SOS = zeros(1,4);

for dir = 1:4
    G = G_all(:,:,dir);
    %% Compute sum/difference p 
    Psum = zeros([1, 2*Bin_size - 1]);
    Pdifference = zeros([1, Bin_size]);
    for k = 2:2*Bin_size
        for i = 1:Bin_size
            if (k-i >= 1 && k-i <= Bin_size)
                Psum(1,k-1) = Psum(1,k-1) + G(i,k-i);
            end
        end
    end
    % k = 0 deal seperatly, incae using 3 for loop here to try to accelerate the processing
    for k = 1:Bin_size-1
        for i = 1:Bin_size
            if (i+k <= Bin_size)
                Pdifference(1,k+1) = Pdifference(1,k+1) + G(i,i+k);
            end
            if (i-k >= 1)
                Pdifference(1,k+1) = Pdifference(1,k+1) + G(i,i-k);
            end
        end
    end
    for i = 1:Bin_size
         Pdifference(1,1) = Pdifference(1,1) + G(i,i);
    end

    %% Compute the features
    MAXP(dir) = max(G(:));
    Tmp = G .* log(G + Epsilon);  % add a small value in case G is 0
    E(dir) = -sum(Tmp(:));
    Tmp = Psum .* log(Psum + Epsilon);
    SUME(dir) = -sum(Tmp(:));
    % following can compute by graycoprops(glcm(unnormalized)), however much slower (10 times slower)
    square = G .^2;
    ASM(dir) = sum(square(:));

    Tmp = abs((1:Bin_size) - (1:Bin_size)');
    CON(dir) = sum(G(:) .* (Tmp(:).^2));
    HOMO(dir) = sum(G(:) ./ (1+Tmp(:)));

    MEANX = (1:Bin_size) * sum(G,2);        % dot product, calculate sum automatically
    MEANY = (1:Bin_size) * sum(G,1)';       % use transpose to fit the dimension that Matlab will do it as dot product 
    STDX = (((1:Bin_size) - MEANX).^2) * sum(G,2);
    STDY = (((1:Bin_size) - MEANY).^2) * sum(G,1)';
    SUMMEAN = MEANX + MEANY;
    
    for i = 1:Bin_size
        for j = 1:Bin_size
            COR(dir) = COR(dir) + (i-MEANX)*(j-MEANY)*G(i,j);
        end
    end
    COR(dir) = COR(dir) / sqrt(STDX*STDY);
    AUTO(dir) = sum((G .* (J .* I')),[1,2]);
    Tmp = (I'+J-SUMMEAN);
    Power = Tmp .^ 3;
    CSHAD(dir) = sum((Power .* G), [1,2]);
    CPROM(dir) = sum(((Power.*Tmp) .* G), [1,2]);
    
    Tmp = Pdifference .* log(Pdifference + Epsilon);
    DIFE(dir) = -sum(Tmp(:));
    % http://murphylab.web.cmu.edu/publications/boland/boland_node26.html
    DIFAV(dir) = ((KDIF * Pdifference'));
    SUMAV(dir) = ((KSUM * Psum'));
    DIFVAR(dir) = ((KDIF.^2) * Pdifference');
    SUMVAR(dir) = (((KSUM - SUME(dir)).^2) * Psum');
    
    PX = sum(G,2);
    PY = sum(G,1);
    PXPY = (PY .* PX);
    LOGPXPY = log(PXPY + Epsilon);
    HX = -sum(PX .* log(PX + Epsilon));
    HY = -sum(PY .* log(PY + Epsilon));
    HXY1 = -sum((G .* LOGPXPY),[1,2]);
    HXY2 = -sum((PXPY .* LOGPXPY),[1,2]);
    IMC1 = (E-HXY1)/ max(HX,HY);
    IMC2 = sqrt(1-exp(-2*(HXY2-E)));
    Tmp = (1:Bin_size)' - MEANX;
    SOS = sum(((Tmp .^2) .* G),[1,2]);
end
E = mean(E);
SUME = mean(SUME);
MAXP = mean(MAXP);
ASM = mean(ASM);
COR = mean(COR);
CON = mean(CON);
HOMO = mean(HOMO);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
AUTO = mean(AUTO);
CSHAD = mean(CSHAD );
CPROM = mean(CPROM);
DIFE = mean(DIFE);
DIFAV = mean(DIFAV);
SUMAV = mean(SUMAV);
DIFVAR = mean(DIFVAR);
SUMVAR = mean(SUMVAR);
IMC1 = mean(IMC1);
IMC2 = mean(IMC2);
SOS = mean(SOS);
end
%% TEST (Correct)
% glcm = [
%      0     1     2     3
%      1     1     2     3
%      1     0     2     0
%      0     0     0     3]
% Normalizes glcm =[
% 
%          0    0.0526    0.1053    0.1579
%     0.0526    0.0526    0.1053    0.1579
%     0.0526         0    0.1053         0
%          0         0         0    0.1579]
%
%Result from code:          Result from API:
% COR = 0.0783              Correlation: 0.0783
% 
% ASM = 0.1191              Energy: 0.1191
% 
% CON = 2.8947              Contrast: 2.8947
% 
% HOMO =0.5658              Homogeneity: 0.5658

%% Calculate elapsed time
% tic;
% elapsed = toc;
% fprintf('TIC TOC1: %g\n', elapsed);