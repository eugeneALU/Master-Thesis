function [SRE,LRE,GLN,RP,RLN,LRLGLE,LRHGLE,HGRE,LGRE,SRHGLE,SRLGLE] = GLRLM(matrix, Bin_size)
%GLR Summary 
%   input:
%       matrix: target, only one channel, not quantize
%       Bin_size: number of quantize level
%   output:
%       SRE: Short-Run Emphasis
%       LRE: Long-Run Emphasis
%       GLN: Gray Level Non-Uniformity 
%       RP: Run Percentage
%       RLN: Run Length Non-uniformity
%       HGRE: High Gray-Level Run Emphasis
%       LGRE: Low Gray-Level Run Emphasis
%       SRLGE: Short Run Low Gray-Level Emphasis
%       SRHGE: Short Run High Gray-Level Emphasis
%       LRLGLE: Long Run Low Gray-Level Emphasis 
%       LRHGLE: Long Run High Gray-Level Emphasis

%% quantize matrix //locally!!!
MIN = min(matrix(:));
matrix = matrix - MIN;
MAX = max(matrix(:));
levels = MAX/Bin_size:MAX/Bin_size:MAX-MAX/Bin_size;
matrix = imquantize(matrix, levels);
[row, col] = size(matrix);
Length_max = max(row, col);

%% initialize four direction of GLRLM
P0 = zeros(Bin_size, Length_max);
P45 = zeros(Bin_size, Length_max);
P90 = zeros(Bin_size, Length_max);
P135 = zeros(Bin_size, Length_max);

% create sequence contain diagonal data for p45, p135
seq_45 = zigzag(matrix);
seq_135 = zigzag(fliplr(matrix));

% padding the matrix, easiear for caculate start/end point later(for 0,90)
matrix = padarray(matrix,[1 1]);

%% find the length for each level (0,90)
for i = 1:Bin_size
    % find corresponding pixels
    cor = uint8(matrix == i);
    
    % find the start and end points of the run length (must ignore the padding 0)
    % endPoint: current - next == 1 ; startPoint: current - previous == 1
    p0s = (cor(2:end-1, 2:end-1) - cor(2:end-1, 1:end-2)) == 1;
    p0e = (cor(2:end-1, 2:end-1) - cor(2:end-1, 3:end)) == 1;
    p90s = (cor(2:end-1, 2:end-1) - cor(1:end-2, 2:end-1)) == 1;
    p90e = (cor(2:end-1, 2:end-1) - cor(3:end, 2:end-1)) == 1;
    
    % get index of value = 1 
    % index are in column-wise, for horizon case(0,135) we need to transpose matrix first
    p0s = p0s'; p0s = find(p0s(:));
    p0e = p0e'; p0e = find(p0e(:));
    p90s = find(p90s(:));
    p90e = find(p90e(:));
    
    % calculate the length, +1 since we consider one element as length 1 (EX:start=8; end=8 -> length=1)
    length0 = p0e - p0s + 1; 
    length90 = p90e - p90s + 1;
    
    % get GLRLM
    P0(i,:) = hist(length0, 1:Length_max);
    P90(i,:) = hist(length90, 1:Length_max);
end
%% find the length for each level (45,135)
for i=1:length(seq_45)
    x_45=seq_45{i};
    % run length Encode of each vector
    index = [ find(x_45(1:end-1) ~= x_45(2:end)), length(x_45) ];
    len = diff([ 0 index ]); % run lengths
    val = x_45(index);       % run values
    % compute current numbers (or contribution) for each bin in GLRLM
    temp =accumarray([val;len]',1,[Bin_size, Length_max]);
    P45 = temp + P45; % accumulate each contribution
end
for i=1:length(seq_135)    
    x_135=seq_135{i};
    % run length Encode of each vector
    index = [ find(x_135(1:end-1) ~= x_135(2:end)), length(x_135) ];
    len = diff([ 0 index ]); % run lengths
    val = x_135(index);      % run values
    % compute current numbers (or contribution) for each bin in GLRLM
    temp =accumarray([val;len]',1,[Bin_size, Length_max]);
    P135 = temp + P135; % accumulate each contribution
end


%% calculate individually and mean later for four direction
P_all = cat(3,P0,P45,P90,P135);
SRE = zeros(1,4);
LRE = zeros(1,4);
HGRE = zeros(1,4);
LGRE = zeros(1,4);
LRHGLE = zeros(1,4);
LRLGLE = zeros(1,4);
SRHGLE = zeros(1,4);
SRLGLE = zeros(1,4);
RP = zeros(1,4);
RLN = zeros(1,4);
GLN = zeros(1,4);

for dir = 1:4
    P = P_all(:,:,dir);
    %% Prepare some matrix
    SUM = sum(P(:));
    Jsquare = (1:Length_max).^2;
    Isquare = (1:Bin_size).^2;
    sum_P1 = sum(P,1);
    sum_P2 = sum(P,2);
    %% calculte features
    SRE(dir) = sum(sum_P1 ./ Jsquare) / SUM;
    LRE(dir) = sum(sum_P1 .* Jsquare) / SUM;
    GLN(dir) = sum(sum_P2 .^2) / SUM;
    RLN(dir) = sum(sum_P1 .^2) / SUM;
    RP(dir) = SUM / (row*col);
    % sum along dimension 2 need a transpose to fit the dimension of Isquare
    HGRE(dir) = sum(sum_P2' .* Isquare) / SUM;
    LGRE(dir) = sum(sum_P2' ./ Isquare) / SUM;
    P_T_J = P .* Jsquare;
    P_D_J = P ./ Jsquare;
    LRHGLE(dir) = sum(sum(P_T_J,2)' .* Isquare) / SUM;
    LRLGLE(dir) = sum(sum(P_T_J,2)' ./ Isquare) / SUM;
    SRHGLE(dir) = sum(sum(P_D_J,2)' .* Isquare) / SUM;
    SRLGLE(dir) = sum(sum(P_D_J,2)' ./ Isquare) / SUM;
end
%% Calculate features average all direction
SRE = mean(SRE);
LRE = mean(LRE);
HGRE = mean(HGRE);
LGRE = mean(LGRE);
LRHGLE = mean(LRHGLE);
LRLGLE = mean(LRLGLE);
SRHGLE = mean(SRHGLE);
SRLGLE = mean(SRLGLE);
RP = mean(RP);
RLN = mean(RLN);
GLN = mean(GLN);
end

%% TEST (correct)
% I =
%      1     1     1     2     2
%      3     4     2     2     3
%      4     4     4     4     4
%      5     5     3     3     3
%      1     1     3     4     5
% GLRLMS 0 =
%      0     1     1     0     0
%      0     2     0     0     0
%      3     0     1     0     0
%      2     0     0     0     1
%      1     1     0     0     0
% GLRLMS 45 =
%      5     0     0     0     0
%      0     2     0     0     0
%      4     1     0     0     0
%      5     1     0     0     0
%      3     0     0     0     0
% GLRLMS 90 =
%      5     0     0     0     0
%      2     1     0     0     0
%      4     1     0     0     0
%      5     1     0     0     0
%      3     0     0     0     0
% GLRLMS 135 =
%      5     0     0     0     0
%      4     0     0     0     0
%      6     0     0     0     0
%      5     1     0     0     0
%      3     0     0     0     0
