function [ labraw ] = labraw2mat( filename )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

% Allow user to select a file if input FILENAME is not provided or is empty
if isempty(filename)
  [fn, pn] = uigetfile({'*.raw'},'Select a RAW file');
  if fn ~= 0
    filename = sprintf('%s%s',pn,fn);
  else
    disp('labraw2mat cancelled');
    return;
  end
end

[data,info] = loadLABRAW(filename);




end

