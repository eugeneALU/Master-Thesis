% Spectroscopy Mainline
% Marquette University,   Milwaukee, WI  USA
% Copyright 2001, 2002, 2003 - All rights reserved.
% Fred J. Frigo
%
%
% This function calls other MR spectroscopy related functions
% to compute results from the given raw data file (Pfile).
%
% Oct 15, 2001 - Original
% Feb 11, 2002 - Multi-channel spectro 
% July 30, 2002 - Combine multi-channel results using reference weighting
% Jan 1, 2003   - Label PPM axis
% June 16, 2003 - Pfile updates for MGD2 / 11.0
% Oct 5, 2005  - Pfile support for 14.0

function spectro( pfilename )

% Check to see if pfile name was passed in
if ( nargin == 0 )
   % Enter name of Pfile
   [fname, pname] = uigetfile('*.*', 'Select Pfile');
   pfilename = strcat(pname, fname);
end

% Open Pfile to read reference scan data.
fid = fopen(pfilename,'r', 'ieee-be');
if fid == -1
    err_msg = sprintf('Unable to locate Pfile %s', pfile)
    return;
end

% Determine size of Pfile header based on Rev number
status = fseek(fid, 0, 'bof');
[f_hdr_value, count] = fread(fid, 1, 'real*4');
rdbm_rev_num = f_hdr_value(1);
if( rdbm_rev_num == 7.0 )
    pfile_header_size = 39984;  % LX
elseif ( rdbm_rev_num == 8.0 )
    pfile_header_size = 60464;  % Cardiac / MGD
elseif (( rdbm_rev_num > 5.0 ) && (rdbm_rev_num < 6.0))  
    pfile_header_size = 39940;  % Signa 5.5
else
    % In 11.0 (ME2) the header and data are stored as little-endian
    fclose(fid);
    fid = fopen(pfilename,'r', 'ieee-le');
    status = fseek(fid, 0, 'bof');
    [f_hdr_value, count] = fread(fid, 1, 'real*4');
    if (f_hdr_value == 9.0)
        pfile_header_size= 61464;
    elseif (f_hdr_value == 11.0)  % 12.0 product release
        pfile_header_size= 66072;
    elseif (f_hdr_value > 11.0) & (f_hdr_value < 100.0)  % 14.0 and later
        status = fseek(fid, 1468, 'bof');
        pfile_header_size = fread(fid,1,'integer*4'); 
    else
        err_msg = sprintf('Invalid Pfile header revision: %f', f_hdr_value )
        return;
    end
end  

% Read header to determine number of channels
status = fseek(fid, 0, 'bof');
[hdr_value, count] = fread(fid, 102, 'integer*2');
da_xres = hdr_value(52);
start_recv = hdr_value(101);
stop_recv = hdr_value(102);
nreceivers = (stop_recv - start_recv) + 1;

% some other useful scan parameters.
nex = hdr_value(37);

% Read 'user07' CV  - temperature in degree C
status = fseek(fid, 0, 'bof');
[f_hdr_value, count] = fread(fid, 62, 'real*4');
tempC = f_hdr_value(62)
fclose(fid);


% Index for results  (Right Hand Side of water peak)
start = round(da_xres*0.025);
stop = round(da_xres*0.25);

% Create PPM axis:   Must shift spectrum for temperature
ppm_start_37C = -4.25;
ppm_stop_37C = 0.30;
ppm_per_degree_C=0.01;
ppm_offset = (tempC-37)*ppm_per_degree_C;
ppm_start = ppm_start_37C + ppm_offset;
ppm_stop = ppm_stop_37C + ppm_offset;
combine_x = linspace(ppm_start,ppm_stop,(stop-start+1));

% Loop to compute phase correction vector for each receiver
for loop = 1:nreceivers
   % Compute phase correction results
   [pcor_vector, ref_vector, receiver_weight(loop)] = phascor( pfilename , loop);
   
   % Compute spectroscopy results
   results(loop,:) = spectro_proc( pcor_vector, ref_vector, pfilename, loop );
   
   % Plot results for each channel
   figure(100);
   subplot(nreceivers,1,loop);
   plot( combine_x, real( results(loop,start:stop)), 'k' );
   mesh_results(loop,:)=real(results(loop,start:stop));

   if( loop == 1)
       my_string= sprintf('Spectro results for: %s ',fname);
       %title( my_string);
       xlabel('ppm');
       if (nreceivers == 1)
          ylabel('Absorption');
       end
       set(gca,'XTick',-4.0:0.5:0.0);
       set(gca,'XTickLabel',{'4.0','3.5','3.0','2.5','2.0','1.5','1.0','0.5','0.0'});
   end
   
end

% Calculate combined results if more than one receiver
if nreceivers > 1
  % Find receiver with strongest signal (using Max reference value)
  [max_weight, strongest_receiver] = max(receiver_weight);

  % Dont use channels whose Max is lower than the threshold 
  receiver_threshold = 0.05*max_weight;   % 0.45 default
  combined_weight = 0.0;
  receivers_to_use = 0;
  for loop = 1:nreceivers
    if receiver_weight(loop) >receiver_threshold
       receiver_to_use = receivers_to_use + 1;
       combined_weight = combined_weight + receiver_weight(loop);
    end
  end

  % Linear weighted combination 
  accum_results = zeros(size(results(1,:)));
  for loop = 1:nreceivers
    if receiver_weight(loop) >receiver_threshold
       weight = receiver_weight(loop) / combined_weight;
       accum_results = accum_results + (weight.*real(results(loop,:)));
    else
       weight = 0.0;
    end
    weight
  end
  combined_results = accum_results;

  % Plot combined results
  figure;
  plot( combine_x, real( combined_results(start:stop)), 'k' );
  my_string= sprintf('Combined spectro results for: %s ',fname);
  title( my_string);
  xlabel('ppm');
  ylabel('Absorption');
  set(gca,'XTick',-4.0:0.5:0.0);
  set(gca,'XTickLabel',{'4.0','3.5','3.0','2.5','2.0','1.5','1.0','0.5','0.0'});
  
 
  % Plot multiple channel results
  figure;
  surf(combine_x, 1:1:nreceivers, mesh_results);
  set(gca,'XTick',-4.0:1.0:0.0);
  set(gca,'XTickLabel',{'4.0','3.0','2.0','1.0','0.0'});
  xlabel('ppm');
  zlabel('Absorption');
  ylabel('receive coil');
  
end