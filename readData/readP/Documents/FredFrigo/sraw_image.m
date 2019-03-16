% sraw_image.m  -  plot raw data for a single echo from a Pfile
%
% Marquette University,   Milwaukee, WI  USA
% Copyright 2002, 2003 - All rights reserved.
% Fred Frigo
%
% Date:  Jan 21, 2002     - based  raw_image.m
%
%  - create 2 images, one for reference data, one for water supressed data
%

function sraw_image( pfile )

% Check to see if pfile name was passed in
if ( nargin == 0 )
   % Enter name of Pfile
   [fname, pname] = uigetfile('*.*', 'Select Pfile');
   pfile = strcat(pname, fname);
end

i = sqrt(-1);

% Open Pfile to read reference scan data.
fid = fopen(pfile,'r', 'ieee-be');
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
    fid = fopen(pfile,'r', 'ieee-le');
    status = fseek(fid, 0, 'bof');
    [f_hdr_value, count] = fread(fid, 1, 'real*4');
    if (f_hdr_value == 9.0)
        pfile_header_size= 61464;
    elseif (f_hdr_value == 11.0)
        pfile_header_size= 66072;
    elseif (f_hdr_value > 11.0) & (f_hdr_value < 100.0)  % 14.0 and later
        status = fseek(fid, 1468, 'bof');
        pfile_header_size = fread(fid,1,'integer*4');     
    else
        err_msg = sprintf('Invalid Pfile header revision: %f', f_hdr_value )
        return;
    end
end  

status = fseek(fid, 0, 'bof');

% Read header information
[hdr_value, count] = fread(fid, 5122, 'integer*2');
nechoes = hdr_value(36);
nframes = hdr_value(38);
point_size = hdr_value(42);
da_xres = hdr_value(52);
da_yres = hdr_value(53);
rc_xres = hdr_value(54);
rc_yres = hdr_value(55);
start_recv = hdr_value(101);
stop_recv = hdr_value(102);
nreceivers = (stop_recv - start_recv) + 1;
slices_in_pass = hdr_value(5122);

% Read 'user19' CV  - number of reference frames
status = fseek(fid, 0, 'bof');
[f_hdr_value, count] = fread(fid, 74, 'real*4');
rdbm_rev_num = f_hdr_value(1);
num_ref_frames = f_hdr_value(74);


% Compute size (in bytes) of each frame, echo and slice
data_elements = da_xres*2*(da_yres-1);
frame_size = da_xres*2*point_size;
echo_size = frame_size*da_yres;
slice_size = echo_size*nechoes;
mslice_size = slice_size*nreceivers;
    

for k = 1:1000		% give a large number 1000 to loop forever
    
  % Enter slice number to plot
  my_slice = 1;
  if ( slices_in_pass > 1 )
      slice_msg = sprintf('Enter the slice number: [1..%d]',slices_in_pass); 
      my_slice = input(slice_msg);
      if (my_slice > slices_in_pass)
          err_msg = sprintf('Invalid number of slices. Slice number set to 1.')
          my_slice = 1;
      end
  end
  
  % Enter echo number to plot
  my_echo = 1;
  if ( nechoes > 1 )
      echo_msg = sprintf('Enter the echo number: [1..%d]',nechoes);
      my_echo = input(echo_msg);
      if (my_echo > nechoes )
          err_msg = sprintf('Invalid echo number. Echo number set to 1.')
          my_echo = 1;
      end
  end
  
  % Enter receiver number to plot
  my_receiver = 1;
  if ( nreceivers > 1 )
      recv_msg = sprintf('Enter the receiver number: [1..%d]',nreceivers);
      my_receiver = input(recv_msg);
      if (my_receiver > nreceivers)
          err_msg = sprintf('Invalid receiver number. Receiver number set to 1.')
          my_receiver = 1;
      end      
  end


  % Compute offset in bytes to start of frame.  (skip baseline view)
  file_offset = pfile_header_size + ((my_slice - 1)*mslice_size) + ...
                      + ((my_receiver -1)*slice_size) + ...
                      + ((my_echo-1)*echo_size) + ...
                      + (frame_size);
  
  status = fseek(fid, file_offset, 'bof');

  % read data: point_size = 2 means 16 bit data, point_size = 4 means EDR )
  if (point_size == 2 )
      [raw_data, count] = fread(fid, data_elements, 'integer*2');
  else
      [raw_data, count] = fread(fid, data_elements, 'integer*4');
  end

  %frame_data = zeros(da_xres);
  for j = 1:(da_yres -1)
     row_offset = (j-1)*da_xres*2;
     for m = 1:da_xres
        frame_data(j,m) = raw_data( ((2*m)-1) + row_offset) + i*raw_data((2*m) + row_offset);
     end
  end
  
  figure(k);
  subplot(2,1,1);
  imagesc( abs(frame_data([1:num_ref_frames],:) ));
  %title(sprintf('Magnitude of Raw Reference Spectro Data, slice %d, recv %d, echo %d', my_slice, my_receiver, my_echo));
  title('Magnitude of Raw Reference Data');
  xlabel('time');
  ylabel('frame number');
  subplot(2,1,2);
  imagesc( abs(frame_data([num_ref_frames+1:nframes-1],:)));
  title('Magnitude of Raw Water Suppressed Data');
  xlabel('time');
  ylabel('frame number');

  % check to see if we should quit
  quit_answer = input('Press Enter to continue, "q" to quit:', 's');
  if ( size( quit_answer ) > 0 )
     if (quit_answer == 'q')
         break;
     end
  end
  
end
fclose(fid);
