% noise_weights.m  -  read prescan noise weights from Pfile and print
%                  -  compute std deviation of noise data for each channel
%
% Author: Fred J. Frigo
% Date:  March 3, 2010
%        Dec 30,  2010   - read center freq
%
%

function noise_weights( pfile )

if(nargin == 0)
    [fname, pname] = uigetfile('*.*', 'Select Pfile');

    pfile = strcat(pname, fname);
end

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
    % In 11.0 and later the header and data are stored as little-endian
    fclose(fid);
    fid = fopen(pfile,'r', 'ieee-le');
    status = fseek(fid, 0, 'bof');
    [f_hdr_value, count] = fread(fid, 1, 'real*4');
    if (f_hdr_value == 9.0)  % 11.0 product release
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

% Determine number of channels, recon scale factor, prescan info
if (f_hdr_value < 100.0)
    status = fseek(fid, 168, 'bof');
    rational_scale_factor = fread(fid,1,'real*4');
    status = fseek(fid, 200, 'bof');  %188
    start_recv = fread(fid,1,'integer*2');
    stop_recv = fread(fid,1,'integer*2');
    num_receivers = stop_recv - start_recv + 1;
    status = fseek(fid, 82, 'bof');
    point_size = fread(fid,1,'integer*2'); 
else
    err_msg = sprintf('Invalid Pfile header revision: %f', f_hdr_value )
end  

if (f_hdr_value < 12.0) % 11.0 + 12.0
    status = fseek(fid, 792, 'bof');
    [receiver_weights, count] = fread(fid,num_receivers,'real*4');  % 16 Max
    status = fseek(fid, 408, 'bof');
    ps_command = fread(fid,1,'integer*4');  % autoprescan vs manual
    status = fseek(fid, 412, 'bof');  
    mps_r1 = fread(fid,1,'integer*4');  
    status = fseek(fid, 416, 'bof'); 
    mps_r2 = fread(fid,1,'integer*4');  
    status = fseek(fid, 424, 'bof'); 
    mps_freq = fread(fid,1,'integer*4');  
    status = fseek(fid, 428, 'bof');  
    aps_r1 = fread(fid,1,'integer*4');  
    status = fseek(fid, 432, 'bof'); 
    aps_r2 = fread(fid,1,'integer*4'); 
    status = fseek(fid, 440, 'bof'); 
    aps_freq = fread(fid,1,'integer*4');  
    status = fseek(fid, 456, 'bof'); %440 center freq
    aps_or_mps = fread(fid,1,'integer*4'); 
elseif (f_hdr_value > 12.0) & (f_hdr_value < 100.0)  % 12.0 and later
    status = fseek(fid, 1508, 'bof');
    prescan_offset = fread(fid,1,'integer*4');
    coil_weight_offset = prescan_offset + 332;
    autoshim_offset = prescan_offset + 320;
    status = fseek(fid, autoshim_offset, 'bof');
    [autoshim, count] = fread(fid,3,'integer*2');
    status = fseek(fid, coil_weight_offset, 'bof');
    [receiver_weights, count] = fread(fid,num_receivers,'real*4');  % 128 Max
    status = fseek(fid, prescan_offset + 48, 'bof');
    aps_or_mps = fread(fid,1,'integer*4');  % autoprescan vs manual
    status = fseek(fid, prescan_offset + 4, 'bof');
    mps_r1 = fread(fid,1,'integer*4');  
    status = fseek(fid, prescan_offset + 8, 'bof');
    mps_r2 = fread(fid,1,'integer*4');  
    status = fseek(fid, prescan_offset + 16, 'bof');
    mps_freq = fread(fid,1,'integer*4');  
    status = fseek(fid, prescan_offset + 20, 'bof');
    aps_r1 = fread(fid,1,'integer*4');  
    status = fseek(fid, prescan_offset + 24, 'bof');
    aps_r2 = fread(fid,1,'integer*4');  
    status = fseek(fid, prescan_offset + 32, 'bof');
    aps_freq = fread(fid,1,'integer*4');  
else
    err_msg = sprintf('Invalid Pfile header revision: %f', f_hdr_value )
end  
fclose(fid);

% Print out Autoshim Parameters if they were stored in Pfile header
if (f_hdr_value > 12.0) & (f_hdr_value < 100.0)  % 12.0 and later
    xshim = autoshim(1);
    yshim = autoshim(2);
    zshim = autoshim(3);
    xshim, yshim, zshim
end

% point_size = 2 for 16 bit data, 4 for EDR (32 bit)
point_size

% aps_or_mps = 1 for manual prescan, 2 for auto prescan
if (aps_or_mps == 1)
   mps_r1, mps_r2, mps_freq 
else
   aps_r1, aps_r2, aps_freq 
end

% Receiver weights computed during prescan noise acquisition
receiver_weights

% scale factor used to normalize pixel intensity
rational_scale_factor

if( num_receivers > 1 )
   figure;
   plot(receiver_weights);
   title('Receiver Weights');
end
