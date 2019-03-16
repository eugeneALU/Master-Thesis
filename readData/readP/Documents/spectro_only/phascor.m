% Spectroscopy - Generate Phase correction coefficients
% Marquette University,   Milwaukee, WI  USA
% Copyright 2001, 2002, 2003, 2004 - All rights reserved.
% Fred J. Frigo
%
% Feb 15, 2001  - Original
% June 26, 2001 - Accept Pfile name as argument
% Sept 3, 2001  - Use Equiripple FIR filter for linear phase correction
% Oct 15, 2001  - Open Pfile directly instead of intermediate file.
%                 add ability to return an array of vectors.
% Dec 8, 2001   - Added DeBoor spline smoothing.
% Feb 11, 2002  - Added multi-channel support
% Jul 30, 2002  - Return Max reference value for multi-channel scaling
% Jun 16, 2003  - Added support for Pfile format for MGD2 / 11.0
% Mar 5, 2004   - Plot enhancements
% Oct 4, 2005   - Pfile format 14.0
%                
%

function [pcor_vector, ref_vector, ref_scale] =  phascor( pfilename, channel_num )
i = sqrt(-1);

% set flag to 1 to plot intermediate results
save_plot = 0;

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
    bandwidth_index = 9839;
elseif ( rdbm_rev_num == 8.0 )
    pfile_header_size = 60464;  % Cardiac / MGD
    bandwidth_index = 14959;
elseif ( rdbm_rev_num == 5.0 ) 
    pfile_header_size = 39940;  % Signa 5.5
    bandwidth_index = 9839;    % ??
else
    % In 11.0 (ME2) the header and data are stored as little-endian
    fclose(fid);
    fid = fopen(pfilename,'r', 'ieee-le');
    status = fseek(fid, 0, 'bof');
    [f_hdr_value, count] = fread(fid, 1, 'real*4');
    if (f_hdr_value == 9.0)  % 11.0 product release
        pfile_header_size= 61464;  
    elseif (f_hdr_value == 11.0)  % 12.0 product release
        pfile_header_size= 66072;
        bandwidth_index = 14959;  %?? may not be correct
    elseif (f_hdr_value > 11.0) & (f_hdr_value < 100.0)  % 14.0 and later
        status = fseek(fid, 1468, 'bof');
        pfile_header_size = fread(fid,1,'integer*4'); 
    else
        err_msg = sprintf('Invalid Pfile header revision: %f', f_hdr_value )
        return;
    end
end  


status = fseek(fid, 0, 'bof');
[hdr_value, count] = fread(fid, 52, 'integer*2');
nex = hdr_value(37);
nframes = hdr_value(38);
da_xres = hdr_value(52);

% Read 'user19' CV  - number of reference frames
status = fseek(fid, 0, 'bof');
[f_hdr_value, count] = fread(fid, 74, 'real*4');
num_ref_frames = f_hdr_value(74);

% Read the reference frames of data
frame_size = 2*da_xres*4;
baseline_size = frame_size;
channel_size = (nframes + 1)*frame_size;
data_offset = pfile_header_size + (channel_size*(channel_num - 1)) + baseline_size;
status = fseek(fid, data_offset, 'bof');

ref_data_elements = 2*da_xres*num_ref_frames;
[raw_data, count] = fread(fid, ref_data_elements, 'integer*4');
fclose(fid);

% Store the reference frames in the ref_frames array
vector_size = da_xres;
ref_frames=[];
vtmp = [1:vector_size]; 

for j = 1:num_ref_frames
    vector_offset = vector_size*2*(j-1);
    for k = 1:vector_size
        vtmp(k) = raw_data((vector_offset + k*2)-1)+ (raw_data(vector_offset + k*2)*i);
    end
    ref_frames(j,:)=vtmp;
end

ref_size = da_xres;
vtmp = 0.0;
% Average the reference data frames 
for j = 1:num_ref_frames
   vtmp = vtmp + ref_frames(j,:);
end
ref = vtmp / num_ref_frames;

% return the averaged reference vector
ref_vector = ref;

% Plot Input Reference data
if (save_plot == 1)
   plot_complex( 'Averaged reference data', ref); % fig 8
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% normalize ref data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ref_scale = max( abs(ref) );
ref_norm = ref / ref_scale;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DC mixing
%   Find the largest frequency component
%   Create a sinusoid of same frequency and opposite phase?
%   Multiply by the sinusoid to cancel out this LARGE freq?
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Take FFT of Average Ref Scan data
ref_ft=fft(ref_norm);

% Find Max of FFT'd data
%   We dont really want to get the last half of points though.
%   Perhaps we can multiply by an alteration vector to look at 
%   just the center points?
%
[refmax, index] = max(ref_ft(1:ref_size-8));
max_index = sprintf('Max freq weight in ref scan frames is %d', index);

% Generate ramp vector, muliply by index of max value
dc = linspace(0.0, (-2.0*pi) , ref_size);
dc = dc.*index;

% Create sinusoid with pure frequency
cos_dc = cos(dc);
sin_dc = sin(dc);

% DC mixing
corr = cos_dc + sin_dc*i;
ref_raw = ref_norm.*corr;

% Plot corrected Reference data,  and phase correction vector
if (save_plot == 1)
    plot_complex('Phase correction vector after DC mixing', corr); % fig 9
    plot_complex('Reference data after DC mixing', ref_raw); % fig 10
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Zero phasing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
zeroterm =  ref_raw(1);
% Complex conjugate of the first point in the DC corrected ref frame
zeroterm = real(zeroterm) - imag(zeroterm)*i;

% The phase angle is now zero for the first point in the ref frame
ref_raw = ref_raw * zeroterm;
corr = corr * zeroterm;

% Plot corrected Reference data,  and phase correction vector
if (save_plot == 1)
    plot_complex('Phase correction vector after zero phase adjustment', corr); % fig 12
    plot_complex('Reference data after zero phase adjustment', ref_raw); % fig 11
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Linear phase correction factor
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Calculate phase of ref vector
ref_ang = angle(ref_raw);

% Unwrap Phase of reference frame
unwr_ref = unwrap( ref_ang);

% Plot phase of reference data, and unwrapped phase
if (save_plot == 1)
    plot_2_real('Phase \phi_z_p[n] (radians)', ref_ang, 'Unwrapped phase \phi_z_p[n] (radians)', unwr_ref); % fig 13
end

% Add up how many periods are present in the ref frame
pscale = unwr_ref(ref_size);


% Generate linear phase vector, muliply by unwrapped phase
ramp = linspace(0.0, -1 , ref_size);
lin_phas = pscale.*ramp;

cos_linp = cos( lin_phas);
sin_linp = sin( lin_phas);

lp_corr = cos_linp + (i*sin_linp);

% Apply linear phase vector to reference frame and to phase corr vector
ref_raw = lp_corr.*ref_raw;
corr = lp_corr.*corr;

% Plot corrected Reference data,  and phase correction vector
if (save_plot == 1)
    plot_complex('Linear phase correction vector', lp_corr); % fig 14
    plot_complex('Phase correction vector after linear phase correction', corr); % fig 16
    plot_complex('Reference data after linear phase correction', ref_raw); % fig 15
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Smooth the phase of the Ref Frame
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ref_phas = angle( ref_raw );
uref_phas = unwrap(ref_phas);

% compute e**( -0.25*ln(abs(ref_raw))
mag_raw = abs(ref_raw);
ln_raw = -0.25*log(mag_raw);
dy = exp(ln_raw);

smooth_factor = 0.9999;

% Spline smoothing (DeBoor)
filt_phas = smooth_spline( uref_phas, dy, ref_size, smooth_factor);

% Plot unwrapped phase of phase corrected reference data, and smoothed phase
if (save_plot == 1)
    plot_2_real('Unwrapped phase \phi_lp[n] (radians)', uref_phas, 'Unwrapped phase \phi_s[n] (radians)', filt_phas); % fig 17
end

% Generate sinusoidal waveform based on smoothed phase
filt_phas = -1.0*filt_phas;
cos_fphs = cos( filt_phas);
sin_fphs = sin( filt_phas);
fphs = cos_fphs + i.*sin_fphs;

% Final step.. Multiply by corr vector and by ref vector
ref_raw = ref_raw.*fphs;
corr = corr.*fphs;

% Plot corrected Reference data,  and phase correction vector
if (save_plot == 1)
    plot_complex('Phase correction vector', corr); % fig 19
    plot_complex('Reference data after phase correction', ref_raw); % fig 18
end

% return the phase correction vector
pcor_vector = corr;


