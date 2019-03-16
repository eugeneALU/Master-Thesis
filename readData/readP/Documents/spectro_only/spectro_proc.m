% Spectroscopy - Phase correct, water subtract and Fourier Transform
% Marquette University,   Milwaukee, WI  USA
% Copyright 2000, 2001, 2002, 2003, 2004 - All rights reserved.
% Fred Frigo
%
% This function generates MR spectroscopy results from:
% the phase correction vector, pc, (computed by the phascor function)
% the averaged reference data, ref, (computed by the phascor funtion)
% the name (Pfile) of the raw data file containing water suppressed data.
% and the receiver number, channel_num.
%
%
% Dec 28, 2000 - Original
% May 2, 2001  - updates to plot intermediate results
% June 26, 2001 - updates to accept Pfile name as arg
% Oct 15, 2001 - updates to read from Pfile directly, plus new args
% Feb 11, 2002 - modified for multi-channel spectro 
% Nov 2, 2003  - Updates to read 5.X and 11.0 format files
% Mar 8, 2004  - Plot enhancements
%

function results = spectro_proc ( pc, ref, pfilename, channel_num)


i = sqrt(-1);

% flag set to 1 for plots of intermediate results
save_plot = 0;

% Open Pfile to read reference scan data.
fid = fopen(pfilename,'r', 'ieee-be');
if fid == -1
    err_msg = sprintf('Unable to locate Pfile %s', pfilename)
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
elseif ( rdbm_rev_num == 5.0 ) 
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


status = fseek(fid, 0, 'bof');
[hdr_value, count] = fread(fid, 52, 'integer*2');
nex = hdr_value(37);
nframes = hdr_value(38);
da_xres = hdr_value(52);

% Read 'user19' CV  - number of reference frames
status = fseek(fid, 0, 'bof');
[f_hdr_value, count] = fread(fid, 74, 'real*4');
num_ref_frames = f_hdr_value(74);

% Read Pfile to get water supressed signal
ref_offset = 2*da_xres*(num_ref_frames+1)*4;
frame_size = 2*da_xres*4;
channel_size = (nframes + 1)*frame_size;
data_offset = pfile_header_size + (channel_size*(channel_num - 1)) + ref_offset;

status = fseek(fid, data_offset, 'bof');
num_sig_frames = nframes - num_ref_frames;
data_elements = 2*da_xres*num_sig_frames;
[raw_data, count] = fread(fid, data_elements, 'integer*4');
fclose(fid);

% Store the reference frames in the ref_frames array
vector_size = da_xres;
sig_frames=[];
vtmp = [1:vector_size]; 

for j = 1:num_sig_frames
    vector_offset = vector_size*2*(j-1);
    for k = 1:vector_size
        vtmp(k) = raw_data((vector_offset + k*2)-1)+ (raw_data(vector_offset + k*2)*i);
    end
    sig_frames(j,:)=vtmp;
end

vtmp = 0.0;
% Average the reference data frames 
for j = 1:num_sig_frames
   vtmp = vtmp + sig_frames(j,:);
end

% Create averaged water suppressed signal vector.
sig = vtmp / num_sig_frames;

% For debug: store water-suppressed signal to file
recv_string = sprintf('%d', channel_num);
signal_file = strcat( pfilename, '.recv', recv_string,'.raw.dat');
fidref = fopen(signal_file, 'w+b');
for findex=1:da_xres
    fwrite(fidref, real(sig(findex)), 'real*4');
    fwrite(fidref, imag(sig(findex)), 'real*4');
end
fclose(fidref);


x=[1:da_xres];
% Plot input signal
if (save_plot == 1)
   plot_complex('Band-limited MR spectroscopy signal', sig);  % fig20
end

% Phase Correct Water supressed signal (sig) and Water signal (ref)
sig = pc.*sig;
ref = pc.*ref;

% Plot phase corrected signal and ref
if (save_plot == 1)
   plot_complex('Phase-corrected water-supressed data', sig);  %fig 21
   plot_complex('Phase corrected non-water-suppressed reference data', ref); %fig 22
end

% Subtract to 'signal' from 'water' obtain 'pure water'
pure_water = ref - sig;

% Plot pure water signal
if (save_plot == 1)
   plot_complex('Pure water', pure_water); %fig23
end

% Negate every other element (this shifts the water peak to the center)
a_pure_wat = pure_water;
a_sig = sig;
for n = 1:(da_xres/2)
   a_pure_wat(2*n)= -1.0*a_pure_wat(2*n);
   a_sig(2*n) = -1.0*a_sig(2*n);
end

% Apodization Window 
hanning_size = da_xres/1.6;
half_han_size = hanning_size/2;
win=hanning(hanning_size);
apod = linspace(0.0, 0.0, da_xres);
apod(1:half_han_size) = win((half_han_size+1):hanning_size);

% Plot apodization window
if (save_plot == 1)
   plot_2_real('w_1[n]',apod);  % fig24 
end

% Apply apodization window to water and signal vectors
w_pure_wat = apod.*a_pure_wat;
w_sig = apod.*a_sig;

% Fourier Transform the apodized, alternated water and signal vectors
ft_wat = fft( w_pure_wat );
ft_sig = fft( w_sig);

% Plot Fourier Transform of Pure water and Signal
if (save_plot == 1)
   plot_2_real('Fourier transform of pure water, S_w[k]', abs(ft_wat), ...
               'Fourier transform of signal, S_s[k]', abs(ft_sig));  %fig 25
end

% Scale the pure water. 
%    Assume water in signal and reference is the same.
%    Use 'real' coefficients in 16 element band near center
min_xres=(da_xres/2)-(da_xres/128);
max_xres=(da_xres/2)+(da_xres/128);
mag_wat = ft_wat(min_xres:max_xres);
mag_sig = ft_sig(min_xres:max_xres);
mag_wat = abs( real( mag_wat) );
mag_sig = abs( real( mag_sig) );
water_max = max( mag_wat );
sig_max = max( mag_sig );

% Scale the pure water so it can be subtracted off
scale = sig_max / water_max;
pure_water = scale.*pure_water;

% Subtract "scaled" pure water from signal. 
pure_sig = sig - pure_water;

% Plot Pure Signal
if (save_plot == 1)
   plot_complex('Water-subtracted pure signal', pure_sig);  %fig 26
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For debug: save phase-corrected, water-subtracted signal
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
recv_string = sprintf('%d', channel_num);
signal_file = strcat( pfilename, '.recv', recv_string,'.signal.dat');
fidsig = fopen(signal_file, 'w+b');
for findex=1:da_xres
    fwrite(fidsig, real(pure_sig(findex)), 'real*4');
    fwrite(fidsig, imag(pure_sig(findex)), 'real*4');
end
fclose(fidsig);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For debug: save reference signal
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
recv_string = sprintf('%d', channel_num);
signal_file = strcat( pfilename, '.recv', recv_string,'.ref.dat');
fidref = fopen(signal_file, 'w+b');
for findex=1:da_xres
    fwrite(fidref, real(ref(findex)), 'real*4');
    fwrite(fidref, imag(ref(findex)), 'real*4');
end
fclose(fidref);


% Window for the "pure signal" 
% win=hanning(1024);
win=hanning((da_xres*2));
awin = linspace(0.0, 0.0, da_xres);
% awin(1:512) = win(513:1024);
awin(1:da_xres) = win((da_xres+1):(da_xres*2));

if (save_plot == 1)
   plot_2_real('w_2[n]', awin);  % fig 27
end

% Apodize and zero pad the "pure signal" prior to the Fourier transform
zero_pad = 1;
a_pure_sig = linspace( 0.0, 0.0, (da_xres*2)*zero_pad);
a_pure_sig(1:da_xres) = awin.*pure_sig;
nmr_spect = fft( a_pure_sig );
 
if (save_plot == 1)  %fig 28
   plot_complex('Phase-corrected, apodized, water-suppressed signal with residual water removed',a_pure_sig);
end

results = nmr_spect;
