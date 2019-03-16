function par=readparrec(varargin)
%tic
% HELP text
% This function reads PAR and REC files exported in one batch from
% the philips scanner for further explanations of the parameter
% definitions (directions, units, etc) read the .PAR file
%
% Output is a cell array called Par, each cell element contains a
% struct with self explaining fields. The field names corresponds to
% the naming used in the *.PAR file. The image data is stored in the
% field "image"
%
% IN variabels:
%
% OUT variabels:
%
% Output is a cell array called Par, each cell element contains a
% struct with self explaining fields. The field names corresponds to
% the naming used in the *.PAR file. The image data is stored in the
% field "image"
% use par{i}.typ to seperated different image volumes
% typ = 0, modulus images
% typ = 1, real images
% typ = 2, imaginary images
% typ = 3, phase images
%
%
% Call functions:
% getvalue.m


% Function developed 200X-XX-XX, by Olof Dahlqvist
% Revised 2008-04.14, Anders Tisell
%

% Add path to getvalue
%addpath /home/mr/marka/Matlab/Library/readData
addpath /Volumes/marka/Matlab/Library/readData

% ------------Initialization----------------
if nargin == 0
    wd=pwd;
    
    %cd('~/Temporary/NMRdata/')
    [FileName,PathName] = uigetfile('*.PAR','Choose a *.par file');
    cd(wd)
    Parfile = fullfile(PathName,FileName);
    Recfile = fullfile(PathName,[FileName(1:end-3) 'REC']);
elseif nargin == 1
    Parfile = varargin{1};
    Recfile = [varargin{1}(1:end-3) 'REC'];
else
    error('Wrong number of arguments')
end

s = sprintf('\n');
% ------------------------------------------


% ------------Read nformation from the PAR header ------------------------------
fid = fopen(Parfile); [A,COUNT] = fscanf(fid,'%c'); fclose(fid);

par.Version = getvalue('image export tool','l','#',A,'text');
par.Dataset_name = getvalue('Dataset name',':','#',A,'text');
par.Patient_name = getvalue('Patient name',':','. ',A,'text');
par.Examination_name = getvalue('Examination name',':','.  ',A,'text');
par.Protocol_name = getvalue('Protocol name',':','. ',A,'text');
par.Examination_time = getvalue('Examination date/time',':','. ',A,'text');
par.Series_Type = getvalue('Series Type',':','. ',A,'text');
par.Acquisition_nr = getvalue('Acquisition nr',':','. ',A,'num');
par.Reconstruction_nr = getvalue('Reconstruction nr',':','. ',A,'num');
par.Scan_Duration = getvalue('Scan Duration [sec]',':','. ',A,'num');
par.Max_number_of_cardiac_phases = getvalue('Max. number of cardiac phases',':','. ',A,'num');
par.Max_number_of_echoes = getvalue('Max. number of echoes',':','. ',A,'num');
par.Max_number_of_slices = getvalue('Max. number of slices/locations',':','. ',A,'num');
par.Max_number_of_dynamics = getvalue('Max. number of dynamics',':','. ',A,'num');
par.Max_number_of_mixes = getvalue('Max. number of mixes',':','. ',A,'num');

par.Patient_position = getvalue('Patient position',':','. ',A,'text');
par.Preparation_direction = getvalue('Preparation direction',':','. ',A,'text');
par.Technique = getvalue('Technique',':','. ',A,'text');
par.Scan_resolution = getvalue('Scan resolution',':','. ',A,'num');
par.Scan_mode = getvalue('Scan mode',':','. ',A,'text');
par.TR = getvalue('Repetition time [ms]',':','. ',A,'num');
par.FOV = getvalue('FOV (ap,fh,rl)',':','. ',A,'num');
par.Water_Fat_shift = getvalue('Water Fat shift [pixels]',':','. ',A,'num');
par.Angulation_midslice = getvalue('Angulation midslice(ap,fh,rl)[degr]',':','. ',A,'num');
par.Off_Centre_midslice = getvalue('Off Centre midslice(ap,fh,rl) [mm]',':','. ',A,'num');
par.Flow_compensation = getvalue('Flow compensation <0=no 1=yes> ?',':','. ',A,'num');
par.Presaturation = getvalue('Presaturation     <0=no 1=yes> ?',':','. ',A,'num');
par.Phase_encoding_velocity = getvalue('Phase encoding velocity [cm/sec]',':','. ',A,'num');
par.MTC = getvalue('MTC               <0=no 1=yes> ?',':','. ',A,'num');
par.SPIR = getvalue('SPIR              <0=no 1=yes> ?',':','. ',A,'num');
par.EPI_factor = getvalue('EPI factor        <0,1=no EPI>',':','. ',A,'num');
par.Dynamic_scan = getvalue('Dynamic scan      <0=no 1=yes> ?',':','. ',A,'num');
par.Diffusion = getvalue('Diffusion         <0=no 1=yes> ?',':','. ',A,'num');
par.Diffusion_echo_time = getvalue('Diffusion echo time [ms]',':',s,A,'num');

if strcmp(par.Version,'V4')
    image_par = getvalue('turbo delay','y','#',A,'num');
    if ~isnumeric(image_par)
        image_par = getvalue(sprintf('turbo\tdelay'),'y','#',A,'num');
    end
    %disp('V4')
elseif strcmp(par.Version,'V4.1')
    image_par = getvalue(sprintf('is\tdiffusion'),'n','#',A,'num');
    if ~isnumeric(image_par)
        image_par = getvalue('is diffusion','n','#',A,'num');
    end
    %disp('V4.1')
elseif strcmp(par.Version,'V4.2')
    image_par = getvalue(sprintf('L.ty'),'y','#',A,'num');

    if ~isnumeric(image_par)
        image_par = getvalue('L.ty','n','#',A,'num');
    end
    %disp('V4.2')
end

par.Unique_image_type_mr = unique(image_par(:,5));
par.Unique_scanning_sequence = unique(image_par(:,6));
par.image_par=image_par;
par.slice_number = image_par(:,1);
par.echo_number = image_par(:,2);
par.dynamic_scan_number = image_par(:,3);
par.cardiac_phase_number = image_par(:,4);
par.image_type_mr = image_par(:,5);
par.scanning_sequence = image_par(:,6);
par.index_in_REC_file = image_par(:,7);
par.image_pixel_size = image_par(:,8);
par.scan_percentage = image_par(:,9);
par.recon_resolution = image_par(:,10:11);
par.rescale_intercept = image_par(:,12);
par.rescale_slope = image_par(:,13);
par.scale_slope = image_par(:,14);
par.window_center = image_par(:,15);
par.window_width = image_par(:,16);
par.image_angulation = image_par(:,17:19);
par.image_offcentre = image_par(:,20:22);
par.slice_thickness = image_par(:,23);
par.slice_gap = image_par(:,24);
par.image_display_orientation = image_par(:,25);
par.slice_orientation = image_par(:,26);
par.fmri_status_indication = image_par(:,27);
par.image_type_ed_es = image_par(:,28);
par.pixel_spacing = image_par(:,29:30);
par.echo_time = image_par(:,31);
par.dyn_scan_begin_time = image_par(:,32);
par.trigger_time = image_par(:,33);
par.diffusion_b_factor = image_par(:,34);
par.number_of_averages = image_par(:,35);
par.image_flip_angle = image_par(:,36);
par.cardiac_frequency = image_par(:,37);
par.minimum_RRinterval = image_par(:,38);
par.maximum_RRinterval = image_par(:,39);
par.TURBO_factor = image_par(:,40);
par.Inversion_delay = image_par(:,41);

% Read parameters added in version 4.1
if strcmp(par.Version,'V4.1') || strcmp(par.Version,'V4.2')
    par.Diffusion_b = image_par(:,42);
    par.Gradient_orientation = image_par(:,43);
    par.Contrast_type = image_par(:,44);
    par.Diffusion_anisotropy = image_par(:,45);
    par.Diffusion_x = image_par(:,46);
    par.Diffusion_y = image_par(:,47);
    par.Diffusion_z = image_par(:,48);
end
% Read parameters added in version 4.2

if strcmp(par.Version,'V4.2')
    par.label_type_ASL = image_par(:,49);
end


%-------------- Read image data from the *.REC file ----------------------

nRows = double(image_par(1,10));
nCols = double(image_par(1,11));
nSlice = size(image_par,1);
fid = fopen(Recfile);
unscaled_image = reshape(fread(fid,nRows*nCols*size(image_par,1),'uint16=>double'),nRows,nCols,size(image_par,1));
fclose(fid);
par.image=zeros(nRows,nCols,nSlice);
%toc
%disp('reScale images')
%tic

for k=1:nSlice
    par.image(:,:,k) = ...
        (unscaled_image(:,:,k) .* par.rescale_slope(k) + par.rescale_intercept(k))./ ...
        (par.rescale_slope(k).* par.scale_slope(k));
end
%toc
