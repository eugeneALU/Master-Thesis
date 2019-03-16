function spar=readsparsdat(varargin)

% HELP text
% readsparsdat reads SPAR and SDAT file to a matlab cell spar
%
% IN variabels:
%
% OUT variabels:
%
% Call functions:
% getvalue.m

% Function is under development, latest update 20060323, Olof Dahlqvist
%
% Revised 2008-04.14, Anders Tisell
%



% ------------Initialization----------------
% Select the spar file

addpath /Volumes/marka/Matlab/Library/readData/readVAXD/
addpath /Volumes/marka/Matlab/Library/readData/

if nargin == 0
    wd = pwd;
    %cd /home/mr/olofd/philips/NMRdata/liver_data/
    %cd('~/NMRdata/')
    %cd('/Volumes/data/radiofys/Data/MR')
    cd('/Volumes/data_radiofys/Data/MR/')
    [FileName,PathName] = uigetfile('*.SPAR','Choose a *.spar file');
    cd(wd)
    Sparfile = fullfile(PathName,FileName);
elseif nargin == 1 % A cell aray with filenames including path
    Sparfile = varargin{1};
else
    error('Could not read spar file, wrong number of arguments')
end

s = sprintf('\n');

%-------------Read the SPAR header -----------------------------

    fid = fopen(Sparfile);
    [A,COUNT] = fscanf(fid,'%c');
    fclose(fid);
    spar.Sparfile = Sparfile;
    spar.Software_version = getvalue('equipment_sw_verions',':',s,A,'text');
    spar.Examination_name = getvalue('examination_name',':',s,A,'text');
    spar.Scan_id = getvalue('scan_id',':',s,A,'text');
    spar.Scan_date = getvalue('scan_date',':',s,A,'text');
    spar.Patient_name = getvalue('patient_name',':',s,A,'text');
    spar.Patient_birth_date = getvalue('patient_birth_date',':',s,A,'text');
    spar.Patient_position = getvalue('patient_position',':',s,A,'text');
    spar.Patient_orientation = getvalue('patient_orientation',':',s,A,'text');
    spar.Samples = getvalue('samples',':',s,A,'num');
    spar.Rows = getvalue('rows',':',s,A,'num');
    spar.Synthesizer_frequency = getvalue('synthesizer_frequency',':',s,A,'num');
    spar.Offset_frequency = getvalue('offset_frequency',':',s,A,'num');
    spar.Sample_frequency = getvalue('sample_frequency',':',s,A,'num');
    spar.Echo_nr = getvalue('echo_nr',':',s,A,'num');
    spar.Mix_number = getvalue('mix_number',':',s,A,'num');
    spar.Nucleus = getvalue('nucleus',':',s,A,'text');
    spar.T0_mu1_direction = getvalue('t0_mu1_direction',':',s,A,'num');
    spar.TE = getvalue('echo_time',':',s,A,'num');
    spar.TR = getvalue('repetition_time',':',s,A,'num');
    spar.Averages = getvalue('averages',':',s,A,'num');
    spar.Volume_selection_enable = getvalue('volume_selection_enable',':',s,A,'text');
    spar.Volumes = getvalue('volumes',':',s,A,'num');
    spar.AP_size = getvalue('ap_size',':',s,A,'num');
    spar.LR_size = getvalue('lr_size',':',s,A,'num');
    spar.CC_size = getvalue('cc_size',':',s,A,'num');
    spar.AP_off_center = getvalue('ap_off_center',':',s,A,'num');
    spar.LR_off_center = getvalue('lr_off_center',':',s,A,'num');
    spar.CC_off_center = getvalue('cc_off_center',':',s,A,'num');
    spar.AP_angulation = getvalue('ap_angulation',':',s,A,'num');
    spar.LR_angulation = getvalue('lr_angulation',':',s,A,'num');
    spar.CC_angulation = getvalue('cc_angulation',':',s,A,'num');
    spar.Volume_selection_method = getvalue('volume_selection_method',':',s,A,'num');
    spar.T1_measurement_enable = getvalue('t1_measurement_enable',':',s,A,'text');
    spar.T2_measurement_enable = getvalue('t2_measurement_enable',':',s,A,'text');
    spar.Nr_echo_times = getvalue('Nr_echo_times',':',s,A,'num');
    spar.Time_series_enable = getvalue('time_series_enable',':',s,A,'text');
    spar.Phase_encoding_enable = getvalue('phase_encoding_enable',':',s,A,'text');
    spar.Nr_phase_encoding_profiles = getvalue('nr_phase_encoding_profiles',':',s,A,'num');
    spar.SI_ap_off_center = getvalue('si_ap_off_center',':',s,A,'num');
    spar.SI_lr_off_center = getvalue('si_lr_off_center',':',s,A,'num');
    spar.SI_cc_off_center = getvalue('si_cc_off_center',':',s,A,'num');
    spar.SI_ap_off_angulation = getvalue('si_ap_off_angulation',':',s,A,'num');
    spar.SI_lr_off_angulation = getvalue('si_lr_off_angulation',':',s,A,'num');
    spar.SI_cc_off_angulation = getvalue('si_cc_off_angulation',':',s,A,'num');
    spar.T0_kx_direction = getvalue('t0_kx_direction',':',s,A,'num');
    spar.T0_ky_direction = getvalue('t0_ky_direction',':',s,A,'num');
    spar.Nr_of_phase_encoding_profiles_ky = getvalue('nr_of_phase_encoding_profiles_ky',':',s,A,'num');
    spar.Phase_encoding_direction = getvalue('phase_encoding_direction',':',s,A,'text');
    spar.phase_encoding_fov = getvalue('phase_encoding_fov',':',s,A,'num');
    spar.slice_thickness = getvalue('slice_thickness',':',s,A,'num');
    spar.image_plane_slice_thickness = getvalue('image_plane_slice_thickness',':',s,A,'num');
    spar.slice_distance = getvalue('slice_distance',':',s,A,'num');
    spar.nr_of_slices_for_multislice = getvalue('nr_of_slices_for_multislice',':',s,A,'num');
    spar.Spec_image_in_plane_transf = getvalue('Spec.image in plane transf',':','!',A,'text');

    spar.spec_data_type = getvalue('spec_data_type',':',s,A,'text');
    spar.spec_sample_extension = getvalue('spec_sample_extension',':','!-',A,'text');
    spar.spec_num_col = getvalue('spec_num_col',':',s,A,'num');
    spar.spec_col_lower_val = getvalue('spec_col_lower_val',':',s,A,'num');
    spar.spec_col_upper_val = getvalue('spec_col_upper_val',':',s,A,'num');
    spar.spec_col_extension = getvalue('spec_col_extension',':','!-',A,'text');
    spar.spec_num_row = getvalue('spec_num_row',':',s,A,'num');
    spar.spec_row_lower_val = getvalue('spec_row_lower_val',':',s,A,'num');
    spar.spec_row_upper_val = getvalue('spec_row_upper_val',':',s,A,'num');
    spar.spec_row_extension = getvalue('spec_row_extension',':','!-',A,'text');
    spar.SUN_num_dimensions = getvalue('num_dimensions',':','!-',A,'text');
    spar.SUN_dim1_ext = getvalue('dim1_ext',':',s,A,'text');
    spar.SUN_dim1_pnts = getvalue('dim1_pnts',':',s,A,'num');
    spar.SUN_dim1_low_val = getvalue('dim1_low_val',':',s,A,'num');
    spar.SUN_dim1_step = getvalue('dim1_step',':',s,A,'num');
    spar.SUN_dim1_direction = getvalue('dim1_direction',':',s,A,'text');
    spar.SUN_dim1_t0_point = getvalue('dim1_t0_point',':','!-',A,'num');
    spar.SUN_dim2_ext = getvalue('dim2_ext',':',s,A,'text');
    spar.SUN_dim2_pnts = getvalue('dim2_pnts',':',s,A,'num');
    spar.SUN_dim2_low_val = getvalue('dim2_low_val',':',s,A,'num');
    spar.SUN_dim2_step = getvalue('dim2_step',':',s,A,'num');
    spar.SUN_dim2_direction = getvalue('dim2_direction',':',s,A,'text');
    spar.SUN_dim2_t0_point = getvalue('dim2_t0_point',':','!-',A,'num');
    spar.SUN_dim3_ext = getvalue('dim3_ext',':',s,A,'text');
    spar.SUN_dim3_pnts = getvalue('dim3_pnts',':',s,A,'num');
    spar.SUN_dim3_low_val = getvalue('dim3_low_val',':',s,A,'num');
    spar.SUN_dim3_step = getvalue('dim3_step',':',s,A,'num');
    spar.SUN_dim3_direction = getvalue('dim3_direction',':',s,A,'text');
    spar.SUN_dim3_t0_point = getvalue('dim3_t0_point',':','!-',A,'num');

    spar.addpar_echo_acquisition=getvalue('echo_acquisition',':',s,A,'text');
    spar.addpar_TSI_factor=getvalue('TSI_factor',':',s,A,'num');
    spar.addpar_spectrum_echo_time=getvalue('spectrum_echo_time',':',s,A,'num');
    spar.addpar_spectrum_inversion_time=getvalue('spectrum_inversion_time',':',s,A,'num');
    spar.addpar_image_chemical_shift=getvalue('image_chemical_shift',':',s,A,'num');
    spar.addpar_resp_motion_comp_technique=getvalue('resp_motion_comp_technique',':',s,A,'text');
    spar.addpar_de_coupling=getvalue('de_coupling',':',s,A,'text');
    spar.addpar_equipment_sw_verions=getvalue('equipment_sw_verions',':',';',A,'text');
    spar.addpar_placeholder1=getvalue('placeholder1',':',s,A,'text');
    spar.addpar_placeholder2=getvalue('placeholder2',':',s,A,'text');
    spar.nr_of_slices_for_multislice = getvalue('nr_of_slices_for_multislice',':',s,A,'num');
  
    
    


%-----------Read Data file -----------------------------------------------
%if strcmp('GLNXA64',computer)
      fid = fopen(strrep(Sparfile,'SPAR','SDAT'),'r','ieee-le');
  [A,COUNT] = fread(fid,'float32','native');
  %size(A)
  frewind(fid);
  A = freadVAXG(fid,length(A),'float32');
  %size(A)
  fclose(fid);
  
  
%     
% else
%     
%     fid = fopen(strrep(Sparfile,'SPAR','SDAT'),'r');
%     
%     
%     [A,COUNT] = fread(fid,'float32','vaxg');
%     fclose(fid);
%     
%     
% end


    for k = 1:spar.spec_num_row
        spar.data{k} = reshape(A(1+(k-1)*2*spar.Samples:k*2*spar.Samples),2,spar.Samples);
    end

end