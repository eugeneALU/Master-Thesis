% read_mr_image -  Read GEHC MR image 
% Marquette University
% Copyright 2009 - All rights reserved.
% Fred J. Frigo
%
% Sept 23, 2009   - Original
%
%                 The 'list_select_ex' utility can be used to extract
%                 this image from the image data base.
%  
% Note: the file "gems-dicom-dict.txt" (MR) can be found in this directory:
%       /export/home/sdc/app-defaults/dicom/gems-dicom-dict.txt
%

function  read_mr_image( dfile1, dict_file )

   if(nargin == 0)
       [fname, pname] = uigetfile('*.*', 'Select MR image');
       dfile1 = strcat(pname, fname);
 
       % Assume DICOM dictionary file is in SAME directory for now
       dict_file = strcat(pname, 'gems-dicom-dict.txt');   
   end
 
   % Set dictionary to the new DICOM DICTIONARY
   dicomdict('set', dict_file); 
   dictionary = dicomdict('get');
   
   % Get DICOM info from image.
   info1 = dicominfo(dfile1);
   exam = info1.StudyID;
   series = info1.SeriesNumber;
   image_number1 = info1.InstanceNumber;
   
   % Read image data from input image.
   I1 = dicomread(info1);
   
   % display the image
   figure;
   colormap('gray');
   imagesc(I1);
   plot_string1 = sprintf('Exam %s, series %d, image %d', exam, series, image_number1);
   title(plot_string1);
end       
 