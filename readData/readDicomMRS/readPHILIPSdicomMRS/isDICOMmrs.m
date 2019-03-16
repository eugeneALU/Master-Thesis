function result = isDICOMmrs( dicomFile )
%isDICOMmrs  True if argument is a dicom file containging MRS data.
%   ISDIR(DIR) returns a 1 if dicomFile is a dicom file containing MRS data
%   and 0 otherwise.

try
    info = dicominfo(dicomFile);
    try
        info.SpectroscopyData;
        result = 1;
    catch
        result = 0;
        warning([dicomFile ' is not a MRS dicom file'])
    end
catch
    warning([dicomFile ' is not a dicom file'])
    result =0;
end
end

