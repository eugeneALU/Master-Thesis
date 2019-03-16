function newImage = readVtkMK(fileName)

fid = fopen(fileName);
if fid < 0
    error(['Could not open stl file: ' fileName]);
end

line = fgetl(fid);
disp(line)
line = fgetl(fid);
disp(line)
line = fgetl(fid);
disp(line)
line = fgetl(fid);
disp(line)
line = fgetl(fid);
disp(line)
imSize(1) = str2double(line(12:14));
imSize(2) = str2double(line(16:18));
imSize(3) = str2double(line(20:end));
line = fgetl(fid);
disp(line)
line = fgetl(fid);
disp(line)
line = fgetl(fid);
disp(line)
line = fgetl(fid);
disp(line)
line = fgetl(fid);
disp(line)

raw = fread(fid);
fclose(fid);
raw = reshape(raw,imSize(1),imSize(2),imSize(3));

newImage = ones(imSize);
newImage(raw<122) = 0;



















