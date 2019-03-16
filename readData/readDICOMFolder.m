function tempData = readDICOMFolder(folder)

files = dir(folder);
files = files(3:end);

nFiles = length(files);

for i = 1:nFiles
    tmpInfo = dicominfo(fullfile(folder,files(i).name));
    tmpImage = dicomread(tmpInfo);
    tmpData.image(:,:,i) = tmpImage;
    tmpData.imagePosition(i,:);
    tmpData.TE(i,1) = tmpInfo.EchoTime;
    tmpData.rescaleIntercept(i,1) = tmpInfo.RescaleIntercept;
    tmpData.rescaleSlope(i,1) = tmpInfo.RescaleSlope;
end
%{
[~,sliceOrder] = sort(tmpData.imagePosition(:,3));
[~,TEOrder] = sort(tmpData.TE(sliceOrder));
newOrder = sliceOrder(TEOrder);


data.image = tmpData.image(:,:,newOrder);
data.imagePostion = tmpData.imagePosition(newOrder,:);
data.TE = tmpData.TE(sliceOrder(newOrder));
data.rescaleIntercept = tmpData.rescaleIntercept(newOrder);
data.rescaleSlope = tmpData.rescaleSlope(newOrder);
%}
