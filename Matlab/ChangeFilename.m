file = dir(['..' filesep 'data xlsx' filesep '*.xlsx']);
length = max(size(file));

for i = 1:length
    oldname = file(i).name;
    newname = erase(oldname, '_Results');
    movefile(oldname, newname);
end