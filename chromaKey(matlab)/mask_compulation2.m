function [alpha, unknow, fgRegion] = mask_compulation2(fg,a,b)

fgR = fg(:,:,1)./255; % red   channel
fgG = fg(:,:,2)./255; % green channel
fgB = fg(:,:,3)./255; % blue  channel

%% getting th mask
[alpha, unknow, fgRegion] = alpha_get2_1(fgG,fgR,fgB,a,b);
end