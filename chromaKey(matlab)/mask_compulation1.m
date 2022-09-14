function [alpha,unknow,fgRegion] = mask_compulation1(fg,a,b)

    fgR = fg(:,:,1); % red   channel
    fgG = fg(:,:,2); % green channel
    fgB = fg(:,:,3); % blue  channel

    %% caculating distance
    dis = 2 * fgG - fgR - fgB;

    %% getting th mask
    [alpha, unknow, fgRegion] = alpha_get1_1(dis,fgG,fgR,fgB,a,b);
    
end