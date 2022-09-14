function [fg] = greenDecrease2(fg,bg1,bg2,unknow,alpha,choice)
%% 沾色和合成背景都使用矩阵计算
% 效果：与分为三步相比，速度更快
    height = size(fg,1); width = size(fg,2);
    fgR = fg(:,:,1); % red   channel
    fgG = fg(:,:,2); % green channel
    fgB = fg(:,:,3); % blue  channel
    
    alpha3d = zeros(height,width,3);
    for i = 1:3
        alpha3d(:,:,i) = alpha;
    end
    
    %% descrease green channel value
    switch(choice)
        case 1,
            flag = double(fgG > ((fgR+fgB)./2)) .* unknow;
            fg(:,:,2) = flag .* ((fgR+fgB)./2) + (1-flag) .* fgG;
        case 2,
            flag = double(fgG > ((fgR+2*fgB)/3)) .* unknow;
            fg(:,:,2) = flag .* ((fgR+2*fgB)/3) + (1-flag) .* fgG;
        case 3,
            flag = double(fgG > ((2*fgR+fgB)/3)) .* unknow;
            fg(:,:,2) = flag .* ((2*fgR+fgB)/3) + (1-flag) .* fgG;
        case 4,
            flag = double(fgG > fgR) .* unknow;
            fg(:,:,2) = flag .* fgR + (1-flag) .* fgG;
        case 5,
            flag = double(fgG > fgB) .* unknow;
            fg(:,:,2) = flag .* fgB + (1-flag) .* fgG;
        case 6,
            flag = double(fgG > ((fgR+fgB)./2)) .* unknow;
            val = max(max(fgG,fgB),fgR) .* unknow;
            val(:,:,1) = val;
            val(:,:,2) = val;
            val(:,:,3) = val;
            fg(:,:,:) = val .* flag + (1-flag) .* fg(:,:,:);
    end
    %% blending image
    fg = fg .* alpha3d + (1-alpha3d) .* bg1;
    fg = fg .* alpha3d + (1-alpha3d) .* bg2;

end