function [alpha,unknow,fgRegion]=genMaskMatrix(fg,choice)
% @author:Tanlang
% @illustration:
    % choice == 1 , green chorma key algorithm 1
    % choice == 2 , green chorma key algorithm 2
    
if choice == 1
    % 参数(fg,a,b) a,b均大于0
    % 增大a值，绿色前景数量增加，反之减少，建议值（0，50）
    % 增大b值，背景区域减少，反正增加，建议值（90，120）
    % 两者根据所处环境调整
        [alpha,unknow,fgRegion] = mask_compulation1(fg,30,100);                                                                     
elseif choice == 2
        [alpha,unknow,fgRegion] = mask_compulation2(fg,1,1.1);
end

%% alpha refine
% 高斯滤波和局部性原理refine alpha，无效
% alpha = alphaRefine(alpha,fg,unknow);

%% get unknow
% se1=strel('square',3);
% unknow = imdilate(unknow,se1);

%% get fgRegion
% se2=strel('square',3);
% fgRegion = imerode(fgRegion,se2);

end