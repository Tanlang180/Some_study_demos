function [alpha,unknow_refine,fgRegion] = alpha_get1_1(dis,fgG,fgR,fgB,a,b)
     
    flag_space = double((fgG > fgR) & (fgG > fgB));
    flag_space_fg = double(dis < a) .* flag_space;


    unknow = double((dis >= a) & (dis < b)) .* flag_space;
    fgRegion = flag_space_fg + 1 - flag_space;
    
    alpha_unknow = unknow .* (255-( 255 / (b-a) .* (dis-a)));
    
    unknow_refine = (alpha_unknow > (255 * 0.2)) & (alpha_unknow < 255);
    
    alpha = (255 .* fgRegion + alpha_unknow) ./ 255;
    
end