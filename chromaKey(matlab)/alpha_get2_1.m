function [alpha,unknow,fgRegion] = alpha_get2_1(fgG,fgR,fgB,a1,a2)

    height = size(fgG,1);
    width = size(fgG,2);
    alpha = zeros(height,width);
    unknow = zeros(height,width);
    fgRegion =zeros(height,width);
    
    for i = 1:height
        for j = 1:width
            temp = 1 - a1*(4.5 * fgG(i,j) - a2*(3 * fgB(i,j)+ 1.5 * fgR(i,j)));
            if temp > 1
                alpha(i,j) = 1;
                fgRegion(i,j) = 1;
            elseif temp < 0
                alpha(i,j) = 0;
            else
                alpha(i,j) = temp;
                unknow(i,j) = 1;
            end
        end
    end
    
end