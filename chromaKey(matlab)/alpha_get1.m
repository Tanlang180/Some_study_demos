function [alpha,unknow,fgRegion] = alpha_get1(dis,fgG,fgR,fgB,a,b)
    height = size(dis,1);
    width = size(dis,2);
    alpha = zeros(height,width);
    unknow = zeros(height,width);
    fgRegion =zeros(height,width);
    for i = 1:height
        for j = 1:width
            if (fgG(i,j)>fgR(i,j) && fgG(i,j)>fgB(i,j))
                temp = alphafunc(dis(i,j),a,b);
                alpha(i,j) = temp;
                if temp < 255 && temp > 255*0.2
                    unknow(i,j) =255;
                end
                if temp == 255
                    fgRegion(i,j) = 255;
                end   
            else
                alpha(i,j) = 255;
                fgRegion(i,j) = 255;
            end
        end       
    end
    alpha = alpha / 255;
    unknow = unknow / 255;
    fgRegion = fgRegion /255;
 end