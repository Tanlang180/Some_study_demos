function img_result = Gaussion_edge3(img,unknow,kernel_size,sigma)
% Í¼Ïñpatches ¸ßË¹ÂË²¨
% sigma ·½²î

    WeightMatrix= zeros(kernel_size,kernel_size);
    for x = 1: kernel_size
        for y = 1:kernel_size  
            WeightMatrix(x, y)=exp(-((x-floor(kernel_size/2)-1)^2+(y-floor(kernel_size/2)-1)^2)/(2*sigma))/(2*pi*sigma^2);
        end
    end
    WeightMatrix = WeightMatrix ./ sum(sum(WeightMatrix));
    WeightMatrix3d = zeros(kernel_size,kernel_size,3);
    WeightMatrix3d(:,:,1) = WeightMatrix;
    WeightMatrix3d(:,:,2) = WeightMatrix;
    WeightMatrix3d(:,:,3) = WeightMatrix;

    height = size(unknow,1); width = size(unknow,2);
    padding = floor(kernel_size/2);
%     img = imgPadding(img,padding);
    unknow = imgPadding(unknow,padding);
    img_result = img;
    
    for i =1+padding : floor(height/2)+padding
        for j = 1+padding : floor(width/2)+padding
            if unknow(i,j) > 0.01      
                miniMatrix1 = img(i-floor(kernel_size/2):i+floor(kernel_size/2),j-floor(kernel_size/2):j+floor(kernel_size/2),:);
                img_result(i,j,:) = sum(sum(miniMatrix1.*WeightMatrix3d ));
            elseif unknow(i,(2*padding+width-j+1)) > 0.01
                miniMatrix2 = img(i-floor(kernel_size/2):i+floor(kernel_size/2),(2*padding+width-j+1)-floor(kernel_size/2):(2*padding+width-j+1)+floor(kernel_size/2),:);
                img_result(i,(2*padding+width-j+1),:) = sum(sum(miniMatrix2.*WeightMatrix3d ));
            elseif unknow((2*padding+height-i+1),j) > 0.01
                miniMatrix3 = img((2*padding+height-i+1)-floor(kernel_size/2):(2*padding+height-i+1)+floor(kernel_size/2),j-floor(kernel_size/2):j+floor(kernel_size/2),:);
                img_result((2*padding+height-i+1),j,:) = sum(sum(miniMatrix3.*WeightMatrix3d ));
            elseif unknow((2*padding+height-i+1),(2*padding+width-j+1)) > 0.01
                miniMatrix4 = img((2*padding+height-i+1)-floor(kernel_size/2):(2*padding+height-i+1)+floor(kernel_size/2),(2*padding+width-j+1)-floor(kernel_size/2):(2*padding+width-j+1)+floor(kernel_size/2),:);
                img_result((2*padding+height-i+1),(2*padding+width-j+1),:) = sum(sum(miniMatrix4.*WeightMatrix3d ));
            else
                continue;
            end
        end
    end
    img_result = img_result(padding+1:padding+height,padding+1:padding+width,:);
end