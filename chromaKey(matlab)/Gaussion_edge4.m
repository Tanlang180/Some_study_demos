function img_result = Gaussion_edge4(img,unknow,kernel_size,sigma)
%使用多线程parfor 

    WeightMatrix= zeros(kernel_size,kernel_size);
    for x = 1: kernel_size
        for y = 1:kernel_size  
            WeightMatrix(x, y)=exp(-((x-floor(kernel_size/2)-1)^2+(y-floor(kernel_size/2)-1)^2)/(2*sigma^2))/(2*pi*sigma^2);
        end
    end
    WeightMatrix3d = zeros(kernel_size,kernel_size,3);
    WeightMatrix3d(:,:,1) = WeightMatrix;
    WeightMatrix3d(:,:,2) = WeightMatrix;
    WeightMatrix3d(:,:,3) = WeightMatrix;

    height = size(img,1); width = size(img,2);
    padding = floor(kernel_size/2)+1;
    img = imgPadding(img,padding);
    unknow = imgPadding(unknow,padding);
    
%     img1 = img(1:floor(height/2)+padding,1:floor(width/2)+padding);
%     img2 = img(1:floor(height/2)+padding,floor(width/2)+padding:floor(width/2)+2*padding);
%     img3 = img(floor(height/2)+padding:floor(height/2)+2*padding,1:floor(width/2)+padding);
%     img4 = img(floor(height/2)+padding:floor(height/2)+2*padding,floor(width/2)+padding:floor(width/2)+2*padding);
%     
%     unknow1 = unknow(1:floor(height/2)+padding,1:floor(width/2)+padding);
%     unknow2 = unknow(1:floor(height/2)+padding,floor(width/2)+padding:floor(width/2)+2*padding);
%     unknow3 = unknow(floor(height/2)+padding:floor(height/2)+2*padding,1:floor(width/2)+padding);
%     unknow4 = unknow(floor(height/2)+padding:floor(height/2)+2*padding,floor(width/2)+padding:floor(width/2)+2*padding);
%     
    img_result = img;
%     img_result2 = img2;
%     img_result3 = img3;
%     img_result4 = img4;
    
    parfor i =1+padding : floor(height/2)+padding
        for j = 1+padding : floor(width/2)+padding
            if unknow(i,j) > 0.01
                miniMatrix1 = img(i-floor(kernel_size/2):i+floor(kernel_size/2),j-floor(kernel_size/2):j+floor(kernel_size/2),:);
                sum(sum(miniMatrix1.*WeightMatrix3d ));
            elseif unknow(i,j) > 0.01
                miniMatrix2 = img(i-floor(kernel_size/2):i+floor(kernel_size/2),(2*padding+width-j+1)-floor(kernel_size/2):(2*padding+width-j+1)+floor(kernel_size/2),:);
                sum(sum(miniMatrix2.*WeightMatrix3d ));
            elseif unknow(i,j) > 0.01
                miniMatrix3 = img((2*padding+height-i+1)-floor(kernel_size/2):(2*padding+height-i+1)+floor(kernel_size/2),j-floor(kernel_size/2):j+floor(kernel_size/2),:);
                sum(sum(miniMatrix3.*WeightMatrix3d ));
            elseif unknow(i,j) > 0.01
                miniMatrix4 = img((2*padding+height-i+1)-floor(kernel_size/2):(2*padding+height-i+1)+floor(kernel_size/2),(2*padding+width-j+1)-floor(kernel_size/2):(2*padding+width-j+1)+floor(kernel_size/2),:);
                sum(sum(miniMatrix4.*WeightMatrix3d ));
            else
                continue;
            end
        end
    end
    img_result = img_result(padding+1:padding+height,padding+1:padding+width,:);
end