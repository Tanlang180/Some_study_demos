function img_result = Gaussion_edge2(img,unknow,kernel_size,sigma,padding)
%垂直和水平高斯核分开滤波
    

    height = size(img,1); width = size(img,2);
    img = imgPadding(img,padding);
    unknow = imgPadding(unknow,padding);
    img_result = img;
    WeightMatrix_hor = fspecial('gaussian',[kernel_size,1],sigma);
    WeightMatrix_ver = fspecial('gaussian',[1,kernel_size],sigma);
    
    WeightMatrix_ver3d = zeros(1,kernel_size,3);
    WeightMatrix_hor3d = zeros(kernel_size,1,3);
    for i = 1 :3
        WeightMatrix_ver3d(:,:,i) = WeightMatrix_ver;
        WeightMatrix_hor3d(:,:,i) = WeightMatrix_hor;
    end

    for i = padding: height+padding 
        for j = padding:width+padding 
            if unknow(i,j) > 0.01        
                miniMatrix_hor = img(i-floor(kernel_size/2):i+floor(kernel_size/2),j,:);
%                 miniMatrix_ver = img(i,j-floor(kernel_size/2):j+floor(kernel_size/2),:);
                img_result(i,j,:) = sum(sum(miniMatrix_hor.*WeightMatrix_hor3d));
%                 img_result(i,j,:) = sum(sum(miniMatrix_ver.*WeightMatrix_ver3d));
            else
                continue;
            end
        end
    end
    
    for i = padding: height+padding 
        for j = padding:width+padding 
            if unknow(i,j) > 0.01    
%                 miniMatrix_hor = img(i-floor(kernel_size/2):i+floor(kernel_size/2),j,:);
                miniMatrix_ver = img(i,j-floor(kernel_size/2):j+floor(kernel_size/2),:);
%                 img_result(i,j,:) = sum(sum(miniMatrix_hor.*WeightMatrix_hor3d));
                img_result(i,j,:) = sum(sum(miniMatrix_ver.*WeightMatrix_ver3d));
            else
                continue;
            end
        end
    end
    
    img_result = img_result(padding+1:padding+height,padding+1:padding+width,:);
    
end