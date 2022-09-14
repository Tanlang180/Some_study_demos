function img_result = Gaussion_edge1(img,unknow,kernel_size,sigma,padding)
% @author:Tanlang
% @illustration:
    % ensure padding > edge_size/2 + kernel_size/2
    % kernel_size must be odd

    height = size(img,1); width = size(img,2);

%     WeightMatrix = fspecial('gaussian',[kernel_size,kernel_size],sigma);
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

    img = imgPadding(img,padding);
    unknow = imgPadding(unknow,padding);
    img_result = img;
    for i = padding: height+padding 
        for j = padding:width+padding 
            if unknow(i,j) > 0.01           
                miniMatrix = img(i-floor(kernel_size/2):i+floor(kernel_size/2),j-floor(kernel_size/2):j+floor(kernel_size/2),:);
                img_result(i,j,:) = sum(sum( miniMatrix.*WeightMatrix3d ));     % Gaussion filter
            else
                continue;
            end
        end
    end
   
    img_result = img_result(padding+1:padding+height,padding+1:padding+width,:);
    
end