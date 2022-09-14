function img_result = KnnKernelFilter3(img,unknow,fgRegion,kernel_size)
    % @author:Tanlang
    % @illustration: 待调试，未成功
        % function: find some foreground pixel values near the edge pixel as the surrogate value
                  % kernel_size is the size of found region, the best choice: {9,7}
                  % k is the num of neigborhood
        % ensure padding > kernel_size/2
        % ensure kernel_size is odd
        % fgRegion is absolute foreground

    %% set variable
    height = size(img,1); width = size(img,2);d=size(img,3);
    padding = floor(kernel_size/2)+1;
    
    img = imgPadding(img,padding);
    unknow = imgPadding(unknow,padding);
    fgRegion = imgPadding(fgRegion,padding);
    img_result = img;
    k = kernel_size+1;
    %% KNN kernel filter
    for i =1+padding : floor(height/2)+padding
        for j = 1+padding : floor(width/2)+padding
            if unknow(i,j) > 0.01
                minimatrix = img(i-floor(kernel_size/2):i+floor(kernel_size/2),j-floor(kernel_size/2):j+floor(kernel_size/2),:);         
                fgMatrix = fgRegion(i-floor(kernel_size/2):i+floor(kernel_size/2),j-floor(kernel_size/2):j+floor(kernel_size/2),:);
                % 矩形框内前景像素数量小于k,则跳过
                if length(find(fgMatrix~=0)) < k
                    continue;
                end
                img_result(i,j,:) = knnReplace(img(i,j,:),minimatrix,fgMatrix,kernel_size);
                
            elseif unknow(i,(2*padding+width-j+1)) > 0.01
                minimatrix = img(i-floor(kernel_size/2):i+floor(kernel_size/2),(2*padding+width-j+1)-floor(kernel_size/2):(2*padding+width-j+1)+floor(kernel_size/2),:);         
                fgMatrix = fgRegion(i-floor(kernel_size/2):i+floor(kernel_size/2),(2*padding+width-j+1)-floor(kernel_size/2):(2*padding+width-j+1)+floor(kernel_size/2),:);
                % 矩形框内前景像素数量小于k,则跳过
                if  length(find(fgMatrix~=0)) < k
                    continue;
                end
                img_result(i,(2*padding+width-j+1),:) = knnReplace(img(i,(2*padding+width-j+1),:),minimatrix,fgMatrix,kernel_size);
                
            elseif unknow((2*padding+height-i+1),j) > 0.01
                minimatrix = img((2*padding+height-i+1)-floor(kernel_size/2):(2*padding+height-i+1)+floor(kernel_size/2),j-floor(kernel_size/2):j+floor(kernel_size/2),:);         
                fgMatrix = fgRegion((2*padding+height-i+1)-floor(kernel_size/2):(2*padding+height-i+1)+floor(kernel_size/2),j-floor(kernel_size/2):j+floor(kernel_size/2),:);
                % 矩形框内前景像素数量小于k,则跳过
                if  length(find(fgMatrix~=0)) < k
                    continue;
                end
                img_result((2*padding+height-i+1),j,:) = knnReplace(img((2*padding+height-i+1),j,:),minimatrix,fgMatrix,kernel_size);
                
            elseif unknow((2*padding+height-i+1),(2*padding+width-j+1)) > 0.01
                minimatrix = img((2*padding+height-i+1)-floor(kernel_size/2):(2*padding+height-i+1)+floor(kernel_size/2),(2*padding+width-j+1)-floor(kernel_size/2):(2*padding+width-j+1)+floor(kernel_size/2),:);         
                fgMatrix = fgRegion((2*padding+height-i+1)-floor(kernel_size/2):(2*padding+height-i+1)+floor(kernel_size/2),(2*padding+width-j+1)-floor(kernel_size/2):(2*padding+width-j+1)+floor(kernel_size/2),:);
                % 矩形框内前景像素数量小于k,则跳过
                if  length(find(fgMatrix~=0)) < k
                    continue;
                end
                img_result((2*padding+height-i+1),(2*padding+width-j+1),:) = knnReplace(img((2*padding+height-i+1),(2*padding+width-j+1),:),minimatrix,fgMatrix,kernel_size);

            else
                continue;
            end
        end
    end
    img_result = img_result(padding+1:padding+height,padding+1:padding+width,:);
end