function img_result = KnnKernelFilter2(img,unknow,fgRegion,kernel_size)
    % @author:Tanlang
    % @illustration:
        % function: find some foreground pixel values near the edge pixel as the surrogate value
                  % kernel_size is the size of found region, the best choice: {9,7}
                  % k is the num of neigborhood
        % ensure padding > kernel_size/2
        % ensure kernel_size is odd
        % fgRegion is absolute foreground

    %% set variable
    height = size(img,1); width = size(img,2);d=size(img,3);
    padding = floor(kernel_size/2);
    k = kernel_size+1;
    img = imgPadding(img,padding);
    unknow = imgPadding(unknow,padding);
    fgRegion = imgPadding(fgRegion,padding);
    img_result = img;
    indexWeights = zeros(kernel_size,kernel_size); 
    %% KNN kernel filter
    for i = padding+1: height+padding
        for j = padding+1:width+padding
            if unknow(i,j) > 0.01
                    minimatrix = img(i-floor(kernel_size/2):i+floor(kernel_size/2),j-floor(kernel_size/2):j+floor(kernel_size/2),:);         
                    fgMatrix = fgRegion(i-floor(kernel_size/2):i+floor(kernel_size/2),j-floor(kernel_size/2):j+floor(kernel_size/2),:);

                    % 矩形框内前景像素数量小于k,则跳过
                    if sum(sum(fgMatrix~=0)) < k
                        continue;
                    end
                    
                    fgMatrix3d(:,:,1) = fgMatrix;
                    fgMatrix3d(:,:,2) = fgMatrix;
                    fgMatrix3d(:,:,3) = fgMatrix;

                    center_feature = repmat(img(i,j,:),[kernel_size,kernel_size,1]);
                    sumOfabs = sum(abs(minimatrix-center_feature).* fgMatrix3d,3);
                    constant = max(max(sumOfabs));
                    weightMatrix = fgMatrix - sumOfabs ./ constant;
                    weightVector = reshape(weightMatrix,[1,kernel_size.^2]);
                    
                    [~,index] = sort(weightVector,2,'descend');  % 邻近值降序排列，得到foreWeight矩阵，和邻近像素的索引index，（即邻近像素位置）
                    index = index(1:k);  % 取得前k个邻近值
                    indexWeights(index) = 1;
                    weightMatrix = indexWeights .* weightMatrix;
                    weightMatrix = weightMatrix./sum(sum(weightMatrix));  % Make the sum of the weight matrix equal to 1  归一化

                    WeightMatrix3d(:,:,1) = weightMatrix;
                    WeightMatrix3d(:,:,2) = weightMatrix;
                    WeightMatrix3d(:,:,3) = weightMatrix;
                    img_result(i,j,:) = sum(sum(minimatrix.*WeightMatrix3d,1),2);

            else
                continue;
            end
        end
    end
%     img_result = img_result(padding+1:padding+height,padding+1:padding+width,:);
end