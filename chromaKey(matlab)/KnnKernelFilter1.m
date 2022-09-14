function img_result = KnnKernelFilter1(img,unknow,alpha,fgRegion,kernel_size,k,padding)
    % @author:Tanlang
    % @illustration:
        % function: find some foreground pixel values near the edge pixel as the surrogate value
                  % kernel_size is the size of found region
                  % k is the num of neigborhood
        % ensure padding > kernel_size/2
        % ensure kernel_size is odd
        % fgRegion is absolute foreground

    %% set variable
    height = size(img,1); width = size(img,2);d=size(img,3);
    img = imgPadding(img,padding);
    unknow = imgPadding(unknow,padding);
    fgRegion = imgPadding(fgRegion,padding);
    alpha = imgPadding(alpha,padding);
    img_result = img;

    WeightGau = fspecial('gaussian',[5,5],1);
    WeightGau3d(:,:,1) = WeightGau;
    WeightGau3d(:,:,2) = WeightGau;
    WeightGau3d(:,:,3) = WeightGau;

    %% KNN kernel filter
    for i = padding: height+padding
        for j = padding:width+padding 
            if unknow(i,j) > 0.01
                    minimatrix = img(i-floor(kernel_size/2):i+floor(kernel_size/2),j-floor(kernel_size/2):j+floor(kernel_size/2),:);         
                    AlphaMatrix = fgRegion(i-floor(kernel_size/2):i+floor(kernel_size/2),j-floor(kernel_size/2):j+floor(kernel_size/2),:);

                    % 矩形框内前景像素小于k,则跳过
                    if  length(find(AlphaMatrix~=0)) < k
                        continue;
                    end

                    AlphaMatrix3d(:,:,1) = AlphaMatrix;
                    AlphaMatrix3d(:,:,2) = AlphaMatrix;
                    AlphaMatrix3d(:,:,3) = AlphaMatrix;

                    minimatrix = (minimatrix .* AlphaMatrix3d);  % get fg pixel

                    [a,b] = ind2sub([kernel_size,kernel_size],1:kernel_size^2);
                    feature_vector = [reshape(minimatrix,kernel_size^2,d)';[a;b]/sqrt(2*kernel_size^2)+rand(2,kernel_size^2)*1e-6 ];   % Normalize the pixel coordinates, and add random noise  

                    ind = find(feature_vector(1,:)~=0);
                    center_feature = repmat([reshape(img(i,j,:),1,d)';[floor(kernel_size/2)+1;floor(kernel_size/2)]/sqrt(2*kernel_size^2)],[1,kernel_size^2]);  % 中心点像素特征矩阵，把它扩展为大小 5*kernel_size^2
                    constant = max(sum(kerf(abs(feature_vector-center_feature)),1));    % 计算最大上界常量，以保证邻近性值小于1
                    weightVector = 1-sum(kerf(abs(feature_vector-center_feature)),1)./constant;  % 核函数计算邻近性    

                    % get positive element
                    weightVector_n = zeros(1,kernel_size^2);    % 尺寸为1*kernel_size^2,存放矩形框内的所有绝对前景像素关于该像素的的邻近值，不是前景像素的元素值置0
                    for n = 1:length(ind)
                        weightVector_n(ind(n)) = weightVector(ind(n));
                    end

                    [foreWeight,index] = sort(weightVector_n,2,'descend');     % 邻近值降序排列，得到foreWeight矩阵，和邻近像素的索引index，（即邻近像素位置）

                  %% 定义相似度下限  V3 改进
                    LowLimit = 0.6;
                    foreWeight = foreWeight.*(foreWeight>LowLimit);
                    if  length(find(foreWeight~=0)) < k/3
                        continue;
                    end

                  %%
                    weights = foreWeight(1:k); index = index(1:k);  % 取得前k个邻近值
                    weights = weights./sum(sum(weights));  % Make the sum of the weight matrix equal to 1  归一化

                    WeightMatrix = zeros(kernel_size,kernel_size);      % 用于存放邻近像素的归一化后的邻近值
                    WeightMatrix(index)=weights;    % 把邻近值存在 WeightMatrix矩阵
                    WeightMatrix3d(:,:,1) = WeightMatrix;
                    WeightMatrix3d(:,:,2) = WeightMatrix;
                    WeightMatrix3d(:,:,3) = WeightMatrix;

                    img_result(i,j,:) = sum(sum(minimatrix.*WeightMatrix3d,1),2);

                  %% Gaussion filter
%                     miniMatrix = img(i-floor(5/2):i+floor(5/2),j-floor(5/2):j+floor(5/2),:);
%                     img_result(i,j,:) = sum(sum( miniMatrix.*WeightGau3d ));

            else
                continue;
            end
        end
    end
    img_result = img_result(padding+1:padding+height,padding+1:padding+width,:);
end