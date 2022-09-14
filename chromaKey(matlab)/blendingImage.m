function img = blendingImage(fg,bg,mask)
% @author:Tanlang
% @illustration:
%     compose image
    height = size(fg,1); width = size(fg,2);
    mask3d = zeros(height,width,3);
    for i = 1:3
        mask3d(:,:,i) = mask;
    end
    img = (fg .* mask3d) + bg .* (1-mask3d);
end