function pad = imgPadding(img,padding)
% @author:Tanlang
% @illustration:
    % padding
height = size(img,1); width = size(img,2);
if size(img,3)>1
    pad = zeros(height+2*padding,width+2*padding,3);
    pad(1+padding:padding+height,1+padding:padding+width,:) = img;
else
    pad = zeros(height+2*padding,width+2*padding);
    pad(1+padding:padding+height,1+padding:padding+width) = img;
end
end
