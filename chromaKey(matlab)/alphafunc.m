function alpha = alphafunc(dis,a,b)
    %% foreground region
    if(dis>=0 && dis<a)
        alpha= 255;
    %% unkonow region
    elseif(dis>=a && dis<b)  
        alpha = 255-(255/(b-a)) * (dis-a);     
    %% background region
    else
        alpha = 0;
    end
end
