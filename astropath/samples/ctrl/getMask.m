function m = getMask(C,A,varargin)
%%--------------------------------------------
%% Build a mask for the image
%%
%% 2020-08-06   Alex Szalay
%%--------------------------------------------
    %
    opt = 0;
    if (numel(varargin)>0)
        opt = varargin{1};
    end
    %
    w = 8.0;
    thr = 11;
    s1 = strel('disk',75);
    %
    b = A.img{1}*0.5;
    for i=2:7
        a = A.img{i};
        b = b + a;
    end
    c = A.img{9};
    %-----------------------------------------------------------
    % apply a small smoothing, thresholding and then morphological
    % open/close operation to create a reasonably contiguous mask
    %-----------------------------------------------------------
    b = imgaussfilt(b,w);
    thr = getThreshold(b);
    m  = b>thr;
    m  = imopen(m,s1);
    m  = imclose(m,s1);
    m  = bwareaopen(m,450000);
    m  = ~bwareaopen(~m,50000);
    %-----------------------------------------
    % show the mask and histogram if needed
    %-----------------------------------------
    if (opt==1)
        figure(1);
        subplot(2,2,1);
            histogram(asinh(b(:)));
            hold on
            yl = get(gca,'YLim');
            plot([thr, thr],[0,yl(2)*0.8],'r:','LineWidth',2);
            hold off            
        subplot(2,2,3);
            imshow(m);
        subplot(2,2,2);
            imshow(2*A.img{9});
        subplot(2,2,4);      
            imshow(2*c);
            hold on
            mc = uint8(imresize(m,1/8));
            mc = mc(1:376,1:503);            
            visboundaries(mc);
            hold off
        shg
    end
    %
end


