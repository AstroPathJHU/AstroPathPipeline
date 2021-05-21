function C = showPolygons(C,test)
%%-----------------------------------------------
%% show the outlines of the annotations
%% together with the QPTIFF info
%% test:
%%  1: C.Q.img, color image, no annotations
%%  2: C.qimg, largest qptiff, original annotations
%%  3: C.qimg, scaled by AP scale, with scaled annotations
%%  4: C.aimg, no scaling
%%  5: C.aimg, warped
%%  6: C.aimg, generic linear fit
%% Alex Szalay, Baltimore, 2019-02-13
%%-----------------------------------------------
    %------------------------
    % start with the QPTIFF
    %------------------------
    if (test<1 | test>6)
        return
    end
    %
    if (isfield(C,'Q')==0)
        C.Q = getQPTiff(C);
    end
    %
    if (isfield(C,'aimg')==0)
        C = getImages(C);
    end
    %
    showLayout(C,test);
    hold on
    %---------------------------------
    % non-tiled linear fit parameters
    %---------------------------------
    % generic mean values from clipped set
    D.px = [0.298011507230586   0.004842728394389   0.001521831906142];
    D.py = [26.346387517257412  -0.001121007596434   0.004262002347126];
    %
    %-------------------------
    % loop through the layers
    %-------------------------
    for n=1:numel(C.PA.layer)
        %
        layer = C.PA.layer(n);
        %--------------------------------
        % get the layer info, like color
        %--------------------------------
        a = C.PA(C.PA.layer==layer,:);
        cc = char(a.color);
        clr = [hex2dec(cc(1:2)), hex2dec(cc(3:4)),hex2dec(cc(5:6))]/255;
        %---------------------------
        % extract relevant regions
        %---------------------------
        r = C.PR(C.PR.layer==layer,:);
        %-------------------
        % get the vertices
        %-------------------
        v = C.PV(ismember(C.PV.regionid,r.regionid),:);
        %
        M = numel(r.regionid);
        %
        sc = 1.0;
        if (test>=3)
            sc = sc*C.iqscale;
        end
        %
        for m=1:M
            %
            ltype = '-';
            if (r.isNeg(m))
                ltype = ':';
            end
            %
            mx = find(v.regionid==r.regionid(m));
            vv = v(mx,:);
            x  = vv.x;
            y  = vv.y;
            %--------------------------------------
            % scale it to the size of the image
            %--------------------------------------
            if (test==5)
                %
                tx = vv.wx/C.ppscale;
                ty = vv.wy/C.ppscale;
                %
            elseif (test==6)
                tx = x + D.px(1)+D.px(2)*x+D.px(3)*y;
                ty = y + D.py(1)+D.py(2)*x+D.py(3)*y;
            else
                tx = sc*x;
                ty = sc*y;
            end
            plot(tx, ty,ltype,'Color',clr,'LineWidth',2,...
                'MarkerSize',2);
        end
    end
    %
    axis equal
    hold off
    %
    shg
    %
end


