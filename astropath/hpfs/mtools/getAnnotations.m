%%----------------------------------------------------
%% read XML file with grid annotations
%% Daphne Schlesinger, 2017
%% edited by Alex Szalay, 2018-0725
%% edited by Benjamin Green, 2018-1212
%%----------------------------------------------------
%% take the path for an xmlfile corresponding to a qptiff. 
%% and produce the number of expected im3's from the annoatations
%%------------------------------------------------------
function [expectim3num] = getAnnotations(root, scan, smp)
filepath = [root,smp,...
    '_', scan '_annotations.xml'];
%
XML = xmlread(filepath);
annList = XML.getFirstChild;
ann = annList.item(5);
% for phenochart update
temp = ann.item(1);
%for i2 = 1:temp.getLength
s = temp.getAttribute('subtype');
if s.equals("ROIAnnotation")
    F = ROIAnnotationRead(ann);
else
    %
    B = cell(1);
    track = 1;
    %
    % get rectangular annotations
    %
    for i1 = 1:2: ann.getLength - 1
        temp = ann.item(i1);
         try
             s = temp.getAttribute('subtype');
         catch
             continue;
         end
        if  s.equals("RectangleAnnotation")
            B{track} = temp;
            track = track + 1;
        else
        end
    end
    %
    F = cell(1);
    track2 = 1;
    for i2 = 1 : length(B)
        %
        node = B{i2};
        history = node.item(7);
        histlastRef = history.getLength-2;
        histRef = history.item(histlastRef);
        %
        f =  histRef.item(3).getTextContent;
        t =  histRef.item(7).getTextContent;
        if strcmp(t, 'Acquired')
            f = char(f);
            F{track2} = f(5:end);
            track2 = track2 + 1;
        end
        %
    end
    %
end
expectim3 = cellfun(@(x)erase(x,'_M2'),F,'Uni',0);
expectim3num = length(unique(expectim3));
end
%% ROIAnnotationRead
%% --------------------------------------------------------------
%% Created by: Benjamin Green - Johns Hopkins - 01/08/2020
%% --------------------------------------------------------------
%% Description:
%%% read the annotation for update inForm verions
%% --------------------------------------------------------------
%%
function F = ROIAnnotationRead(ann)
F = cell(1);
track2 = 1;
%
for i4 = 1:2:ann.getLength
    try
        s = ann.item(i4).getAttribute('subtype');
    catch
        continue
    end
    
    if s.equals("ROIAnnotation") || s.equals('TMASectorAnnotation')
        for i1 = 1:2: ann.item(i4).item(13).getLength - 1
            temp = ann.item(i4).item(13).item(i1);
            s = temp.getAttribute('subtype');
            if  s.equals("RectangleAnnotation")
                node = temp;
                %
                history = node.item(7);
                histlastRef = history.getLength-2;
                histRef = history.item(histlastRef);
                %
                f =  histRef.item(3).getTextContent;
                t =  histRef.item(7).getTextContent;
                if strcmp(t, 'Acquired')
                    f = char(f);
                    F{track2} = f(5:end);
                    track2 = track2 + 1;
                end
            end
        end
        %
    end
end
end
