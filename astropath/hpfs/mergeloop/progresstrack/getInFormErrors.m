%% getInFormErrors
% get the inform errors out of the batch files

function expectedinform = getInFormErrors(sname, informlog, actualim3num_internal)
%
if ~isempty(informlog)
    informlogfile = fullfile(informlog.folder, informlog.name);
    fileID = fopen(informlogfile);
    Batch = textscan(fileID,'%s','EndofLine','\r', 'whitespace','\t\n');
    fclose(fileID);
    Batch = Batch{1};
    %
    if isempty(Batch)
        expectedinform = actualim3num_internal;
    elseif contains(Batch{1}, 'Batch Run with Algorithm') %old version (GUI)
        Batch = Batch(2:end);
        ii = contains(Batch,sname);
        Batch = Batch (ii);
        Batch = extractBefore(Batch, ']');
        ii = unique(extractAfter(Batch, '_['));
        InformErrors = length(ii);
        %
        expectedinform = actualim3num_internal - InformErrors;
    else %new version (cmd line)
        ii = find(contains(Batch, 'Batch run had errors:'));
        if (any(ii))
            BatchErrors = Batch(ii:end);
            BatchErrors = extractBefore(BatchErrors, ']');
            ErrorCoords = extractAfter(BatchErrors, '_[');
            unqiueErrorCoords = unique(ErrorCoords);
            unqiueErrorCoordsnotnull = ~cellfun(@isempty, unqiueErrorCoords);
            InformErrors = sum(unqiueErrorCoordsnotnull);
            %
            expectedinform = actualim3num_internal - InformErrors;
        else
            expectedinform = actualim3num_internal;
        end
    end
else
    expectedinform = actualim3num_internal;
end
%              
end