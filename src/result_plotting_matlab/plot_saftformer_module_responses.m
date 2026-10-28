function [fig, data] = plot_saftformer_module_responses(matFile, outputPath, options)
%PLOT_SAFTFORMER_MODULE_RESPONSES Native-vector ICSA and STF-GLU diagnostics.
% One fixed test example supplies every panel. Shared color limits are
% computed over the entire exported subset, with no per-map normalization.
% Relative RMS is condition RMS / pre-condition branch RMS, not attribution.
arguments
    matFile (1,:) char
    outputPath (1,:) char
    options struct = struct()
end
raw=load(matFile);
required={'iq','icsa_weights','temporal_relative_rms','frequency_relative_rms', ...
    'labels','predictions','channel','snr','classNames','channelNames','sourceIndices','checkpointHash'};
assert(all(isfield(raw,required)),'Layer MAT is missing required fields.');
n=size(raw.iq,1);
sample=option(options,'SampleIndex',1);
validateattributes(sample,{'numeric'},{'scalar','integer','>=',1,'<=',n});
assert(isequal(size(raw.icsa_weights),[n,59,6]));
assert(isequal(size(raw.temporal_relative_rms),[n,6,59]));
assert(isequal(size(raw.frequency_relative_rms),[n,6,59]));
assert(all(isfinite(raw.icsa_weights),'all') && all(raw.icsa_weights>=.5 & raw.icsa_weights<=1.5,'all'));
assert(all(isfinite(raw.temporal_relative_rms) & raw.temporal_relative_rms>=0,'all'));
assert(all(isfinite(raw.frequency_relative_rms) & raw.frequency_relative_rms>=0,'all'));
weights=squeeze(raw.icsa_weights(sample,:,:))';
temporal=squeeze(raw.temporal_relative_rms(sample,:,:));
spectral=squeeze(raw.frequency_relative_rms(sample,:,:));
iq=squeeze(raw.iq(sample,:,:));
amplitude=sqrt(sum(iq.^2,1));
gateExtent=max(.05,ceil(double(max(abs(raw.icsa_weights-1),[],'all'))/.05)*.05);
gateLimits=[1-gateExtent,1+gateExtent];
responseMaximum=max(max(raw.temporal_relative_rms,[],'all'),max(raw.frequency_relative_rms,[],'all'));
responseLimits=[0,max(.1,ceil(double(responseMaximum)*10)/10)];
gateColors=interp1([0,.5,1],[.267,.467,.667;1,1,1;.933,.467,.2],linspace(0,1,129));
responseColors=parula(128);
figureSize=option(options,'FigureSize',[1320,650]);
fontSize=option(options,'FontSize',15);
fig=figure('Color','white','Units','pixels','Position',[80,60,figureSize], 'Renderer','painters');

waveAx=axes(fig,'Units','normalized','Position',[.070,.695,.350,.205], 'Tag','module_input_axes');
hold(waveAx,'on');
inputI=plot(waveAx,0:511,iq(1,:),'Color',[.267,.467,.667], ...
    'LineWidth',1.1,'DisplayName','I','Tag','module_input_i');
inputQ=plot(waveAx,0:511,iq(2,:),'Color',[.267,.667,.600], ...
    'LineWidth',1.1,'DisplayName','Q','Tag','module_input_q');
style_axes(waveAx,fontSize);
xlim(waveAx,[-5,516]); xticks(waveAx,0:128:512);
inputLimit=1.15*max(abs(iq),[],'all');
ylim(waveAx,[-inputLimit,inputLimit]);
xlabel(waveAx,'Sample index'); ylabel(waveAx,'Amplitude');
title(waveAx,'Input I/Q','FontWeight','normal','FontSize',fontSize+1);
inputLegend=legend(waveAx,[inputI,inputQ],{'I','Q'},'NumColumns',2, ...
    'Location','northeast','Box','off','FontName','Times New Roman', ...
    'FontSize',fontSize-1,'TextColor','black','AutoUpdate','off');
inputLegend.ItemTokenSize=[16,8];

gateAx=draw_matrix(fig,[.070,.145,.350,.370],weights,gateLimits,gateColors,fontSize,'module_icsa_axes');
ylabel(gateAx,'Within-frame unit'); xlabel(gateAx,'Frame index');
title(gateAx,'ICSA weights','FontWeight','normal','FontSize',fontSize+1);
draw_scale(fig,[.438,.145,.011,.370],gateLimits,gateColors, ...
    [gateLimits(1),1,gateLimits(2)],'Weight',fontSize,'module_icsa_colorbar');

timeAx=draw_matrix(fig,[.580,.615,.325,.285],temporal,responseLimits,responseColors,fontSize,'module_temporal_axes');
ylabel(timeAx,'Block'); xlabel(timeAx,'Frame index');
title(timeAx,'Temporal condition','FontWeight','normal','FontSize',fontSize+1);
freqAx=draw_matrix(fig,[.580,.145,.325,.285],spectral,responseLimits,responseColors,fontSize,'module_spectral_axes');
ylabel(freqAx,'Block'); xlabel(freqAx,'Frame index');
title(freqAx,'Spectral condition','FontWeight','normal','FontSize',fontSize+1);
scaleTicks=linspace(responseLimits(1),responseLimits(2),4);
draw_scale(fig,[.922,.615,.011,.285],responseLimits,responseColors,scaleTicks, ...
    'Relative RMS',fontSize,'module_temporal_colorbar');
draw_scale(fig,[.922,.145,.011,.285],responseLimits,responseColors,scaleTicks, ...
    'Relative RMS',fontSize,'module_spectral_colorbar');

classNames=string(raw.classNames(:)); channelNames=string(raw.channelNames(:));
data=struct('sampleIndex',sample,'sourceIndex',raw.sourceIndices(sample), ...
    'className',classNames(double(raw.labels(sample))+1),'snr',raw.snr(sample), ...
    'channelName',channelNames(double(raw.channel(sample))+1), ...
    'prediction',classNames(double(raw.predictions(sample))+1),'checkpointHash',string(raw.checkpointHash), ...
    'iq',iq,'amplitude',amplitude,'inputView','I/Q','icsaWeights',weights, ...
    'temporalRelativeRMS',temporal,'frequencyRelativeRMS',spectral, ...
    'gateColorLimits',gateLimits,'responseColorLimits',responseLimits, ...
    'sourceSubsetSize',n,'sourceFile',matFile,'perMapNormalization',false, ...
    'definition','Condition RMS divided by branch RMS; activation magnitude, not attribution.');
fig.UserData=data;
drawnow;
[folder,baseName]=fileparts(outputPath);
if ~isempty(folder) && ~isfolder(folder),mkdir(folder);end
exportgraphics(fig,outputPath,'Resolution',300,'BackgroundColor','white');
exportgraphics(fig,fullfile(folder,[baseName,'.pdf']),'ContentType','vector','BackgroundColor','white');
savefig(fig,fullfile(folder,[baseName,'.fig']));
fprintf('Saved module responses: %s; %s / %s / %g dB; predicted %s.\n', ...
    outputPath,data.className,data.channelName,data.snr,data.prediction);
end

function ax=draw_matrix(fig,position,values,limits,colors,fontSize,tag)
ax=axes(fig,'Units','normalized','Position',position,'Tag',tag); hold(ax,'on');
[rows,columns]=size(values);
indices=1+floor((double(values)-limits(1))/diff(limits)*(size(colors,1)-1));
assert(all(indices>=1 & indices<=size(colors,1),'all'),'Color limits would clip an actual response.');
for r=1:rows
    for c=1:columns
        rectangle(ax,'Position',[c-.5,r-.5,1,1], ...
            'FaceColor',colors(indices(r,c),:),'EdgeColor','none', ...
            'Tag','module_response_cell','UserData',double(values(r,c)));
    end
end
style_axes(ax,fontSize); set(ax,'YDir','reverse','Layer','top');
xlim(ax,[.5,columns+.5]); ylim(ax,[.5,rows+.5]);
xticks(ax,[1,10,20,30,40,50,59]); yticks(ax,1:rows);
end

function draw_scale(fig,position,limits,colors,ticks,label,fontSize,tag)
ax=axes(fig,'Units','normalized','Position',position,'Tag',tag);hold(ax,'on');
step=diff(limits)/size(colors,1);
for k=1:size(colors,1)
    rectangle(ax,'Position',[0,limits(1)+(k-1)*step,1,step], ...
        'FaceColor',colors(k,:),'EdgeColor','none','Tag','module_response_scale_cell');
end
style_axes(ax,fontSize-1);set(ax,'YAxisLocation','right','Layer','top');
xlim(ax,[0,1]);ylim(ax,limits);xticks(ax,[]);yticks(ax,ticks);
ylabel(ax,label,'FontSize',fontSize);
end

function style_axes(ax,fontSize)
set(ax,'Box','on','TickDir','in','FontName','Times New Roman', ...
    'FontSize',fontSize,'XColor','black','YColor','black','LineWidth',.8, ...
    'TickLabelInterpreter','none');
end

function value=option(options,name,default)
if isfield(options,name) && ~isempty(options.(name)),value=options.(name);else,value=default;end
end
