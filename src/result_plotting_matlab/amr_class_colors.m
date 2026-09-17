function colors = amr_class_colors(numClasses)
%AMR_CLASS_COLORS Return muted categorical colors for AMR figures.

palette = [
    0.122, 0.467, 0.706
    1.000, 0.498, 0.055
    0.173, 0.627, 0.173
    0.839, 0.153, 0.157
    0.580, 0.404, 0.741
    0.549, 0.337, 0.294
    0.890, 0.467, 0.761
    0.498, 0.498, 0.498
    0.737, 0.741, 0.133
    0.090, 0.745, 0.812
    0.682, 0.780, 0.910
    1.000, 0.733, 0.471
    0.596, 0.875, 0.541
    1.000, 0.596, 0.588
    0.773, 0.690, 0.835
    0.769, 0.612, 0.580
    0.969, 0.714, 0.824
    0.780, 0.780, 0.780
    0.859, 0.859, 0.553
    0.620, 0.855, 0.898
];

if numClasses <= size(palette, 1)
    colors = palette(1:numClasses, :);
else
    colors = interp1(1:size(palette, 1), palette, ...
        linspace(1, size(palette, 1), numClasses), 'linear');
end
end
