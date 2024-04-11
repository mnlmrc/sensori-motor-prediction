function ax = renderMatlab(vertices, faces, faceColor, borders, borderColor, borderSize, newFigure, backgroundColor, frame)
% Render the data in MATLAB
%
% Args:
%   vertices (matrix)
%       Matrix of vertices
%   faces (matrix)
%       Matrix of Faces
%   faceColor (matrix)
%       RGBA matrix of color and alpha of all vertices
%   borders (matrix)
%       Default is None
%   borderColor (char or MATLAB color)
%       Color of border
%   borderSize (double)
%       Size of the border points
%   newFigure (logical)
%       Create new Figure or render in current axis
%   frame (matrix)
%       [L,R,B,T] of the plotted area
%
% Returns:
%   ax (MATLAB axes)
%       Axis that was used to render the axis

% Default values for optional arguments
if nargin < 3 || isempty(faceColor), faceColor = [.5 .5 .5 1]; end  % White
if nargin < 4 || isempty(borders), borders = []; end             % No borders
if nargin < 5 || isempty(borderColor), borderColor = 'k'; end     % Black
if nargin < 6 || isempty(borderSize), borderSize = 1; end        % Size 1
if nargin < 7 || isempty(newFigure), newFigure = true; end       % New figure
if nargin < 8 || isempty(backgroundColor), backgroundColor = [1 1 1]; end % White background
if nargin < 9 || isempty(frame)
    frame = [min(vertices(:,1)), max(vertices(:,1)), min(vertices(:,2)), max(vertices(:,2))];
end

vertexIn = (vertices(:,1) >= frame(1)) & (vertices(:,1) <= frame(2)) & ...
           (vertices(:,2) >= frame(3)) & (vertices(:,2) <= frame(4));
faceIn = any(vertexIn(faces), 2);

patches = {};

for i = 1:size(faces(faceIn,:), 1)
    f = faces(faceIn(i), :);
    patches{end+1} = polyshape(vertices(f, 1:2));
end

if exist("newFigure")
    fig = figure('Color', backgroundColor, 'Position', [100, 100, 560, 560]);
else
    fig = gcf;
    set(fig, 'Color', backgroundColor);
end

ax = gca;
hold on;
for i = 1:length(patches)
    plot(patches{1, i}, 'FaceColor', faceColor, 'LineStyle', 'none');
end

axis equal;
axis off;
xlim([frame(1), frame(2)]);
ylim([frame(3), frame(4)]);

if ~isempty(borders)
    plot(borders(:,1), borders(:,2), '.', 'Color', borderColor, ...
         'MarkerSize', borderSize, 'LineWidth', 0);
end

end
