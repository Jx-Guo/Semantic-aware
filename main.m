% Semantic-Aware Monostatic ISAC System  with Dynamic Multimodal Fusion
% ------------------------------------------------------------------------
% This script reads beam and channel data, processes received signals,
% estimates distances, overlays points on a BEV image, and adds semantic
% segmentation coloring. It demonstrates a dynamic multimodal fusion
% framework combining radio point cloud and visual semantics.
%
% Author: Jx Guo
% Repository: https://github.com/Jx-Guo/Semantic-aware
%
% First run the program NeuralBEV_and_Confidence.py
% Second run the program BEV_Processing.py 
clear;clc;close all;

%% Initialization and Parameters Setup
beam = readmatrix("beam2.xlsx","Sheet",2);
[r,~] = size(beam);
for i = 1:r
    beam_angle(beam(i,1)+1,1) = beam(i,2);
end
beam_angle(33) = 90;
beam_angle = beam_angle/180*pi;
NumBeam = size(beam_angle,1);

iteration = 12;
df = 120e3;
N = 1024;
c = 3e8;
fs = N*df;
fs_down = fs/4;
sortIdx = [34:64,1:33];
rho = 1;
Rmax = 30;
Rmin = Rmax/5;
Rmin = ceil(Rmin);
theta_min = 14.3615;
theta_max = 165.6385;
Rdisp = Rmax + 1.2;

date = '20241103';
scene = '1-1';
a1 = 0.5;  
b1 = 0;  
a2 = 1.62;  
b2 = 0; 

M = 65;
N = 64;

% Load BEV Image and Semantic Info
background = ['bev\background.jpg'];
background_img = imread(background);
xlsx_path = ['bev\' scene '_info.xlsx'];
data_table = readtable(xlsx_path);

% Read Point Cloud Data and Prepare Signals
path = ['RPCdata\' date '\' scene '\'];
re_t = load(strcat(path,'NR_Sym_Fre_1slot re.csv'));
im_t = load(strcat(path,'NR_Sym_Fre_1slot im.csv'));

framelength = 32932;
framelength = framelength/4;

for i = 1:NumBeam
    framepath = strcat(path,'pss_max_indx',num2str(i-1),'.csv');
    pathre = sprintf('t%dr%dre.csv',i-1,i-1);
    pathim = sprintf('t%dr%dim.csv',i-1,i-1);
    datapathre = strcat(path,pathre);
    datapathim = strcat(path,pathim);
    framestart(i) = load(framepath);
    if framestart(i)>framelength
        framestart(i) = framestart(i) - framelength;
    end

    re_r = load(datapathre);
    im_r = load(datapathim);

    signal_t = re_t + j*im_t;
    signal_r = re_r + j*im_r;
    signal_r = reshape(signal_r,[792,14]);
    channel_freq_domain_all = signal_r./signal_t;

    % average PDP
    [N_FFT,N_OFDM] = size(channel_freq_domain_all);
    for k = 1:N_OFDM
        h(:,k) = ifft(channel_freq_domain_all(:,k),N_FFT);
    end
    channel_time_domain = mean(h,2);
    channel_freq_domain = fft(channel_time_domain);
    channel_freq_domain_grid(:,i) = channel_freq_domain;

    if i == 1

        save('channel_freq_domain.mat','channel_freq_domain');

    end

    % bisection
    r_bisection(i) = bisection_toa_extract(channel_time_domain,iteration);
    intensity(i) = abs(fractional_order_idft(channel_freq_domain,r_bisection(i)-1));
end

%% distance estimate
base = 0;
height = 6*0.8;
r = (height.^2 + base^2).^(1/2);

r1 = framestart/fs_down*c/2;
r2 = (r_bisection-1)/fs*c/2;
bias = 9593.71442279816;
r_hat = r1 + r2 - bias;

r_hat = sqrt(r_hat.^2-base^2);

%% BEV and Raw Point Cloud

figure('color','w');
imshow(background_img);
hold on;

[img_height, img_width, ~] = size(background_img);
[X, Y] = pol2cart(beam_angle' + pi/2, r_hat);

valid_indices = (r_hat >= 3);
X_filtered = X(valid_indices);
Y_filtered = Y(valid_indices);
intensity_filtered = intensity(valid_indices);

% Calculate the scaling factor so that the point cloud covers the entire BEV image
max_X = 32/sqrt(2);
max_Y = 32/sqrt(2);
scale_factor = min(img_width / (2 * max_X), img_height / max_Y);

X_scaled = X_filtered * scale_factor + img_width / 2; 
Y_scaled = img_height - (Y_filtered * scale_factor); 
X_scaled = X_scaled + 0.5;

valid_points = (X_scaled >= 0 & X_scaled <= img_width) & (Y_scaled >= 0 & Y_scaled <= img_height);
X_filtered1 = X_scaled(valid_points);
Y_filtered1 = Y_scaled(valid_points);
intensity_filtered = intensity_filtered(valid_points);

X_filtered2 = X_filtered1 - 0.5;
Y_filtered2 = Y_filtered1 - 0.5;
X_ceil = ceil(X_filtered2);
Y_ceil = ceil(Y_filtered2);

point_ids = img_width * (Y_ceil - 1) + X_ceil;
intensity_normalized = intensity_filtered ./ (1 + abs(intensity_filtered));

%% Weight Computation and Point Selection

pixel_ids = data_table.PixelID; 
rgb_values = data_table.RGB;  
confidence_values = data_table.Confidence; 

rgb_values = cellfun(@(x) str2num(x(2:end-1)), rgb_values, 'UniformOutput', false); % 去掉括号并转化为数组
rgb_values = cell2mat(rgb_values); 

selected_confidences = [];
saved_points = []; % [Point ID, R]
kept_indices = [];

for idx = 1:length(point_ids)
    idx_xlsx = find(pixel_ids == point_ids(idx));
    if ~isempty(idx_xlsx)
        rgb = rgb_values(idx_xlsx, :);
        confidence = confidence_values(idx_xlsx);
        if confidence == 0; continue; end

        % BEV weight
        w1(idx) = a1 * confidence + b1;

        % Point cloud weight
        m = X_filtered1(idx);  
        n = Y_filtered1(idx); 
        tanc = abs(M - m) / abs(N - n);
        c_rad = atan(tanc);
        c_deg = rad2deg(c_rad);     
        R = 1 - 1 ./ exp(intensity_normalized(idx) .* sin(pi/2 - c_rad));
        w2(idx) = a2 * R + b2;

        % Total fusion weight
        f(idx) = w1(idx) + w2(idx);
        f_sigmoid(idx) = 1 / (1 + exp(-f(idx)));

        % Point cleaning
        if f_sigmoid(idx) >= 0
            kept_indices = [kept_indices, idx];
            saved_points = [saved_points, R];
        end

    end

end

% Collect confidences of kept points
for idx_kept = 1:length(kept_indices)
    x_coord = X_ceil(kept_indices(idx_kept));
    y_coord = Y_ceil(kept_indices(idx_kept)); 
    pixel_id = img_width * (y_coord - 1) + x_coord;
    idx_xlsx = find(pixel_ids == pixel_id);
    
    if ~isempty(idx_xlsx)
        confidence = confidence_values(idx_xlsx);
        selected_confidences = [selected_confidences; confidence];
    end

end

%% Plot Filtered Point Cloud on BEV

valid_points = false(size(X_filtered1));
valid_points(kept_indices) = true;
X_filtered3 = X_filtered1(valid_points);
Y_filtered3 = Y_filtered1(valid_points);
intensity_filtered3 = intensity_filtered(valid_points);

colormap(jet);
scatter(X_filtered3, Y_filtered3, 40, intensity_filtered3.^rho, 'filled');
grid on;
axis equal;
hold on;

% Adding angle tick marks
theta_polar_grid = 15:15:165;
hold on;
for i = theta_polar_grid
    [x, y] = pol2cart(deg2rad(i), Rmax * scale_factor);
    x_shifted = x + img_width / 2;
    y_shifted = img_height - y;
    plot([(img_width + 1) / 2 x_shifted], [img_height y_shifted], 'k-', 'LineWidth', 0.5, 'Color', [0.5 0.5 0.5]);
    text(x_shifted * 1.01, y_shifted * 1.01 - 3, [num2str(i), '\circ'], 'HorizontalAlignment', 'center', 'Color', 'black');
end

% Add distance tick marks
for r = Rmin:Rmin:Rmax
    theta_tick = linspace(deg2rad(theta_min), deg2rad(theta_max), 2000); 
    rho_tick = r * scale_factor * ones(size(theta_tick)); 
    [x_tick, y_tick] = pol2cart(theta_tick, rho_tick);
    x_tick_shifted = x_tick + img_width / 2;
    y_tick_shifted = img_height - y_tick;
    plot(x_tick_shifted, y_tick_shifted, 'k-', 'LineWidth', 0.5, 'Color', [0.5 0.5 0.5]);
    text(x_tick_shifted(1) * 1, y_tick_shifted(1) * 1.07, num2str(r), 'HorizontalAlignment', 'center', 'Color', 'black');
end

text(img_width / 2, img_height + 5, '', ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
    'FontSize', 12, 'FontWeight', 'bold', 'Color', 'black');

%% Semantic part

valid_points_3x3 = [];

for idx = 1:length(kept_indices)

    x_center = X_ceil(kept_indices(idx)); 
    y_center = Y_ceil(kept_indices(idx));     
    x_range = x_center-1:x_center+1;
    y_range = y_center-1:y_center+1;

    x_range = max(min(x_range, img_width), 1);  
    y_range = max(min(y_range, img_height), 1); 
    weighted_confidences = zeros(3, 3);

    for i = 1:length(x_range)
        for j = 1:length(y_range)
            point_id = img_width * (y_range(j) - 1) + x_range(i);
            idx_xlsx = find(pixel_ids == point_id);
            
            if ~isempty(idx_xlsx)
                rgb = rgb_values(idx_xlsx, :);
                confidence = confidence_values(idx_xlsx);              
                R = saved_points(idx); 
                weighted_confidence = confidence * R;
                valid_points_3x3 = [valid_points_3x3; point_id, weighted_confidence];
            end

        end
    end

    exp_confidences = exp(weighted_confidences - max(weighted_confidences(:)));  % 减去最大值进行数值稳定性处理
    probabilities = exp_confidences / sum(exp_confidences(:));  % Softmax归一化

    [~, max_idx] = max(probabilities(:));
    [max_i, max_j] = ind2sub(size(probabilities), max_idx);
    max_x = x_range(max_i);
    max_y = y_range(max_j);
    
    max_point_id = img_width * (max_y - 1) + max_x;
    idx_xlsx_max = find(pixel_ids == max_point_id);
    if ~isempty(idx_xlsx_max)
        rgb_max = rgb_values(idx_xlsx_max, :);
    end

    % Loop over kept_indices to draw 3x3 blocks with weighted confidences
    for i = 1:length(x_range)
        for j = 1:length(y_range)
            rectangle('Position', [x_range(i)-0.5, y_range(j)-0.5, 1, 1], ...
                      'EdgeColor', rgb_max / 255, 'LineWidth', 1, ...
                      'FaceColor', rgb_max / 255);            
        end
    end
end

% Semantic Segmentation Overlay and Legend
semantic_colors = struct( ...
    'Forest', [0, 92, 9], ...
    'Parking', [255, 229, 145], ...
    'Grass', [188, 255, 143], ...
    'Playground', [150, 133, 125], ...
    'Park', [0, 158, 16], ...
    'Building', [84, 155, 255], ...    
    'Water', [184, 213, 255], ...
    'Busway', [255, 128, 0], ...        
    'Wall', [0, 0, 0], ...
    'Hedge', [107, 68, 48], ...
    'Kerb', [255, 234, 0], ...
    'BuildingOutline', [0, 0, 255], ...
    'Path', [8, 237, 0], ...
    'Cycleway', [0, 251, 255], ...    
    'Road', [255, 0, 0], ...
    'TreeRow', [0, 92, 9], ...
    'Fence', [238, 0, 255], ...
    'Void', [int32(255 * 0.9), int32(255 * 0.9), int32(255 * 0.9)]);

category_names = fieldnames(semantic_colors); 

% Creating a Legend
num_cols = 2;  
num_rows = ceil(length(category_names) / num_cols);  

legend_x_offset = img_width * 1.15; 
legend_x_gap = 22; 

legend_y_offset = img_height * (-0.5); 
legend_height = 2; 
legend_spacing = 5.5;  

% Double column legend
for idx = 1:length(category_names)
    category = category_names{idx}; 
    rgb_value = semantic_colors.(category); 

    col_idx = mod(idx-1, num_cols); 
    row_idx = floor((idx-1) / num_cols);  

    x_pos = legend_x_offset + col_idx * legend_x_gap; 
    y_pos = legend_y_offset + row_idx * (legend_height + legend_spacing); 

    rectangle('Position', [x_pos, y_pos, 5, 4.5], ...
              'FaceColor', rgb_value / 255, 'EdgeColor', 'none');

    text(x_pos + 7, y_pos + legend_height / 2, category, ...
         'VerticalAlignment', 'middle', 'HorizontalAlignment', 'left', ...
         'FontSize', 12, 'Color', 'black');
end

%% Helper Function

function [L] = bisection_toa_extract(channel_time_domain,iteration)

channel_freq_domain = fft(channel_time_domain);
[A,B] = sort(abs(channel_time_domain),'descend');
max_val = A(1);
sec_val = A(2);
L = B(1);
R = B(2);

for i = 1:iteration
    mid = (L+R)/2;
    mid_val = abs(fractional_order_idft(channel_freq_domain,mid-1));
    if mid_val > max_val
        sec_val = max_val;
        max_val = mid_val;
        R = L;
        L = mid;
    elseif mid_val > sec_val
        sec_val = mid_val;
        R = mid;
    else
        return;
    end
end

end

function [x] = fractional_order_idft(channel_freq_domain,n)

N_FFT = length(channel_freq_domain);
Phase = (0:N_FFT-1)/N_FFT*n*2*pi;
MF = exp(j*Phase);

x = MF*channel_freq_domain/N_FFT;

end
