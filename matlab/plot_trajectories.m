% Reading trajectories from dataset:
dataset_trajectories_file = fopen('../dataset/trajectory.dat', 'r');
odom_trajectory = [];
gt_trajectory = [];

while ~feof(dataset_trajectories_file)
    line = fgetl(dataset_trajectories_file);
    A = sscanf(line, 'pose: %f%f%f%f%f%f%f', [1 7]);
    odom_trajectory = [odom_trajectory; A(2:4)];
    gt_trajectory = [gt_trajectory; A(5:end)];
end

fclose(dataset_trajectories_file);

% Reading trajectory from SLAM:
slam_trajectory_file = fopen('../bin/slam.dat');
slam_trajectory = [];

while ~feof(slam_trajectory_file)
    A = fscanf(slam_trajectory_file, '%f%f%f', [1 3]);
    slam_trajectory = [slam_trajectory; A];
end

fclose(slam_trajectory_file);

% Reading landmarks from dataset:
dataset_world_file = fopen('../dataset/world.dat', 'r');
landmarks = [];

while ~feof(dataset_world_file)
    line = fgetl(dataset_world_file);
    A = sscanf(line, '%f', [1 14]);
    landmarks = [landmarks; A(2:4)];
end

fclose(dataset_world_file);

% Plotting the data:
plot(gt_trajectory(:,1), gt_trajectory(:,2), 'g', ...
     slam_trajectory(:,1), slam_trajectory(:,2), 'r', ...
     odom_trajectory(:,1), odom_trajectory(:,2), 'k', ...
     landmarks(:,1), landmarks(:,2), 'go', 'MarkerSize', 2.5);

title('PlanarMonocularSLAM')
xlabel('x');
ylabel('y');
legend({'Ground Truth', 'SLAM', 'Odometry', 'Landmarks (GT)'},'Location','southwest');
grid on;
grid minor;

saveas(gcf, 'PlanarMonocularSLAM.png');

% Save as .pdf for latex:
set(gcf, 'Units', 'Inches');
pos = get(gcf, 'Position');
set(gcf, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);
saveas(gcf, 'PlanarMonocularSLAM.pdf');
