data = load('ex1data2yo.txt');
m = size(data, 1)
%X = [ones(m, 1), data(:,1:2)];

X = [data(:,1:2)];
y = data(:,3);

[X_norm, mu, sigma] = featureNormalize(X);

X_norm
mu
sigma
