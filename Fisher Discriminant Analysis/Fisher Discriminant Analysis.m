% Loading Data
clear all
close
load("Data_Ques4.mat");
% Since it can be seen there are 3 classes, we will calculate each values for each class manually 
% rather than in a for loop since the number of classes is manageable.

% Convert into column vectors for each measurement

x = Data;

[dim, N ] = size(x);
dim_reduced = 2;

N1 = sum( src_id ==1);
N2 = sum( src_id ==2);
N3 = sum( src_id ==3);


% Sample mean and variance for each class
mu1 = mean(x( :, src_id ==1),2);
mu2 = mean(x(:,src_id ==2),2);
mu3 = mean(x(:,src_id ==3),2);
% Covariance
Sigma1 = cov(x(:,src_id ==1)');
Sigma2 = cov(x(:,src_id ==2)');
Sigma3 = cov(x(:,src_id ==3)');

% Full dimension classification
d1 = -1/2 * log(det(2*pi*Sigma1)) - 1/2*dot( x - mu1, Sigma1 \ (x - mu1));
d2 = -1/2 * log(det(2*pi*Sigma2)) - 1/2*dot( x - mu2, Sigma2 \ (x - mu2));
d3 = -1/2 * log(det(2*pi*Sigma3)) - 1/2*dot( x - mu3, Sigma3 \ (x - mu3));

select_full (d1 > d2) = 1;
select_full (d2 > d1) = 2;
select_full (d2 > d3) = 2;
select_full (d3 > d2) = 3;
select_full (d3 > d1) = 3;
select_full (d1 > d3) = 1;


% Count the times the results are wrong
error_full = sum( select_full ~= src_id) / N 
% Fisher Linear Discriminant
delta1 = x(:, src_id ==1) - mu1;
delta2 = x(:, src_id ==2) - mu2;
delta3 = x(:, src_id ==3) - mu3;

S1 = delta1*delta1';
S2 = delta2*delta2';
S3 = delta3*delta3';
SW = S1 + S2 + S3;

mu = mean (x, 2);

deltam1 = mu1 - mu;
deltam2 = mu2 - mu;
deltam3 = mu3 - mu;

SB = N1 * (deltam1*deltam1') + N2 * (deltam2 * deltam2') + N3 * (deltam3 * deltam3');

% Sort descendingly and get the largestest eigenvalues
[ VFLD, DFLD] = eigs( SB, SW, 2);
[~, idx_eig] = sort(diag(DFLD), 'descend');
idx_keep = idx_eig(1 : dim_reduced);
W_FLD = VFLD(:, idx_keep);
% W = VFLD (:, end-1:end);
W = W_FLD;

reduced = W'*x;

mu1_reduced = W'*mu1;
mu2_reduced = W'*mu2;
mu3_reduced = W'*mu3;


Sigma1_reduced = W' * Sigma1 * W;
Sigma2_reduced = W' * Sigma2 * W;
Sigma3_reduced = W' * Sigma3 * W;

d1_reduced = -1/2*log(det(2*pi*Sigma1_reduced)) - 1/2*dot( reduced - mu1_reduced ...
    , Sigma1_reduced \ ( reduced - mu1_reduced));
d2_reduced = -1/2*log(det(2*pi*Sigma2_reduced)) - 1/2*dot( reduced - mu2_reduced ...
    , Sigma2_reduced \ ( reduced - mu2_reduced));
d3_reduced = -1/2*log(det(2*pi*Sigma3_reduced)) - 1/2*dot( reduced - mu3_reduced ...
    , Sigma3_reduced \ ( reduced - mu3_reduced));

select_fld(d1_reduced > d2_reduced) = 1;
select_fld(d2_reduced > d1_reduced) = 2;
select_fld(d2_reduced > d3_reduced) = 2;
select_fld(d3_reduced > d2_reduced) = 3;
select_fld(d3_reduced > d1_reduced) = 3;
select_fld(d1_reduced > d3_reduced) = 1;
error_fld = sum(select_fld ~= src_id) / N
plot(reduced (1, src_id == 1), reduced( 2, src_id ==1) , 'bo', reduced(1, src_id == 2), ...
    reduced(2, src_id ==2), 'ro', reduced(1, src_id == 3), reduced(2, src_id ==3), 'go');
