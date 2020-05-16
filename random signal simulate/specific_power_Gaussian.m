N = 2048;           % 设定采样点数2048个；
B = 20000;          % 设定带宽20000Hz；
k = 1;              % 设定功率谱密度1；
f = 0-B/2:B/N:B-B/N-B/2;    % 频谱列表

% 此处得到的函数相当于原函数乘以窗（门）函数
% 因此得到的各种频谱函数也相当于频函数卷积窗函数
% 并且由于门函数的边缘效应会使边缘信号的权重降低
% 因此设立各种窗函数来降低由此带来的影响（如hanmming、hann等）

noise = randn(size(f))*(k*B)^0.5;    
% 此处由于窗函数，且带通或低通高斯信号功率等于方差，而功率=带宽*功率谱密度，由此产生了具有特定功率谱的高斯随机信号
subplot(2,2,1); hist(noise, 50);title('噪声的概率分布');

% 周期图法
Nf = fft(noise,N);
GN = power(abs(Nf), 2)/N;             % 噪声功率谱= （1/N）*N（w）^2

subplot(2,2,2); plot(f,GN);title('周期图法功率谱');xlabel('f/Hz');
fprintf('周期图法功率谱密度准确度为：%f\n', (mean(GN)/B)/k);

% 自相关函数法
noise_corr = xcorr(noise);
Gn = abs(fft(noise_corr));
index = 0:round(length(Gn)/2-1);
Gn = Gn(index+1)/N;

subplot(2,2,3);plot(f,Gn);title('自相关函数法功率谱');xlabel('f/Hz');
fprintf('自相关函数法功率谱密度准确度为：%f\n', (mean(Gn)/B)/k);


% 不对劲
window = hann(length(f));
G = periodogram(noise,window, N);

subplot(2,2,4);plot(G);title('法噪声功率谱');xlabel('f/Hz');
fprintf('法功率谱密度准确度为：%f\n', (mean(G)/B)/k);





