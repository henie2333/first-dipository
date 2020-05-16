% s = cosw1*t + cosw2*t + N(0,1)
% f1 = 300hz, f2 = 310hz
t = 0:0.001:0.3;
f1 = 300;f2 = 310;
s = cos(2*pi*f1*t) + cos(2*pi*f2*t) + randn(size(t))  % randn:标准正态分布
s_mean = mean(s);
s_var = var(s);
s_corr = xcorr(s)/150000;    % 150000为数据点数300乘频点带宽
Fs = 1000; nfft = 512;

subplot(2,2,1)
periodogram(s,[], nfft, Fs);
title('no window')

subplot(2,2,2)
window = hamming(301);
periodogram(s,window, nfft, Fs);
title('hamming window')

subplot(2,2,3)
s_G = fft(s_corr, nfft);
s_G = 10*log10(abs(s_G));
index = 0:round(length(s_G)/2-1);
f = index*Fs/length(s_G);
s_G = s_G(index+1);
plot(f, s_G)
title('corrleation fft')
axis([0, 500, -50, 0]);
grid on


