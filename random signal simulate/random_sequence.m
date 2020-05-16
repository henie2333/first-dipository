% 按照如下模型产生一组随机序列:?x(n)=0.8x(n-1)+w(n)，其中w(n)为均值为0，方差为4的高斯白噪声序列
% （1）模拟产生X(n)序列的500?观测样本函数，绘出波形图。?
% （2）用观测点估计信号的均值和方差。?
% （3）估计该过程的自相关函数和功率谱密度，并画出图形。

% 由于冲激函数h（n）=1/（1-0.8z**-1）
b = [1];a = [1 -0.8];
[h, t] = impz(b,a,50);


% w(n)为均值为0，方差为4的高斯白噪声序列
w = normrnd(0,2,1,200);

% x = w卷积h
x = [];
for n = 1:200
    x_n=0;
    for i = 1:min([n-1, 50])
        x_n = x_n + h(i)*w(n-i);
    end
    x = [x, x_n];
end
n = 1:200;
subplot(3,2,1);
plot(n, x);
title('traditional way');
new_x = filter(b,a,w);
subplot(3,2,2);
plot(n,x,'r');
title('way with toolbox');

x_mean = mean(x);
subplot(3,2,3);
plot([0,1],[x_mean, x_mean]);
title('x_m_e_a_n');
subplot(3,2,4)
x_var = var(x);
plot([0,1], [x_var,x_var]);
title('x_v_a_r');


% 求自相关函数
Mlag = 20;
R = xcorr(x, Mlag, 'coeff');
m = -Mlag:Mlag;
subplot(3,2,5);
plot(m, R);
title('Rx function')
grid on


% 求功率谱
Fs = 1000;  % 采样率
nfft = 1024;    % fft所含数据点数
CXk=fft(R,nfft);
Pxx=abs(CXk);
index=0:round(nfft/2-1);
plot_Pxx=10*log10(Pxx(index+1));
f = index*Fs/nfft;  % 频率
subplot(3,2,6);
plot(f, plot_Pxx);
title('power specturm');

% 直接DFT不会做
% tmp = [];
% w = 2*pi/41;
% for k = 1:41
%     new_tmp = 0;
%     for i=1:41
%        new_tmp = new_tmp + R(i)*exp(-1j*k*i*w);
%     end
%     tmp = [tmp, new_tmp];
% end
% tmp = abs(tmp);
% Gx = 10*log10(tmp);
% subplot(3,2,6)
% plot([1:41], Gx)



