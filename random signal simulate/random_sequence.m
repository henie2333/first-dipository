% ��������ģ�Ͳ���һ���������:?x(n)=0.8x(n-1)+w(n)������w(n)Ϊ��ֵΪ0������Ϊ4�ĸ�˹����������
% ��1��ģ�����X(n)���е�500?�۲������������������ͼ��?
% ��2���ù۲������źŵľ�ֵ�ͷ��?
% ��3�����Ƹù��̵�����غ����͹������ܶȣ�������ͼ�Ρ�

% ���ڳ弤����h��n��=1/��1-0.8z**-1��
b = [1];a = [1 -0.8];
[h, t] = impz(b,a,50);


% w(n)Ϊ��ֵΪ0������Ϊ4�ĸ�˹����������
w = normrnd(0,2,1,200);

% x = w���h
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


% ������غ���
Mlag = 20;
R = xcorr(x, Mlag, 'coeff');
m = -Mlag:Mlag;
subplot(3,2,5);
plot(m, R);
title('Rx function')
grid on


% ������
Fs = 1000;  % ������
nfft = 1024;    % fft�������ݵ���
CXk=fft(R,nfft);
Pxx=abs(CXk);
index=0:round(nfft/2-1);
plot_Pxx=10*log10(Pxx(index+1));
f = index*Fs/nfft;  % Ƶ��
subplot(3,2,6);
plot(f, plot_Pxx);
title('power specturm');

% ֱ��DFT������
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



