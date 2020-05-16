N = 2048;           % �趨��������2048����
B = 20000;          % �趨����20000Hz��
k = 1;              % �趨�������ܶ�1��
f = 0-B/2:B/N:B-B/N-B/2;    % Ƶ���б�

% �˴��õ��ĺ����൱��ԭ�������Դ����ţ�����
% ��˵õ��ĸ���Ƶ�׺���Ҳ�൱��Ƶ�������������
% ���������ź����ı�ԵЧӦ��ʹ��Ե�źŵ�Ȩ�ؽ���
% ����������ִ������������ɴ˴�����Ӱ�죨��hanmming��hann�ȣ�

noise = randn(size(f))*(k*B)^0.5;    
% �˴����ڴ��������Ҵ�ͨ���ͨ��˹�źŹ��ʵ��ڷ��������=����*�������ܶȣ��ɴ˲����˾����ض������׵ĸ�˹����ź�
subplot(2,2,1); hist(noise, 50);title('�����ĸ��ʷֲ�');

% ����ͼ��
Nf = fft(noise,N);
GN = power(abs(Nf), 2)/N;             % ����������= ��1/N��*N��w��^2

subplot(2,2,2); plot(f,GN);title('����ͼ��������');xlabel('f/Hz');
fprintf('����ͼ���������ܶ�׼ȷ��Ϊ��%f\n', (mean(GN)/B)/k);

% ����غ�����
noise_corr = xcorr(noise);
Gn = abs(fft(noise_corr));
index = 0:round(length(Gn)/2-1);
Gn = Gn(index+1)/N;

subplot(2,2,3);plot(f,Gn);title('����غ�����������');xlabel('f/Hz');
fprintf('����غ������������ܶ�׼ȷ��Ϊ��%f\n', (mean(Gn)/B)/k);


% ���Ծ�
window = hann(length(f));
G = periodogram(noise,window, N);

subplot(2,2,4);plot(G);title('������������');xlabel('f/Hz');
fprintf('���������ܶ�׼ȷ��Ϊ��%f\n', (mean(G)/B)/k);





