fs = 30000;
T = 0.005;
t = 0:1/fs:T;
f_a = 1000;
w_a = 2*pi*f_a;

x1 = randn(size(t));
x2 = randn(size(t));        %x1��x2�޹�
f = 0:f_a/(length(t)-1):f_a;

f_s = 100;      % ��Ƶ�ز�
ht = w_a*exp(-w_a*t);   % ��ͨ�˲���
As = conv(x1,ht);   % ������ͨ�˲���ĸ�˹�ź�
As = T*As(1:length(t)).*cos(f_s*t);      % ����ز�
Gs = fft(As,length(t));
Gs = power(abs(Gs),2)/length(t);
subplot(2,2,1);
plot(t*1000,As);title('As(t)');xlabel('ms');

Ac = conv(x2,ht);
Ac = T*Ac(1:length(t)).*sin(f_s*t);
Gc = fft(Ac,length(t));
Gc = power(abs(Gc),2)/length(t);
subplot(2,2,2)
plot(t*1000,Ac);title('Ac(t)');xlabel('ms');

f0 = 10000;     % ��Ƶ�ĵ����ź�
y =  Ac.*cos(2*pi*f0*t) - As.*sin(2*pi*f0*t);

Gs = fft(y,length(t));
Gs = power(abs(Gs),2)/length(t);    % ����G(w) = 1/N*��f��w��^2��
subplot(2,2,3)
plot(Gs);title('խ����˹����y������');xlabel('w')
subplot(2,2,4)
plot(t*1000,real(y));title('y = Ac(t)*cos(w0*t)-As(t)*sin(w0*t)');xlabel('ms');

    


