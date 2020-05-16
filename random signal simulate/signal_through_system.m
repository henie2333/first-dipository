% 书3.4
syms o b a positive   % 规定大于0变量
syms delta_t w
R_x = o^2*exp(-b*abs(delta_t));
G_x_w = fourier(R_x, delta_t, w)
H_w = a^2/(w^2+a^2);
G_y = ifourier(G_x_w * (H_w), w, delta_t);
G_y = simplify(G_y)

