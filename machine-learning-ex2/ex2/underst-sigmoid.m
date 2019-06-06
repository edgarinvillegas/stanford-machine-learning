neg = [2 3 6 8];
pos = [11 12 13 15 17];
hold on;
plot(neg, zeros(1, length(neg)), 'bx');
plot(pos, zeros(1, length(pos)), 'ro');
plot([9,9],[0,1], 'k');
x = linspace(0,20);
plot(x, sigmoid(-9 + x), 'g');
plot(neg, sigmoid(-9 + neg), 'b+');
plot(pos, sigmoid(-9 + pos), 'r*');
hold off;

