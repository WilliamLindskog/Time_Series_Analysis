function e_hat = modFilt(c, a, y)
e_hat = filter(c, a, y);
e_hat = e_hat(length(c):end);
end