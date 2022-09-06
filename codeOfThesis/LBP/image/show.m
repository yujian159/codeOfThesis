imgL = zeros(112 * 10, 92 * 40);

for i = 1:10
    for j = 1:40
        img = imread(['.\s' num2str(j) '\' num2str(i) '.pgm']);
        imgL(112 * (i-1) + 1:112 * i, 92 * (j-1) + 1:92 * j) = img;
    end
end

imshow(uint8(imgL));