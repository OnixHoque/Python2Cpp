rows = 10000;
columns = 10000;
fraction = 0.0000025; % approx fraction*rows*columns number of non-zeroes
k = 25; % number of matrices
is_double = false; % do you want elements to be double(else it would be int)
infile_name = '1.in';
outfile_name = '1.out';
left = 1;
right = 5;
% values in matrices (before addition) will be from left to right uniformly
% distributred, can also get gaussian distributed by changing sprand
% function to sprandn in the first for loop

% end of parameters to be set by user

sum = sprand(rows, columns, 0);
empty = [];
dlmwrite(infile_name, empty, 'delimiter', ' ')
dlmwrite(outfile_name, empty, 'delimiter', ' ')
dlmwrite(infile_name, k, '-append', 'delimiter', ' ')

for i = 1 : k
    temp = sprand(rows, columns, fraction);
    [row, col, v] = find(temp);
    v = ((right-left)*v) + left;
    if not(is_double)
        v = floor(v);
    else
    end
    temp = sparse(row, col, v, rows, columns);
    sum = sum + temp;
    col_ptr = zeros(1,columns+1);
    current = 1;
    prev = 0;
    counter = 1;
    [row_1, col_1] = size(col);
    for j = 1 : row_1
        if(col(j,1) == prev+1)
            col_ptr(1, current) = counter;
            current = current + 1 ;
            counter = counter + 1 ;
            prev = col(j,1);
        elseif (col(j,1) == prev)
            counter = counter + 1;
        else
            for k = 1 : (col(j,1) - prev)
                col_ptr(1, current) = counter;
                current = current + 1;
            end
            counter = counter + 1;
            prev = col(j,1);
        end
    end
    
    if not(current == columns+1)
        for j = 1: (columns+1-current)
            col_ptr(1, current+j-1) = counter;
        end
    end
    
    col_ptr(1,columns+1) = row_1 + 1;
    
    info = [rows, columns, row_1];
    col_ptr = col_ptr - 1;
    row = row - 1;
    dlmwrite(infile_name, info, '-append', 'delimiter', ' ')
    dlmwrite(infile_name, col_ptr , '-append', 'delimiter', ' ')
    dlmwrite(infile_name, row.' , '-append', 'delimiter', ' ')
    dlmwrite(infile_name, v.' , '-append', 'delimiter', ' ')
    
end

% appending sum to outfile

    [row, col, v] = find(sum);
    col_ptr = zeros(1,columns+1);
    current_sum = 1;
    prev = 0;
    counter_sum = 1;
    [row_1, col_1] = size(col);
    for j = 1 : row_1
        if(col(j,1) == prev+1)
            col_ptr(1, current_sum) = counter_sum;
            current_sum = current_sum + 1 ;
            counter_sum = counter_sum + 1 ;
            prev = col(j,1);
        elseif (col(j,1) == prev)
            counter_sum = counter_sum + 1;
        else
            for k = 1 : (col(j,1) - prev)
                col_ptr(1, current_sum) = counter_sum;
                current_sum = current_sum + 1;
            end
            counter_sum = counter_sum + 1;
            prev = col(j,1);
        end
    end
    
    if not(current_sum == columns+1)
        for j = 1: (columns+1-current_sum)
            col_ptr(1, current_sum+j-1) = counter_sum;
        end
    end
    
    col_ptr(1,columns+1) = row_1 + 1;
    
    info = [rows, columns, row_1];
    col_ptr = col_ptr - 1;
    row = row - 1;
    dlmwrite(outfile_name, info, '-append', 'delimiter', ' ')
    dlmwrite(outfile_name, col_ptr , '-append', 'delimiter', ' ')
    dlmwrite(outfile_name, row.' , '-append', 'delimiter', ' ')
    dlmwrite(outfile_name, v.' , '-append', 'delimiter', ' ')

