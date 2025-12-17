function [acc] = accuracy(tr_p, Xtr, ytr, w_star,y)
    % Initialize accuracy counter
    correct_count = 0;

    % Calculate accuracy
    for j = 1:tr_p
        % Predict the label using the neural network function y
        prediction = round(y(Xtr(:, j), w_star));

        % Check if the prediction matches the true label
        if prediction == ytr(j)
            correct_count = correct_count + 1;
        end
    end

    % Compute accuracy as a percentage
    acc = (correct_count / tr_p) * 100;
end
