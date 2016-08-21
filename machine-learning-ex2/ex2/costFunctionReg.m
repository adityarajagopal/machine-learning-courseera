function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
H = sigmoid(X*theta); 
l1 = log(H); 
l2 = log(ones(m,1) - H); 
a1 = y.'*l1; 
a2 = (ones(m,1)-y).'*l2; 
theta1 = theta(2:end); %so that we don't regularise the theta(1)
J = (-1/m)*(a1+a2) + (lambda/(2*m))*(theta1.'*theta1); 

t1 = H.'*X; 
t2 = y.'*X; 
nr_grad = (1/m)*(t1-t2);
grad = [nr_grad(1),(lambda/m)*(theta1.') + nr_grad(:,2:end)]; 
% =============================================================
end
