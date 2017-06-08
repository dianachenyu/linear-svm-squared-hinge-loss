# linear-svm-squared-hinge-loss


This code implement linear support vector machine with squared hinge loss.


\begin{equation}
\nabla F(\beta) = \frac{2}{n}\sum_{i=1}^n -y_i x_i\max(0,1-y_i x_i^T \beta)+2\lambda\beta\
\end{equation}