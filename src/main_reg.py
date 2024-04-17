import torch
import torch.nn.functional as F


class VarianceCovarianceRegularizationFunction(torch.autograd.Function):
    # Forward pass
    # We assume the input has zero mean per channel
    # In practice, we apply a batch demean operation before calling the function
    @staticmethod
    def forward(tensor, alpha, beta, epsilon):
        return tensor

    @staticmethod
    def setup_context(ctx, inputs, output):
        tensor, alpha, beta, epsilon = inputs
        ctx.save_for_backward(tensor)
        ctx.alpha = alpha
        ctx.beta = beta
        ctx.epsilon = epsilon

    # Backward pass
    @staticmethod
    def backward(ctx, grad_output):
        if grad_output.isnan().any().item():
            return None, None, None, None
        (input,) = ctx.saved_tensors
        # Reshape the input to have (n, d) shape
        flattened_input = input.flatten(start_dim=0, end_dim=-2)
        n, d = flattened_input.shape
        # Calculate the covariance matrix
        covariance_matrix = torch.mm(flattened_input.t(), flattened_input) / (n - 1)
        # Calculate the gradient
        diagonal = F.threshold(
            torch.rsqrt(covariance_matrix.diagonal() + ctx.epsilon), 1.0, 0.0
        )
        std_grad_input = diagonal * flattened_input
        cov_grad_input = torch.mm(flattened_input, covariance_matrix.fill_diagonal_(0))
        grad_input = (
            grad_output
            - ctx.alpha / (d * (n - 1)) * std_grad_input.view(grad_output.shape)
            + 4 * ctx.beta / (d * (d - 1)) * cov_grad_input.view(grad_output.shape)
        )
        return grad_input, None, None, None


regularize = lambda x: VarianceCovarianceRegularizationFunction.apply(
    x, 0.64, 0.08, 1e-5
)

x = torch.randn(1, requires_grad=True)
x = regularize(x)
y = x / 0.0
y = y / y
y.backward()
print(x.grad)
# tensor([nan])
