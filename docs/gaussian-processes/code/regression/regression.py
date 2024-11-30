# %%
import torch
import gpytorch
import matplotlib.pyplot as plt
import gpytorch.constraints
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch.linalg

# %%
torch.set_default_dtype(torch.float64)
# %%
# --8<-- [start:line_data]
sigt = 0.1
wt = torch.tensor([1.0, 1.0])

def gen_model(w, sig, x):
    x = torch.cat((torch.ones_like(x), x), dim=1)
    f = x @ w
    noise = sig*torch.randn_like(f)
    return f + noise

test_x = torch.linspace(0, 1, 10).reshape(-1, 1)

test_y = gen_model(wt, sigt, test_x)
true_y = gen_model(wt, 0.0, test_x)

line_data = go.Figure()
line_data.add_trace(go.Scatter(x=test_x.ravel(), y=test_y, mode='markers', name='Gen. Model'))
line_data.add_trace(go.Scatter(x=test_x.ravel(), y=true_y, mode='lines', name='True Model'))
line_data.update_layout(xaxis={'title': 'x'}, yaxis={'title': 'y'})
line_data.show()
with open('line_data.json', 'w+') as f:
    f.write(line_data.to_json('line_data.json'))

# --8<-- [end:line_data]

# plt.savefig('line_data.png', bbox_inches='tight')

# %%

# --8<-- [start:posterior_sampling]
test_X = torch.cat((torch.ones_like(test_x), test_x), dim=1)

# Will assume simple covariance matrix, could think about constructing 
# an extremely weak prior
Sigma_p = torch.eye(2)
sigma_n = 0.2
A = sigma_n ** (-2) * test_X.T @ test_X + torch.linalg.inv(Sigma_p)
Ainv = torch.linalg.inv(A)
wb = sigma_n ** (-2) * Ainv @ test_X.T @ test_y

posterior = torch.distributions.MultivariateNormal(loc=wb, precision_matrix=A)

w_range = torch.linspace(0.5, 1.5, 100)
W0, W1 = torch.meshgrid(w_range, w_range, indexing='xy')
F = torch.exp(posterior.log_prob(torch.dstack((W0, W1)).view(-1, 2)).view_as(W0))

# %%

fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.05)

fig.add_trace(go.Contour(x=w_range, y=w_range, z=F), row=1, col=1)
fig.add_trace(go.Scatter(x=test_x.ravel(), y=test_y, mode='markers'), row=1, col=2)
fig.add_trace(
    go.Scatter(
        x=test_x.ravel(),
        y=true_y,
        mode='lines',
        line=dict(color=px.colors.qualitative.D3[0]),
    ),
    row=1,
    col=2,
)

for _ in range(20):
    fig.add_trace(
        go.Scatter(
            x=test_x.ravel(),
            y=gen_model(posterior.sample(), 0.0, test_x),
            mode='lines',
            opacity=0.2,
            line=dict(color=px.colors.qualitative.D3[1]),
        ),
        row=1,
        col=2,
    )

fig.update_layout(showlegend=False, margin=dict(l=30, r=30, t=20, b=20))
fig.show()
# --8<-- [end:posterior_sampling]
with open('posterior_sampling.json', 'w+') as f:
    f.write(fig.to_json())

# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 6), layout='tight')
axs[0].contourf(W0, W1, F)
axs[1].scatter(test_x, test_y)
axs[1].plot(test_x, true_y, color='k', zorder=-1)

for _ in range(20):
    axs[1].plot(test_x, gen_model(posterior.sample(), 0.0, test_x), linewidth=0.5, color='tab:orange')

axs[1].plot(test_x, gen_model(posterior.mean, 0.0, test_x))
# %%
# --8<-- [start:posterior_sampling2]
sigt = 0.1
wt = torch.tensor([1.0, 1.0])
test_X = torch.cat((torch.ones_like(test_x), test_x), dim=1)

Sigma_p = torch.eye(2)
sigma_n = 0.01
A = sigma_n ** (-2) * test_X.T @ test_X + torch.linalg.inv(Sigma_p)
Ainv = torch.linalg.inv(A)
wb = sigma_n ** (-2) * Ainv @ test_X.T @ test_y

posterior = torch.distributions.MultivariateNormal(loc=wb, precision_matrix=A)

w0_range = torch.linspace(0.95, 1., 100)
w1_range = torch.linspace(1, 1.1, 100)
W0, W1 = torch.meshgrid(w0_range, w1_range, indexing='xy')

F = torch.exp(posterior.log_prob(torch.dstack((W0, W1)).view(-1, 2)).view_as(W0))

fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.05)

fig.add_trace(go.Contour(x=w0_range, y=w1_range, z=F), row=1, col=1)
fig.add_trace(go.Scatter(x=test_x.ravel(), y=test_y, mode='markers'), row=1, col=2)
fig.add_trace(
    go.Scatter(
        x=test_x.ravel(),
        y=true_y,
        mode='lines',
        line=dict(color=px.colors.qualitative.D3[0]),
    ),
    row=1,
    col=2,
)

for _ in range(20):
    fig.add_trace(
        go.Scatter(
            x=test_x.ravel(),
            y=gen_model(posterior.sample(), 0.0, test_x),
            mode='lines',
            opacity=0.2,
            line=dict(color=px.colors.qualitative.D3[1]),
        ),
        row=1,
        col=2,
    )

fig.update_layout(showlegend=False, margin=dict(l=30, r=30, t=20, b=20))
fig.show()
# --8<-- [end:posterior_sampling2]
with open('posterior_sampling2.json', 'w+') as f:
    f.write(fig.to_json())


# %%
