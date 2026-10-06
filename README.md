FiLM conditioning of a residual-attention corrector on ensemble-covariance context, for the 3D system in https://arxiv.org/abs/2609.30746.

From the repository root:

    JAX_PLATFORMS=cpu python -m pytest -q
    JAX_PLATFORMS=cpu python demo.py

This is the low-dimensional system only. No quasi-geostrophic run, no STORN, no claim about 99th-percentile exceedance on the paper's trajectories. Widths are smaller than Table 5 (d_hid 16 not 32, context hidden 8 not 48, one layer not two). The tail loss is an exceedance weight, not a copied hyperparameter. OTD modes are not computed; the ensemble leading eigenvector is the conditioning signal the paper uses as a Jacobian-free proxy.

The context network maps the length-8 vector (member mean, member std, two leading covariance eigenvalues) through a tanh hidden layer of width 8, then to gamma and beta in R^16. Hidden width 8 is used instead of the paper's 48 because the backbone is tiny. The backbone projects each coarse state in a length-8 window to d_hid 16, adds sinusoidal positions, and runs one pre-norm transformer block (LayerNorm, 2-head attention, GELU FFN of width 32). FiLM is h_film = (1 + gamma) * h + beta on that encoder output. Heads emit a pre-gate correction delta and a gate g; the gate bias is initialized to -2. The corrected window is z_coarse + g * delta. context=None skips FiLM, which is gamma = 0, beta = 0.

The training loss is L = L_path + 0.01 L_delta + 0.1 L_mom + 0.1 L_tail. These lambdas are not copied from the paper: they are not stated as a single number in the appendix slice used here, and they are kept small so the path term dominates. L_tail = mean((1 + relu(|z_ref| - tau)) * (z_hat - z_ref)^2) with a fixed threshold tau = 1. That is a stand-in for the paper's exceedance-weighted tail term, not a copied hyperparameter.

Reference parameters are a1 = 2, a2 = 2, b = 20, gamma = 1, epsilon = 0.05. The coarse model uses gamma = 15 and the same epsilon. Integration is RK4 with dt = 0.01. Covariance and std use the population convention (divide by M).
