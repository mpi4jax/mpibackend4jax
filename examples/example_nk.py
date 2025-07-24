import mpibackend4jax as _mpi4jax  # noqa: F401
import jax

print("Setup initialize", flush=True)
jax.distributed.initialize()
print(f"{jax.process_index()}/{jax.process_count()} :", jax.local_devices())
print(f"{jax.process_index()}/{jax.process_count()} :", jax.devices())


import netket as nk
from netket import experimental as nkx
import optax

# 1D Lattice
L = 20
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)

# Ising spin hamiltonian
ha = nk.operator.Ising(hilbert=hi, graph=g, h=1.0)

# RBM Spin Machine
ma = nk.models.RBM(alpha=1, param_dtype=float)

# Metropolis Local Sampling
sa = nk.sampler.MetropolisLocal(hi, n_chains=16)
print(sa.n_chains, sa.n_chains_per_rank)

# Optimizer with a decreasing learning rate
op = nk.optimizer.Sgd(learning_rate=optax.linear_schedule(0.1, 0.0001, 500))

# Variational state
vs = nk.vqs.MCState(sa, ma, n_samples=1008, n_discard_per_chain=10)

# Variational monte carlo driver with a variational state
gs = nkx.driver.VMC_SR(
    ha,
    op,
    variational_state=vs,
    diag_shift=0.01,
)

# Run the optimization for 500 iterations
gs.run(n_iter=500, out="test", timeit=True)
