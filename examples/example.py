import mpitrampoline4jax as _mpi4jax  # noqa: F401
import jax

print("Setup initialize", flush=True)
jax.distributed.initialize()
print(f"{jax.process_index()}/{jax.process_count()} :", jax.local_devices())
print(f"{jax.process_index()}/{jax.process_count()} :", jax.devices())

x = jax.numpy.ones(
    (jax.device_count(),),
    device=jax.sharding.NamedSharding(
        jax.sharding.Mesh(jax.devices(), "i"), jax.sharding.PartitionSpec("i")
    ),
)

print(f"{jax.process_index()}/{jax.process_count()} :", x.sum())
