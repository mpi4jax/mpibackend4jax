import mpitrampoline4jax

import jax
import os
print("CIAO", flush=True)
jax.distributed.initialize("127.0.0.1:50000", 
    int(os.environ["PMI_SIZE"]), 
    int(os.environ["PMI_RANK"]))
print("DELLO", flush=True)
print(f"{jax.process_index()}/{jax.process_count()} :", jax.local_devices())
print(f"{jax.process_index()}/{jax.process_count()} :", jax.devices())
x = jax.numpy.ones((jax.device_count(),), device=jax.sharding.NamedSharding(jax.sharding.Mesh(jax.devices(), "i"), jax.sharding.PartitionSpec("i")))
print(f"{jax.process_index()}/{jax.process_count()} :", x.sum())

