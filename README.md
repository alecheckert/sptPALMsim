# sptPALMsim (`sps`)
An optical-dynamic simulator for experiments supporting a state array paper

## Install

Clone the source and install with `pip`:

```
    git clone https://github.com/alecheckert/sptPALMsim.git
    cd sptPALMsim
    pip install .
```

If you use `conda`, an environment compatible with `sps`
is included in `sptsimenv.yml`.

## GPU acceleration

`sps` can use CUDA-enabled GPUs to accelerate the optics simulation.
This requires `cupy`, which you can get via `conda`:
```
    conda install cupy
```
