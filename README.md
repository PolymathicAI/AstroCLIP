# AstroCLIP


## Install
Note that an up-to-date eventlet is required for wandb (again old version on rusty)
The following packages are excluded from the project's dependencies to allow for a more flexible system configuration (i.e. allow the use of module subsystem).

```bash
pip install --upgrade pip
pip install --upgrade eventlet torch lightning[extra]
pip install -e .
```
