# Studies

| Slot | Top-down Microarchitecture Analysis |
| --- | --- |
| Retiring | [retiring/ipc](x86_64/retiring/ipc.ipynb) |
| Bad speculation | [bad_speculation/branch](x86_64/bad_speculation/branch.ipynb), [bad_speculation/if_else](x86_64/bad_speculation/if_else.ipynb) |
| Frontend bound | [frontend_bound/icache](x86_64/frontend_bound/icache.ipynb) |
| Backend bound | [backend_bound/dcache](x86_64/backend_bound/dcache.ipynb), [backend_bound/dsize](x86_64/backend_bound/dsize.ipynb) |

## Run

```sh
jupyter lab studies/
```

```sh
find studies/ -name "*.ipynb" -exec papermill {} {} \;
```
