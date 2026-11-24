# Studies

| Slot            | Top-down Microarchitecture Analysis                                                            |
| --------------- | ---------------------------------------------------------------------------------------------- |
| Retiring        | [ipc](x86_64/retiring/ipc.ipynb)                                                               |
| Bad speculation | [branch](x86_64/bad_speculation/branch.ipynb), [if_else](x86_64/bad_speculation/if_else.ipynb) |
| Frontend bound  | [icache](x86_64/frontend_bound/icache.ipynb)                                                   |
| Backend bound   | [dcache](x86_64/backend_bound/dcache.ipynb), [dsize](x86_64/backend_bound/dsize.ipynb)         |

## Run

```sh
jupyter lab studies/
```

```sh
find studies/ -name "*.ipynb" -exec papermill {} {} \;
```
