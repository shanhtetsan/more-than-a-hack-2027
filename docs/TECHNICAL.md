# <team_name> — Technical document

## Approach
## Assets used (and dropped)
## Possession rules implemented
Loose balls, challenges, restarts.
## Results
Public 5 minutes (`make score`). 17 possessions: not a ranking signal on its own.
## Limitations
## Reproduction
```bash
uv sync --frozen                 # Python 3.11, exact pins from uv.lock
make submission TEAM=<team_name> # data in ./data or $MTAH_DATA_DIR
make reproduce TEAM=<team_name>  # byte-identical rebuild check
```
