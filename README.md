# PropHunt

PropHunt is a tool for automated optimization of syndrome measurement circuits of CSS codes. 

## Setup
The recommended method is to use the provided Docker image. This also includes building of the [`loandra`](https://github.com/jezberg/loandra) MaxSAT solver. The Docker image can be built by running `make docker`.

If built without Docker, the paths in `prop_hunt/ambiguous_error.py` will need to be updated to match the `loandra` install location.

## Usage
Example usage can be seen in `scripts/prophunt_experiment.py`. The simplest method is to use `prop_hunt.prop_graph.prop_graph_from_code` with a `CSSCode` from the [`qLDPC` repository](https://github.com/qLDPCOrg/qLDPC) and construct a `PropHuntCompiler` using the resulting `PropagationGraph`. Estimated effective distances are output after each optimization iteration. 

