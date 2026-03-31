# Using the Stage Mapper

The stage mapper assigns backends and inference parameters to stages before execution. When you set a backend on each stage manually, you are doing explicit mapping. The stage mapper formalizes this as an interface so you can implement your own assignment logic and validate that every stage has a backend before the workflow runs.

## Explicit mapping

The built-in `ExplicitStageMapping` checks that every stage already has a backend. It does not reassign anything. This is useful as a validation step before executing a workflow:

```python
from pyorla import Stage, ExplicitStageMapping, StageMappingInput

light = new_sglang_backend("Qwen/Qwen3-4B", "http://sglang:30000/v1")
heavy = new_sglang_backend("Qwen/Qwen3-32B", "http://sglang:30001/v1")

classify = Stage("classify", light)
solve = Stage("solve", heavy)

mapping = ExplicitStageMapping()
result = mapping.map(StageMappingInput(
    stages=[classify, solve],
    backends=[light, heavy],
))
```

If any stage is missing a backend, `ExplicitStageMapping` raises a `ValueError` with the stage name. This catches configuration errors before they become runtime failures.

## Applying mapping output

The mapping output contains a `StageAssignment` per stage with the backend and optional inference parameters. Apply it to your stages with `apply_stage_mapping_output`:

```python
from pyorla import apply_stage_mapping_output

apply_stage_mapping_output([classify, solve], result)
```

This writes the assigned backend, max_tokens, temperature, top_p, and response_format back onto each stage. Fields that are `None` in the assignment are left unchanged on the stage.

## Writing a custom stage mapper

Subclass `StageMapping` and implement the `map` method. The input gives you the list of stages and available backends. Return a `StageMappingOutput` with an assignment per stage.

A mapper that assigns stages by name convention:

```python
from pyorla import StageMapping, StageMappingInput, StageMappingOutput, StageAssignment

class NameBasedMapping(StageMapping):
    def __init__(self, light_backend, heavy_backend, light_prefixes=("classify", "triage")):
        self.light = light_backend
        self.heavy = heavy_backend
        self.light_prefixes = light_prefixes

    def map(self, input: StageMappingInput) -> StageMappingOutput:
        output = StageMappingOutput()
        for stage in input.stages:
            if any(stage.name.startswith(p) for p in self.light_prefixes):
                output.assignments[stage.id] = StageAssignment(
                    backend=self.light,
                    max_tokens=256,
                )
            else:
                output.assignments[stage.id] = StageAssignment(
                    backend=self.heavy,
                    max_tokens=1024,
                )
        return output
```

Use it the same way:

```python
mapper = NameBasedMapping(light, heavy)
result = mapper.map(StageMappingInput(stages=[classify, solve], backends=[light, heavy]))
apply_stage_mapping_output([classify, solve], result)
```

A mapper that picks the cheapest backend meeting a quality threshold:

```python
class CostAwareMapping(StageMapping):
    def __init__(self, quality_floors: dict[str, float]):
        self.quality_floors = quality_floors

    def map(self, input: StageMappingInput) -> StageMappingOutput:
        output = StageMappingOutput()
        for stage in input.stages:
            floor = self.quality_floors.get(stage.name, 0.0)
            candidates = [
                b for b in input.backends
                if b.quality is not None and b.quality >= floor and b.cost_model is not None
            ]
            if not candidates:
                candidates = input.backends
            cheapest = min(candidates, key=lambda b: b.cost_model.output_cost_per_mtoken if b.cost_model else float("inf"))
            output.assignments[stage.id] = StageAssignment(backend=cheapest)
        return output
```

This gives you the same cost-aware routing that Orla's built-in accuracy policies provide, but at the mapping level before execution rather than per-request at runtime. Use this when you know the quality requirements upfront and want to lock in backend assignments for the entire workflow.

## When to use stage mapping

Stage mapping runs once before execution. It is useful when:

- You want to validate that every stage has a backend before running a workflow.
- Your assignment logic depends on the full list of stages and backends, not just one stage at a time.
- You want to separate the "which backend" decision from the node logic in your graph.

For per-request decisions that depend on runtime data like triage labels or prompt content, use cost policies or dynamic scheduling hints instead.
