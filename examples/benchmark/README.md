# OpenML Benchmark

Evaluate Scald framework on standard OpenML classification tasks.

## Setup

```bash
uv sync --group benchmarks
```

## Usage

Run all benchmarks:
```bash
uv run python examples/benchmark/amlb.py
```

Run specific tasks:
```bash
uv run python examples/benchmark/amlb.py --tasks Australian credit-g
```

Configure iterations:
```bash
uv run python examples/benchmark/amlb.py --max-iterations 10
```

## Available Tasks

- Australian (146818)
- blood-transfusion (10101)
- car (146821)
- christine (168908)
- cnae-9 (9981)
- credit-g (31)
- dilbert (168909)
- fabert (168910)
- jasmine (168911)
- kc1 (3917)
- kr-vs-kp (3)
- mfeat-factors (12)
- phoneme (9952)
- segment (146822)
- sylvine (168912)
- vehicle (53)
