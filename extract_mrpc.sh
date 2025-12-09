#!/bin/bash
uv run python src/extract_act_main.py --config-name="qwen2.5-0.5b_mrpc.yaml"
uv run python src/extract_act_main.py --config-name="qwen2.5-1.5b_mrpc.yaml"
uv run python src/extract_act_main.py --config-name="qwen2.5-3b_mrpc.yaml"
uv run python src/extract_act_main.py --config-name="qwen3-0.6b-base_mrpc.yaml"
uv run python src/extract_act_main.py --config-name="phi-1_mrpc.yaml"
uv run python src/extract_act_main.py --config-name="phi-1_5_mrpc.yaml"
uv run python src/extract_act_main.py --config-name="phi-2_mrpc.yaml"
uv run python src/extract_act_main.py --config-name="open_llama_3b_mrpc.yaml"