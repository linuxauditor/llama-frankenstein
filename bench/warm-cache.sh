#!/usr/bin/env bash
set -euo pipefail

URL="${URL:-http://localhost:8081/v1/chat/completions}"
MODEL="${MODEL:-qwen36-35b-a3b-mtp-q6}"
RUNS="${RUNS:-20}"
TOKENS="${TOKENS:-1000}"

PROMPTS=(
  "Explain the Linux kernel scheduler, including CFS, run queues, CPU affinity, NUMA effects, and context switching."
  "Explain photosynthesis and the Calvin cycle at an undergraduate biochemistry level, including ATP and NADPH."
  "Write a detailed history of the Roman Republic from the Punic Wars through the rise of Augustus."
  "Explain transformer inference, KV caches, grouped-query attention, RoPE, RMSNorm, and memory bandwidth bottlenecks."
  "Design a PostgreSQL database schema for a global logistics company handling shipments, warehouses, customs, and invoicing."
  "Explain general relativity, spacetime curvature, gravitational time dilation, black holes, and gravitational waves."
  "Write a technical analysis of modern aircraft turbofan engines, including compressor stages, bypass ratio, combustion, and turbine cooling."
  "Explain monetary policy, inflation targeting, government bond markets, yield curves, and quantitative easing."
  "Describe how TLS 1.3 works from TCP connection establishment through certificate validation, key exchange, and encrypted application traffic."
  "Explain human immune response including innate immunity, adaptive immunity, antibodies, T cells, and immunological memory."
  "Design a distributed object storage system. Discuss replication, erasure coding, consistency, failure recovery, and metadata."
  "Explain plate tectonics, subduction zones, mantle convection, earthquakes, volcanism, and continental drift."
  "Write a detailed comparison of Renaissance, Baroque, Romantic, and Impressionist art and their historical contexts."
  "Explain compiler design from lexical analysis through parsing, intermediate representations, optimization, register allocation, and machine code."
  "Describe the chemistry and metallurgy of steel production, including blast furnaces, alloying, heat treatment, and phase transformations."
  "Explain TCP congestion control, retransmission, receive windows, BBR, packet loss, latency, and high-bandwidth networks."
  "Explain DNA replication, transcription, translation, gene regulation, mutation, and modern genome sequencing."
  "Design the electrical power system for a data center, covering utility feeds, transformers, UPS systems, generators, PDUs, and redundancy."
  "Explain how international maritime shipping works, including container logistics, ports, bills of lading, freight rates, and customs."
  "Explain reinforcement learning including Markov decision processes, value functions, policy gradients, exploration, and reward shaping."
)

printf "%-5s %-10s %-10s %-10s\n" "RUN" "TOK/S" "ACCEPT" "DRAFT"
printf "%-5s %-10s %-10s %-10s\n" "---" "-----" "------" "-----"

for ((i=1; i<=RUNS; i++)); do
    prompt="${PROMPTS[RANDOM % ${#PROMPTS[@]}]}"
    outfile="/tmp/frankenstein-bench-${i}.json"

    jq -n \
      --arg model "$MODEL" \
      --arg prompt "$prompt" \
      --argjson tokens "$TOKENS" \
      '{
        model: $model,
        messages: [{role:"user", content:$prompt}],
        max_tokens: $tokens,
        temperature: 0.6,
        stream: false
      }' |
    curl -sf "$URL" \
      -H 'Content-Type: application/json' \
      --data-binary @- > "$outfile"

    python3 - "$i" "$outfile" <<'PY'
import json
import sys

run = int(sys.argv[1])

with open(sys.argv[2]) as f:
    x = json.load(f)

t = x["timings"]

speed = t["predicted_per_second"]
draft = t.get("draft_n", 0)
accepted = t.get("draft_n_accepted", 0)
acceptance = 100 * accepted / draft if draft else 0

print(f"{run:<5} {speed:<10.2f} {acceptance:<9.1f}% {draft:<10}")
PY
done
