# llama-frankenstein

Experimental `llama.cpp` fork for running large MoE models on consumer AMD GPUs.

This combines the MoE expert offload/cache work from Miltos's `llama-wackMall` with some additional work I did around MTP speculative decoding, expert caching and ROCm.

The setup I have spent most time testing is:

```text id="4fdd30"
Qwen3.6-35B-A3B Q6_K
RX 6800 XT 16 GB
Ryzen 9 7940HS
DDR5
ROCm 7.2
```

After the expert cache warms up, this runs at around **39 tok/s sustained**, with individual runs reaching **~42 tok/s**.

Not bad for a 6800 XT.

## Where this came from

The underlying MoE expert offload/cache implementation is from **Miltos / llama-wackMall**.

That is where the work for moving experts between system RAM and GPU memory, tracking hot experts and managing GPU residency comes from.

This fork adds my experiments on top of that, mainly:

- a modified MTP speculative decoding path
- accept-only MTP catch-up
- MTP profitability tracking
- expert-cache/routing experiments
- timing and instrumentation
- a fair amount of AMD/ROCm testing and tuning

If you are interested in the expert offload implementation itself, start with Miltos's project.

This branch came from the older experimental `llama.cpp-wackMall-merge-request` tree rather than the current `llama-wackMall` main branch.

## Why the RX 6800 XT?

The 6800 XT is not an AI card.

It is RDNA2. There are no modern AI-oriented matrix paths, no FP8 hardware and none of the newer low-precision acceleration that makes current GPUs much better suited to LLM inference.

What it does have is **16 GB of VRAM and roughly 512 GB/s of memory bandwidth**.

For LLM inference, memory bandwidth matters a lot.

For MoE it gets even more interesting. The entire model does not need to be active for every token. If the frequently used experts can stay in VRAM while colder experts live in system RAM, the card can do considerably more than its 16 GB capacity would suggest.

The 6800 XT is also old enough that used cards can be found relatively cheaply.

So while the hardware paths are definitely not AI-friendly, there is still quite a lot of raw memory bandwidth available for the money.

That is basically why I started experimenting with this.

## MoE expert caching

A Mixture-of-Experts model contains many experts but only activates some of them for each token.

Instead of trying to fit every expert into VRAM, the cache keeps a limited number of hot experts on the GPU and leaves colder experts in system RAM.

The routing statistics are tracked while the model runs and expert placement changes over time.

My current configuration uses:

```text id="tr9v0p"
hot expert slots:       96
expert sync period:     32
hysteresis:             1.5
dwell:                  32
heat decay:             0.9995
expert move mode:       1
expert pin:             0
expert sidecar:         enabled
```

The important thing to understand is that the cache needs to warm up.

If you start the server and immediately benchmark it, the result will be bad.

On my current setup the first request was around **26 tok/s**. After several requests across different domains, the same server was running around **39–42 tok/s**.

Nothing was restarted or reconfigured between those tests. The expert placement simply got better.

## MTP

The other half of this setup is MTP speculative decoding.

This is not just stock MTP enabled on top of the expert cache.

I changed how the MTP context is updated during speculative decoding.

The basic flow looks like this:

```text id="3fx2r6"
MTP drafts tokens
       |
       v
target verifies them
       |
   +---+---+
   |       |
accepted  rejected
   |       |
   v       v
commit   discard
   |
   v
catch MTP up to
committed state
```

The important part is the last step.

During generation, the verification batch is temporarily kept around instead of immediately advancing the MTP context with all of the speculative rows.

Once we know what the target model actually accepted, only the committed prefix is replayed into the MTP context.

Rejected speculative rows never become part of the persistent MTP state.

Enable this with:

```bash id="gph2xd"
export LLAMA_MTP_ACCEPT_ONLY_CATCHUP=1
```

The implementation is currently deliberately narrow. It was written for the setup I was actually using:

- one sequence
- non-shared memory
- one reusable MTP head
- Qwen-style MTP

I would rather keep the known-good path narrow than pretend this has been tested against every possible configuration.

## MTP profitability

There is also an optional kill threshold:

```bash id="r8bbam"
export LLAMA_MTP_KILL_ACCEPT=<ratio>
```

Every 64 speculative cycles the code can check how many draft tokens are actually being accepted.

If MTP is doing badly enough, it can be disabled for the rest of that request instead of continuing to spend compute maintaining a speculative path that is not helping.

I added this because it seemed sensible while experimenting.

In practice, I have not needed it.

Acceptance on Qwen3.6 has generally been good enough that I never actually hit the kill threshold during normal testing.

## Hardware

The best results so far have been on:

```text id="aw8kr1"
CPU:       AMD Ryzen 9 7940HS
Memory:    DDR5 SODIMM
GPU:       AMD Radeon RX 6800 XT
VRAM:      16 GB
Bandwidth: ~512 GB/s
GPU arch:  gfx1030
ROCm:      7.2
OS:        Ubuntu 24.04 VM
```

The GPU is passed through to the VM.

Host memory matters here.

A significant amount of model data is coming from system RAM, so this is one of those workloads where changing the CPU/memory platform can change the result even though the GPU is identical.

I previously ran this sort of setup on an older DDR4 platform. The 7940HS/DDR5 machine has been noticeably better for it.

## Model

The model used for the clean reproduction was:

```text id="k2hlg5"
Qwen3.6-35B-A3B-MTP-UD-Q6_K.gguf
```

`llama-server` reports:

```text id="7ofshk"
Parameters:        35.5B
Quantization:      Q6_K
GGUF size:         ~30 GB
Configured ctx:    98,304
Training ctx:      262,144
```

Qwen3.6-35B-A3B has been a particularly good fit for this experiment.

The model is much larger than the GPU, but because it is MoE the working set is a different problem from trying to run a similarly sized dense model.

## Known-good branch

This is the version I have now rebuilt and tested from scratch:

```text id="sxwlzy"
Branch:
frankenstein-6800xt

Commit:
cf48791c2056071982d1f9e8af088b6c6ef09f40

Tag:
rx6800xt-known-good-20260823
```

Use the tag if you want to reproduce my results:

```bash id="dvazlh"
git clone https://github.com/linuxauditor/llama-frankenstein.git
cd llama-frankenstein

git checkout rx6800xt-known-good-20260823
```

## Building

This was tested from a completely fresh checkout with ROCm 7.2:

```bash id="pgk2z7"
cmake -B build \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS=gfx1030

cmake --build build -j16
```

The resulting binary correctly picked up the ROCm backend:

```text id="jgy84r"
Available devices:
  ROCm0: AMD Radeon RX 6800 XT (16368 MiB, 16342 MiB free)
```

The original container build also used:

```text id="6rlc5v"
-DGGML_BACKEND_DL=OFF
-DLLAMA_BUILD_TESTS=OFF
```

Those were not required for the clean rebuild.

## Running it

Environment:

```bash id="w24ut3"
export LLAMA_EXPERT_HOT_FORCE=1
export LLAMA_MTP_ACCEPT_ONLY_CATCHUP=1

export HSA_OVERRIDE_GFX_VERSION=10.3.0
export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0
```

My current known-good launch command:

```bash id="y6c1c7"
./build/bin/llama-server \
  -m /path/to/Qwen3.6-35B-A3B-MTP-UD-Q6_K.gguf \
  --host 0.0.0.0 \
  --port 8081 \
  --alias qwen36-35b-a3b-mtp-q6 \
  -c 98304 \
  --parallel 1 \
  --cont-batching \
  --batch-size 4096 \
  --ubatch-size 1024 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --cache-ram 2560 \
  --cache-idle-slots \
  --ctx-checkpoints 8 \
  --checkpoint-min-step 8192 \
  --split-mode none \
  --main-gpu 0 \
  --fit off \
  -ot '^blk\.40\.ffn_(gate|down|up)_exps\.weight$=ROCm0' \
  -ehs 96 \
  --expert-move-mode 1 \
  --expert-pin 0 \
  --expert-sidecar \
  --expert-sync-period 32 \
  --expert-hyst 1.5 \
  --expert-dwell 32 \
  --expert-heat-decay 0.9995 \
  --expert-heat-log-period 128 \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  --spec-draft-p-min 0 \
  --no-spec-draft-backend-sampling \
  --temp 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.0 \
  --presence-penalty 0.0 \
  --frequency-penalty 0.0 \
  --repeat-penalty 1.0 \
  -lv 2
```

There are probably still knobs here that can be improved.

This is simply the configuration I know works.

The block 40 override:

```text id="gzlt7d"
-ot '^blk\.40\.ffn_(gate|down|up)_exps\.weight$=ROCm0'
```

keeps those expert tensors on the GPU.

The hot expert cache is:

```text id="t3uxxg"
-ehs 96
```

## Performance

I wanted to make sure I had not preserved a branch that only worked because of something left over in my old build tree.

So I did a clean test:

1. cloned the public repository into a new directory
2. checked out `rx6800xt-known-good-20260823`
3. built it from scratch
4. loaded the model
5. ran a set of prompts from different domains

The server started with an empty expert cache.

Results:

```text id="d5euy1"
RUN   TOK/S    MTP ACCEPT
1     25.75      73.5%
2     30.68      67.5%
3     34.41      70.4%
4     36.59      82.3%
5     40.67      70.6%
6     39.55      66.4%
7     39.04      78.8%
8     38.76      73.1%
9     39.43      71.6%
10    39.85      74.3%
11    41.67      77.8%
12    36.11      64.6%
13    36.85      66.0%
14    41.89      69.2%
15    38.66      77.6%
16    37.60      70.8%
17    38.78      65.0%
18    38.26      74.5%
19    39.18      66.2%
20    37.67      59.2%
```

Runs 5–20 average about:

```text id="v7q31r"
Decode:          ~39.0 tok/s
MTP acceptance:  ~70%
Peak:             41.89 tok/s
```

That is the number I would use for this setup:

**~39 tok/s sustained, ~42 tok/s observed peak.**

## Cold cache vs warm cache

This matters enough to have its own section.

Look at the first five runs:

```text id="n9k5a4"
25.75
30.68
34.41
36.59
40.67 tok/s
```

The server did not change.

The MTP acceptance was already **73.5% on the first run**, so MTP was not suddenly starting to work five requests later.

The expert cache was warming up.

By sending prompts from different domains, the routing statistics build up and the cache gets a much better idea of which experts are worth keeping in VRAM.

If you benchmark this immediately after startup, you are mostly benchmarking a cold expert cache.

## Benchmark script

`bench/warm-cache.sh` sends sequential requests from different subject areas.

That is intentional.

Sending the same prompt twenty times would give the cache a much narrower routing workload. I wanted something that would move between software, biology, history, networking, economics, physics and other unrelated topics.

Run it with:

```bash id="f8vyj1"
RUNS=20 TOKENS=1000 ./bench/warm-cache.sh
```

I would suggest warming the cache before doing any serious performance comparison.

Concurrency is a separate test. The numbers above are for a single sequence.

## MTP depth

I settled on:

```text id="4mp2f9"
--spec-draft-n-max 2
```

Earlier testing showed MTP depth 2 beating depth 1 by roughly 26% in a matched test, even though its acceptance rate was lower.

That was a useful reminder not to optimize this stuff around one counter.

Higher acceptance is good, but what I actually care about is how many useful tokens come out of the server per second.

The same applies to the expert cache. The highest reported hot-expert percentage was not always the fastest configuration.

There are several things fighting each other here:

- expert reuse
- RAM traffic
- VRAM residency
- expert movement
- synchronization
- MTP acceptance
- speculative overhead
- GPU compute

The final tok/s number wins.

## About the 6800 XT result

I think the hardware is part of what makes this experiment interesting.

A 6800 XT has no business being described as an AI accelerator.

It predates the current rush toward dedicated low-precision AI hardware, has only 16 GB of VRAM, and does not have the hardware paths that newer cards use for formats such as FP8.

But it does have a wide memory subsystem and around **512 GB/s of VRAM bandwidth**.

LLM inference is often heavily constrained by moving weights rather than pure arithmetic throughput. MoE makes that relationship even more interesting because only part of the model is needed for each token.

With the right model and enough system memory behind it, the card is still surprisingly useful.

The model here is about **30 GB at Q6_K**, running on a GPU with **16 GB of VRAM**, and after the expert cache warms up it sits around **39 tok/s** with peaks around **42 tok/s**.

For the amount of money these cards now cost, I think that is a pretty good result.

It is not competing with modern dedicated AI hardware on features.

It does not need to.

The interesting part is how much useful inference performance can still be extracted from relatively cheap, older consumer hardware when the workload is arranged around what the card is actually good at.

## Determinism

The accept-only path keeps rejected speculative rows out of persistent MTP state.

In other words, the MTP context follows the sequence the target model actually committed.

That was important to me when implementing it.

I am **not** claiming general bit-for-bit determinism across every backend, concurrency level and configuration. I have not tested enough combinations to make that claim.

## Caveats

This is experimental code built around hardware and models I actually own.

Things to keep in mind:

- the MTP path was written around Qwen-style MTP
- the known-good setup is single-sequence
- host RAM performance matters
- the expert cache needs to warm up
- ROCm behavior varies by GPU and version
- other models may route experts very differently
- the best MTP acceptance is not necessarily the fastest configuration
- the highest hot-expert percentage is not necessarily the fastest configuration
- concurrency needs to be tested separately

If you try this on different hardware, measure it rather than assuming my tuning values are optimal.

## License

This repository contains code from multiple upstream sources.

The Miltos expert offload/cache additions retain their applicable Apache 2.0 licensing terms.

Code inherited from `llama.cpp` retains its original MIT notices and licensing.

See the source headers and license files for the applicable terms.

## Credits

**ggml-org / llama.cpp**

The inference engine all of this is built on.

**Miltos / llama-wackMall**

The MoE expert offloading, caching and tiering work this fork builds on.

Without that work I would not have had the interesting part to start experimenting with.
