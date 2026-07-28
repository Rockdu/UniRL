# Qwen-Image FlowGRPO: miles-diffusion reproduction on UniRL — run log

Archive branch for the 2026-07-23/24 reproduction & bisection campaign.
Base commit: `393ff13` (Tencent-Hunyuan/UniRL main). Hardware: 1 node × 8×H200
(`kangrui-h200-qwen`). Env: official uv sglang venv per INSTALL.md, **diffusers
pinned 0.38.0** (0.39 breaks Qwen-Image — see Known Issues).

Changes on this branch:
- `examples/diffusion/qwen_image/qwen_image_flowgrpo_miles_aligned.yaml` — new
  recipe, knob-for-knob aligned with miles-diffusion
  `run-diffusion-grpo-pickscore-5gpu-flowgrpo-aligned.sh` (flow_grpo
  `pickscore_qwenimage` lineage). Full mapping table in `REPRO_NOTES_ZH.md`.
- `unirl/train/lora.py` — LoRA init PEFT-default → `"gaussian"` (miles/flow_grpo
  parity), both inject and reset paths.
- `pyproject.toml` — diffusers capped `<0.39` (0.39 removed
  `QwenImageTransformer2DModel.forward(txt_seq_lens=...)`; UniRL main crashes).

## Headline findings

1. **The miles-aligned config does not learn on UniRL** (flat train reward,
   monotonically declining eval), while the same setting climbs on
   miles-diffusion.
2. Eight-cut bisection localized the anomaly: **true-CFG (guidance 4.0)
   training shows a rise→dip "hump" over rollouts ~30-90** — reproduced in 3
   distinct configs, while every no-CFG config climbs monotonically.
   **[REVISED 2026-07-28]** The long-horizon rerun (#10) shows the dip is a
   TRANSIENT: after ~150 rollouts the CFG run recovers and converges to
   **0.907 mean / 0.937 max by rollout ~1200** — far above the no-CFG plateau
   (0.854). UniRL CFG training is NOT broken; it traverses a long hump that
   short windows (and our 90-rollout verdicts) misread as decay. The remaining
   miles-vs-UniRL delta is climb DYNAMICS (miles climbs without the hump),
   not capability. Curve: `cfg4long_curve_rollout1671.txt` (200-window means:
   0.846 → 0.855 → 0.878 → 0.896 → 0.902 → 0.907 plateau).
3. Exonerated by experiment: LoRA target scope (attn-8 suffices no-CFG), FlowSDE
   kernel, eta 1.2, static sde_indices [3,4], geometry, engine (trainside used
   throughout), anchor (`old_logp_source: rollout` throughout).
4. Exonerated by code audit vs miles: SDE kernel math (both verbatim flow_grpo),
   CFG blend formula + norm correction, CFG negative-branch gradient (both
   fully differentiable), dynamic σ shift.
5. Caveat discovered en route: **~30-step windows are unreliable for CFG runs**
   (rise-then-decay "hump" mimics a climb). CFG verdicts need 90+ rollouts.
   Same-config rerun variance ≈ 0.01 reward (trainside per-step SDE noise uses
   the global RNG), so cross-run level comparisons are noise below that.

## Runs

wandb projects: [`unirl-qwen-image-miles-aligned`](https://wandb.ai/kangrdu/unirl-qwen-image-miles-aligned)
(most runs), [`unirl`](https://wandb.ai/kangrdu/unirl) (two control runs — the
stock recipe hardcodes its project name).

| # | run | wandb | config | verdict |
|---|-----|-------|--------|---------|
| 1 | v1 aligned (attn-8 LoRA) | [1jjy6w3z](https://wandb.ai/kangrdu/unirl-qwen-image-miles-aligned/runs/1jjy6w3z) | this branch's recipe, 4 GPU | ✗ flat 126 rollouts (0.845 band), eval 0.8589→0.8538 declining |
| 2 | v2 aligned (attn+MLP-12 LoRA) | [ht1a560q](https://wandb.ai/kangrdu/unirl-qwen-image-miles-aligned/runs/ht1a560q) | + 4 MLP LoRA targets, 4 GPU | ✗ flat 60 rollouts — LoRA scope not the cause |
| 3 | control r1 (4 GPU) | [f5jux6hf](https://wandb.ai/kangrdu/unirl/runs/f5jux6hf) | stock `qwen_image_dancegrpo` | 10 rollouts, killed by node earlyoom (see Ops) |
| 4 | control r2 (8 GPU) | [iif1at4s](https://wandb.ai/kangrdu/unirl/runs/iif1at4s) | stock, `forward_batch_size=8 micro=4` | ✓ climbs 0.812→0.854 over 80 rollouts |
| 5 | cut: +CFG4 (short) | `bisect_stock_plus_cfg4` | stock + `++sampling.guidance_scale=4.0` | (34 rollouts — verdict later invalidated by the hump caveat) |
| 6 | cut: FlowSDE kernel | [z6zjgnxl](https://wandb.ai/kangrdu/unirl-qwen-image-miles-aligned/runs/z6zjgnxl) | `qwen_image_trainside` as-is | ✓ climbs 0.812→0.845/35 |
| 7 | cut: eta 1.2 | `bisect_flowsde_eta12` | trainside + `++sampling.eta=1.2` | ✓ climbs 0.778→0.832/45 (low start = early-window σ→1 noise blowup, see notes) |
| 8 | cut: SDE triple | `bisect_sde_triple` | + `"++sampling.sde_indices=[3,4]"` | ✓ climbs 0.806→0.848/23 (short — hump caveat applies) |
| 9 | cut: triple + CFG4 | `bisect_sde3_cfg4` | + `++sampling.guidance_scale=4.0` | ✗ hump: peak 0.847 @25-36 → 0.836 @73-84 |
| 10 | cut: CFG4 long rerun | `bisect_cfg4_long` | stock SDE + CFG4, ran to **1671** | hump @37-63, then **recovers: 0.907 mean plateau @1000+, max 0.937** — see REVISED finding 2 |

Launch pattern (single node):

```bash
source .venv-sglang/bin/activate
ray start --head --port=6379 --num-gpus=8 --object-store-memory=50000000000 --include-dashboard=false
RAY_ADDRESS=127.0.0.1:6379 python -m unirl.train_diffusion \
  --config-name=diffusion/qwen_image/<recipe> num_devices=8 \
  ++logging.report_to_wandb=true [overrides per table] \
  ++rollout.forward_batch_size=8 ++stack.micro_batch_size=4
# aligned recipe on 4 GPUs additionally needs:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS_PER_NODE=4 ... ++devices_per_node=4
```

`forward_batch_size=8` / `micro_batch_size=4` are math-neutral perf overrides
(stock `fbs=1` targets 96GB H20; on H200 it leaves ~6× throughput on the table:
17 min/rollout → 1.9 min/rollout at 8 GPU).

## Known issues found upstream (UniRL)

1. **sglang_diffusion × true-CFG is broken**: with a negative branch the grouped
   request path fails (`initial_noise batch dim 8 does not match batch_size=1`,
   `_patches/patch_latent_prep.py` expansion rule doesn't cover CFG). No qwen
   sglang recipe exercises guidance>1 (SD3 classic-CFG via PE recipe works).
2. **diffusers 0.39 breaks Qwen-Image** (`txt_seq_lens` kwarg removed) and the
   uv override `diffusers>=0.38.0` resolves to it. This branch caps `<0.39`.
3. **[REVISED]** ~~true-CFG training decays~~ → true-CFG training traverses a
   ~30-150-rollout hump before climbing to a HIGHER plateau than no-CFG
   (finding 2). Open question downgraded from bug to dynamics: why does miles
   climb hump-free while UniRL dips first (negative-embeds content and
   off-policy-second-update interaction remain the candidates). Practical
   guidance for UniRL CFG users: do not early-stop before ~200 rollouts.

## Ops notes (shared-node H200 devboxes)

- Platform pods run `earlyoom -m 10 --prefer ^(python|ray.*)$` — long trainings
  on shared nodes get SIGKILLed when node memory dips; run a restart watchdog.
- Cap ray object store (`--object-store-memory=50e9`); /dev/shm is 64G.
- `pkill -f` patterns must not overlap the relaunch command line or other runs'
  module paths (two self-inflicted kills during this campaign).

🤖 Assembled with Claude Code; full narrative in `REPRO_NOTES_ZH.md`.

## Appendix: verbatim launch commands

All runs: `cd /scratch/unirl && source .venv-sglang/bin/activate`, fresh
`ray start --head --port=6379 --num-gpus=<N> --object-store-memory=50000000000 --include-dashboard=false`,
`export RAY_ADDRESS=127.0.0.1:6379 WANDB_API_KEY=... REPORT_TO_WANDB=true`.
`P` = `++logging.project_name=unirl-qwen-image-miles-aligned`, `R` = `++logging.run_name=`.

```bash
# 1 v1 aligned (recipe was then attn-8 = *_v1_attn8.yaml)
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS_PER_NODE=4 bash examples/run_experiment_single_node.sh \
  diffusion/qwen_image/qwen_image_flowgrpo_miles_aligned ++devices_per_node=4
# 2 v2 aligned (recipe as on this branch, 12 targets)
#   same as #1 with WANDB_RUN_NAME=qwen_image_flowgrpo_miles_aligned_v2_mlp_lora
# 3 control r1 (4 GPU, second ray cluster on GPUs 4-7, RAY_TMPDIR=/scratch/ray2, port 6390)
python -m unirl.train_diffusion --config-name=diffusion/qwen_image/qwen_image_dancegrpo \
  num_devices=4 ++devices_per_node=4 ++logging.report_to_wandb=true
# 4 control r2 (8 GPU)
python -m unirl.train_diffusion --config-name=diffusion/qwen_image/qwen_image_dancegrpo \
  num_devices=8 ++logging.report_to_wandb=true ++rollout.forward_batch_size=8 ++stack.micro_batch_size=4
# 5 cut +CFG4 short
#   = #4 + $P ${R}bisect_stock_plus_cfg4 ++sampling.guidance_scale=4.0
# 6 cut FlowSDE kernel
python -m unirl.train_diffusion --config-name=diffusion/qwen_image/qwen_image_trainside \
  num_devices=8 ++logging.report_to_wandb=true $P ${R}bisect_flowsde_kernel_eta07 \
  ++rollout.forward_batch_size=8 ++stack.micro_batch_size=4
# 7 cut eta 1.2         = #6 + ${R}bisect_flowsde_eta12 "++sampling.eta=1.2"
# 8 cut SDE triple      = #7 + ${R}bisect_sde_triple "++sampling.sde_indices=[3,4]"
# 9 cut triple+CFG4     = #8 + ${R}bisect_sde3_cfg4 ++sampling.guidance_scale=4.0
# 10 cut CFG4 long      = #4 + $P ${R}bisect_cfg4_long ++sampling.guidance_scale=4.0  (run to 90)
```

Failed pre-runs (documented for completeness, no wandb): sglang-engine aligned
recipe (died on the sglang×true-CFG geometry bug), trainside relaunch on
diffusers 0.39 (txt_seq_lens TypeError), kernel cut via `strategy._target_=`
override (Hydra "not in struct" — the trainside recipe keys it under
`pipeline.strategy`; use the `qwen_image_trainside` recipe instead).
