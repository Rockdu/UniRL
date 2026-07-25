# UniRL 复现 miles-diffusion qwen-image FlowGRPO — 对齐记录

日期:2026-07-23 晚 · 机器:`kangrui-h200-qwen`(8×H200,租约至 07-23 21:55 UTC)
UniRL commit:`393ff13`(main,官方 uv venv sglang 环境,torch 2.11.0+cu130)
对齐基准:miles-diffusion `origin/main` `scripts/run-diffusion-grpo-pickscore-5gpu-flowgrpo-aligned.sh`(flow_grpo `pickscore_qwenimage` lineage)

## 运行位置

- 代码:`/scratch/unirl`(venv `.venv-sglang`)
- recipe:`examples/diffusion/qwen_image/qwen_image_flowgrpo_miles_aligned.yaml`
- 日志:`/scratch/train_qwen_miles_aligned.log`
- 启动:`CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS_PER_NODE=4 bash examples/run_experiment_single_node.sh diffusion/qwen_image/qwen_image_flowgrpo_miles_aligned ++devices_per_node=4`
- wandb:project `unirl-qwen-image-miles-aligned`

## 逐 knob 对齐表

| knob | miles 5gpu | UniRL aligned recipe |
|---|---|---|
| 模型 | Qwen/Qwen-Image | 同(共享缓存) |
| prompts/rollout × samples | 32 × 16 = 512 | 同 |
| optimizer steps/rollout | 2(不相交半批) | num_updates_per_batch: 2(同语义,CountPlanner 切半) |
| 分辨率 / 采样步数 | 512² / 10 | 同 |
| SDE 步 | window [3,5) → [3,4] | sde_indices: [3, 4](静态) |
| noise_level (eta) | 1.2 | 同 |
| CFG | true_cfg 4.0 | guidance_scale 4.0 + negative_prompt " " |
| SDE kernel | flow_grpo | FlowSDEStrategy |
| LoRA | r64 α128 gaussian | 同(gaussian 需 patch,见偏差①) |
| lr / β2 / wd / clip / gnorm | 3e-4 / 0.999 / 1e-4 / 1e-4 / 1.0 | 同 |
| 精度 | master fp32 + forward bf16 + AC | FSDP mixed_precision bf16 + AC |
| advantage | GRPO,global std,per-prompt mean | adv_use_global_std: true |
| KL | beta 0 | FlowGRPO 默认 beta 0 |
| π_old 锚 | rollout(引擎 logp) | old_logp_source: rollout |
| reward | PickScore_v1 + CLIP-H processor,logit/26,bs 8 | 逐字符同公式同 ckpt |
| 数据 | flowgrpo_pickscore train/test.jsonl | **diff 验证过与 UniRL 自带 pickscore txt 完全一致**(25432/2048) |
| seed | 42 | 42 |
| 训练卡数 | 4(+1 reward 独立卡) | 4(reward colocate,用户确认无所谓) |
| num_rollout / eval | 400 / interval 30 | 同 |
| weight sync | LoRA 每 rollout | weight_sync_interval: 1 |

## 已知偏差(不可 knob 映射)

1. **LoRA gaussian init**:UniRL 硬编码 PEFT 默认 init;在 `/scratch/unirl/unirl/train/lora.py` 两处打了 `init_lora_weights="gaussian"` patch(69/98 行,带 `# miles/flow_grpo parity` 注释)。
2. **eval 采样步数**:miles eval 用 50 步,UniRL evaluate() 复用训练的 10 步。不影响训练轨迹。
3. **seed 派生机制**:miles 算术区间(rollout_seed + group_index×N + k)vs UniRL 哈希配方(blake2b)。设置对齐但 RNG 实现不同,bit 级轨迹必然不同——这是框架本质差异,复现目标是曲线量级/趋势对齐。
4. **数据 shuffle 实现**:同一 prompt 集、同 seed 42,但两框架 shuffle 代码不同,prompt 消费顺序不同。
5. **micro batch 切法**:miles tile(sample=8, tstep=1)/rank;UniRL micro_batch_size=4(=4 sample×2 step,同 8 cells/forward)。纯梯度累积粒度,不影响 optimizer step 数学。
6. **rollout 引擎:UniRL 用 trainside,miles 用 sgl-d colocate**。原计划用 UniRL 的 sglang_diffusion 引擎(和 miles 同源),但实测发现 UniRL 的 sglang 路径**没有任何 true-CFG 先例**:带 negative 分支时 grouped request 几何崩(`initial_noise batch dim 8 does not match batch_size=1`,patch_latent_prep.py 的展开规则不覆盖 CFG 路径)——他们所有 qwen sglang recipe 都跑 guidance 1.0。trainside 是 UniRL 的 reference 引擎(parity oracle),QwenImagePipeline 有完整 true-CFG 实现(guidance>1 自动补 " " 负提示)。训练数学不变;trainside 免 weight sync(in-process,权重新鲜度与 miles 每 rollout 同步等价)。**这个 sglang CFG bug 值得给 UniRL 报 issue。**

## v1 失败复盘 → v2(2026-07-23 晚)

**v1(run `1jjy6w3z`,126 rollouts)不涨**:train reward 0.842→0.853→回落 0.845 平台,eval 微降;miles 同期已显著上涨。诊断:组内无塌缩(zero_std_group=0,advantage_std~0.28)、探索正常(reward_std 0.065)、梯度存在(gn~2e-4)但 126 步间 ratio drift/gn/clip **全程无变化**——传动链疑点。

**根因:LoRA 注入范围没对齐**。miles 的 `configs/qwen_image.py`(代码默认值,不在启动脚本里!)注入 12 类模块 = 8 attention 投影 + **4 个 MLP**(`img_mlp.net.0.proj/net.2`, `txt_mlp.net.0.proj/net.2`);我从 UniRL 自家 recipe 抄的 target_modules 只有 8 个 attention。DiT 容量大头在 MLP,attention-only r64 拉不动。UniRL 自己有同类先例(PR #183:WAN LoRA 只 wrap 90/240 → curve went flat)。

**v2**:target_modules 补齐 12 类(两处:lora_cfg + model_config),run name `qwen_image_flowgrpo_miles_aligned_v2_mlp_lora`,日志 `/scratch/train_qwen_v2.log`。v1 保留为 "attention-only LoRA" 消融数据点。

miles 该配置文件里另一个值得知道的默认值:`_rebuild_pos_embed_freqs_on_cuda`——diffusers 的 QwenEmbedRope 缓存在 CPU 上建,CPU/CUDA 的 `torch.pow` 差 fp32 ULP,导致 miles trainside↔sgl-d 冻结权重下 noise_pred mean|Δ|~2e-2,他们在 CUDA 上重建缓存修掉。我们 trainside 自洽(rollout=replay 同一份缓存),不受影响;但**对比两框架绝对数值时要记得这个 RoPE 差异源**。

## v2 中期观察 + 对照实验(2026-07-23 深夜)

**v2 前 33 步与 v1 逐窗口几乎逐位一致**(均值差 <1e-3,max 完全相同)——同 seed 确定性采样下说明两个 run 的策略在函数空间都基本没动,LoRA 范围不是唯一根因。CFG blend 数学已排除(miles `cfg_combine` 与 UniRL 同款 norm-preserving,mirrors sglang-d)。

关键观察:UniRL 已验证的爬坡曲线全是 no-CFG 低起点(qwen-edit-plus 0.78→0.88、SD3 0.74→0.87);我们 CFG-4.0 起点 0.845,直接落在他们曲线的终点区,PickScore headroom 被压扁(但 miles 同 CFG 能涨,headroom 非完整解释)。

**对照实验**(GPU 4-7,第二 Ray 集群,`/scratch/train_bisect.log`,wandb `bisect_stock_dancegrpo_noCFG`):跑 UniRL 自家 `qwen_image_dancegrpo` 原版(no-CFG/384²/12步/eta0.7/Dance/replay 锚)。首 rollout reward 0.8081(≈他们的 no-CFG 起点 ✓)。判定逻辑:它涨 → UniRL qwen 机器能学,抑制因子在我们 delta{CFG4.0, eta1.2, sde[3,4], 512², rollout 锚}里,下一步逐项二分;它平 → **UniRL 的 qwen-image t2i 训练本身无爬坡先例可依,报告结论升级**。

## v2 正式判定(2026-07-23 23:35)

v2 到 60 步:窗口均值 1-20/21-40/41-60 = 0.8422/0.8430/0.8429,vs v1 同窗 0.8421/0.8449/0.8501——**v2 与 v1 无统计差异,MLP LoRA 修复未解锁学习**。两个 run 的策略在 60-126 步内都未产生可观测的函数空间移动。抑制因子在 LoRA 范围之外。

事故记录:清理对照 run 时 `pkill -f "train_diffusio[n]"` 交叉匹配误杀了 v2 driver(v2 判定数据已完整,损失有限)。教训:同模块名多 run 并存时,pkill 必须带 config 名或 PID 精确匹配。

## 对照 run 升级为 8 卡原生几何(2026-07-23 23:40)

4 卡版对照 17min/步太慢,且 v2 已判定 → 8 卡全部押对照:stock `qwen_image_dancegrpo` 原生几何(num_devices=8, batch 48),仅覆盖两个数学不变的性能参数(`forward_batch_size 1→8`,`micro_batch_size→4`)。wandb: kangrdu/unirl 项目 `bisect_stock_dancegrpo_noCFG_8gpu`。4 卡版 7 个 rollout 的读数:0.80-0.82 噪声带,过短无结论。

## 对照实验判定(2026-07-24 02:50)——UniRL 链路能学,抑制因子在 delta 里

stock `qwen_image_dancegrpo`(8×H200,r2,wandb kangrdu/unirl/iif1at4s)80 rollouts:0.812→0.854(窗口均值),max 0.877,**教科书爬坡 +0.042**。同引擎(trainside)、同锚(rollout)、同 LoRA 代码(attn-8 就够学!)、同 reward——**LoRA 范围理论正式出局,锚/引擎/配对全部洗清**。

活动嫌疑收缩至 miles-aligned delta:{true-CFG 4.0, eta 1.2, 静态 sde[3,4], 512², 10步, FlowSDE kernel, batch 32, wd 1e-4}。配合 eval 单调下行 +单边 lt clip 的"缓慢学反"指纹,首嫌 = true-CFG(UniRL 无先例路径)。

**二分刀 1 — CFG(已判,无罪)**:stock + 仅 guidance 4.0(`bisect_stock_plus_cfg4`),34 步窗口 0.8413→0.8369→0.8435→0.8507→~0.856,**照常爬坡**。CFG 出局——顺带首次验证了 UniRL true-CFG 训练路径可用(此前零先例)。注意它的 clip≈0.12-0.15(与不学习的对齐 run 同款,而 no-CFG stock 是 0.00-0.01)——说明高 clip 分数是 CFG 的伴生现象,不是病因指纹,之前把它当"学反"证据的一环需要修正。

**二分刀 2 — FlowSDE kernel(已判,无罪)**:`qwen_image_trainside`(=stock 唯一差 kernel,recipe 头注释背书),35 步 0.81→0.853,与 stock 同速。(首次尝试用 `strategy._target_=` 覆盖失败——trainside recipe 的 kernel 在 `pipeline.strategy` 下,顶层无 strategy 键,Hydra "not in struct" 静默死产 3.5h。)

**二分刀 3 — eta 1.2(已判,无罪)**:trainside + `++sampling.eta=1.2`,45 步 0.778→0.832(+0.054)。起点掉 3 分揭示重要机制:UniRL stock 的 AllSDE 窗口在前半段(σ→1 区),`std∝√(σ/(1-σ))` 发散,eta1.2 在该窗口噪声巨大但仍能学;flow_grpo/miles 的 [3,4] 窗口恰好避开爆炸区("bug"因祸得福)。

**二分刀 4 — SDE 三件套(进行中)**:trainside + eta1.2 + 静态 `sde_indices=[3,4]`(`bisect_sde_triple`,/scratch/train_triple.log)= 对齐配置的完整 SDE 组合@stock几何。若爬 → SDE 全出清,剩 {512²/10步, batch32, wd, CFG×SDE 交互},转反向二分(从死配置逐项还原);若死 → 凶手锁定 [3,4]×eta1.2 交互,再与 miles 的等效工作点(10步/512²)对钉。

代码审计侧的出清记录:kernel 公式与 miles 逐字符同源 flow_grpo;σ schedule 的 static-shift 假警报(build_schedule_policy 有 require_dynamic 防御);miles 配置里的 RoPE CPU/CUDA ULP 修复对 trainside 自洽无影响。

**二分刀 5 — SDE三件套+CFG4(已判,死;2026-07-24 16:35)**:89 步曲线**先升后衰**(峰 25-36 窗 0.8467 → 73-84 窗 0.8358 跌破起点)——与对齐 run 同款死相,且在 stock 几何(384²/12步/batch48/wd0)复现。**几何出局;CFG4 参与的组合是凶手**。同时 cfg4 单刀的旧无罪判定作废(34 步收刀恰在"隆起"峰上,系统性误判模式:CFG 档短窗似爬、长窗衰减;no-CFG 四刀长窗全部保持涨幅)。

**二分刀 6 — cfg4 单刀长跑(进行中)**:stock(Dance/0.7/早窗)+ guidance 4.0,目标 90 步(`bisect_cfg4_long`)。死 → CFG4 本身定罪(公式与 miles 同,机制去查负分支 embedding / CFG 梯度路径 / logp-blend 一致性);活 → CFG×FlowSDE 三件套交互定罪。

运维教训又 +1:pkill 的 bracket 技巧会被同一命令行里的启动明文破掉(exit 143 自杀一次),杀/启必须分两次 rx 调用。

- UniRL 带 CFG 训过的只有两例:`wan21_t2v_videoalign_dancegrpo`(trainside,5.0)和 `pe_sglang_full_wise`(sglang_diffusion+SD3 经典 CFG,4.5)。
- **Qwen-Image 全部 recipe guidance 1.0,官方从未带 true-CFG 训过**——我们这个 run 是 UniRL 生态首个 qwen-image true-CFG 训练。sglang 的 qwen true-CFG 路径无先例且实测碎(经典 CFG 的 SD3 路径是通的,勿混淆);vllm-omni 的 qwen CFG 代码完整但无人验证过。

## 启动踩坑记录

- launcher 用 `nvidia-smi -L` 数卡(无视 CUDA_VISIBLE_DEVICES)→ 需 `GPUS_PER_NODE=4`
- DevicePool 默认 devices_per_node=8 → 需 `++devices_per_node=4`
- **diffusers 版本漂移**:uv override `diffusers>=0.38.0` 解析到 0.39.0,但 0.39 删了 `QwenImageTransformer2DModel.forward(txt_seq_lens=...)` → UniRL main 在 0.39 下 Qwen-Image 直接 TypeError。修法:pyproject override 改 `>=0.38.0,<0.39` + 重装 0.38.0。**也值得给 UniRL 报 issue。**
- eval 偏差补充:UniRL 在 step 0 也跑了 eval(miles 是 --skip-eval-before-train),无害。

## 运行状态(2026-07-23 08:51 UTC)

- wandb: https://wandb.ai/kangrdu/unirl-qwen-image-miles-aligned/runs/1jjy6w3z
- EVAL step 0: reward=0.8589(cfg=4.0, eta=0, 4 samples/prompt)
- rollout 1/400: reward=0.8455, ratio=1.0000±0.0000(update-0 无 gap ✓), clip=0.04, lr=3e-4
- 步长 ≈ 280s/rollout(512 图生成 + PickScore + 2 optimizer steps,4×H200 满载)
- 租约到 07-23 21:55 UTC ≈ 还能跑 ~165 rollouts;**要跑满 400 需明早 `rx devbox extend kangrui-h200-qwen`**
- 未配 checkpoint 保存(miles save-interval 10;曲线对比不需要,如需 ckpt 再加)
