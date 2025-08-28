import torch
import math

from diffusers_helper.k_diffusion.uni_pc_fm import sample_unipc
from diffusers_helper.k_diffusion.wrapper import fm_wrapper
from diffusers_helper.utils import repeat_to_batch_size


# 定义一个函数 flux_time_shift，输入 t（时间步），mu 和 sigma 参数
def flux_time_shift(t, mu=1.15, sigma=1.0):
    # 返回一个基于指数函数和幂次计算的值，用来控制“时间偏移”的分布
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def calculate_flux_mu(context_length, x1=256, y1=0.5, x2=4096, y2=1.15, exp_max=7.0):
    # 根据上下文长度 context_length 计算 mu 值
    k = (y2 - y1) / (x2 - x1)  # 计算斜率 k（线性插值用）
    b = y1 - k * x1            # 计算截距 b
    mu = k * context_length + b  # 根据直线公式 y = kx + b 计算 mu
    mu = min(mu, math.log(exp_max))  # 限制 mu 的最大值，不能超过 log(exp_max)
    return mu  # 返回计算得到的 mu


  # 输入 n（步数）和 mu，计算 sigmas
def get_flux_sigmas_from_mu(n, mu):
    sigmas = torch.linspace(1, 0, steps=n + 1)  # 在 [1,0] 区间生成 n+1 个等间距点
    sigmas = flux_time_shift(sigmas, mu=mu)     # 对这些点应用 flux_time_shift 函数，得到平滑过渡的 sigmas
    return sigmas


@torch.inference_mode()
def sample_hunyuan(
        transformer,           # Transformer 模型
        sampler='unipc',       # 采样器类型（默认 unipc）
        initial_latent=None,   # 初始潜变量（可选）
        concat_latent=None,    # 拼接的潜变量（可选）
        strength=1.0,          # 控制初始潜变量影响程度
        width=512,
        height=512,
        frames=16,
        real_guidance_scale=1.0,     # 实际指导的权重
        distilled_guidance_scale=6.0,# 蒸馏指导的权重
        guidance_rescale=0.0,        # 指导重缩放参数
        shift=None,                  # 偏移量（可选）
        num_inference_steps=25,      # 推理步数
        batch_size=None,             # 批大小
        generator=None,              # 随机数生成器
        prompt_embeds=None,          # 文本 prompt 的嵌入
        prompt_embeds_mask=None,     # 文本 prompt 的掩码
        prompt_poolers=None,         # 文本池化表示
        negative_prompt_embeds=None, # 负面 prompt 的嵌入
        negative_prompt_embeds_mask=None, # 负面 prompt 的掩码
        negative_prompt_poolers=None,     # 负面 prompt 的池化表示
        dtype=torch.bfloat16,        # 数据类型
        device=None,                 # 设备（CPU/GPU）
        negative_kwargs=None,        # 负面额外参数
        callback=None,               # 每步回调函数
        **kwargs,                    # 其他额外参数
):
    device = device or transformer.device  # 如果没有指定 device，则用模型所在的设备

    if batch_size is None:  # 如果没指定批大小
        batch_size = int(prompt_embeds.shape[0])  # 用 prompt_embeds 的 batch 维度来确定

    # 生成随机潜变量 latents (B,C,T,H,W)
    latents = torch.randn((batch_size, 16, (frames + 3) // 4, height // 8, width // 8), generator=generator, device=generator.device).to(device=device, dtype=torch.float32)

    B, C, T, H, W = latents.shape  # 解包维度
    seq_length = T * H * W // 4    # 序列长度

    if shift is None:
        mu = calculate_flux_mu(seq_length, exp_max=7.0)  # 如果没有指定 shift，就自动计算 mu
    else:
        mu = math.log(shift)  # 如果指定了 shift，就取 log

    sigmas = get_flux_sigmas_from_mu(num_inference_steps, mu).to(device)  # 根据 mu 生成 sigmas

    k_model = fm_wrapper(transformer)  # 包装 transformer，得到用于采样的模型

    if initial_latent is not None:  # 如果有初始潜变量
        sigmas = sigmas * strength  # 调整 sigmas
        first_sigma = sigmas[0].to(device=device, dtype=torch.float32)  # 第一个 sigma
        initial_latent = initial_latent.to(device=device, dtype=torch.float32)  # 转到目标设备
        # 将 initial_latent 与随机 latents 混合
        latents = initial_latent.float() * (1.0 - first_sigma) + latents.float() * first_sigma

    if concat_latent is not None:  # 如果有拼接的潜变量
        concat_latent = concat_latent.to(latents)  # 转换到 latents 的 dtype 和 device

    # 蒸馏指导张量（扩大 1000 倍）
    distilled_guidance = torch.tensor([distilled_guidance_scale * 1000.0] * batch_size).to(device=device, dtype=dtype)

    # 批量对齐（把输入都扩展到 batch_size 大小）
    prompt_embeds = repeat_to_batch_size(prompt_embeds, batch_size)
    prompt_embeds_mask = repeat_to_batch_size(prompt_embeds_mask, batch_size)
    prompt_poolers = repeat_to_batch_size(prompt_poolers, batch_size)
    negative_prompt_embeds = repeat_to_batch_size(negative_prompt_embeds, batch_size)
    negative_prompt_embeds_mask = repeat_to_batch_size(negative_prompt_embeds_mask, batch_size)
    negative_prompt_poolers = repeat_to_batch_size(negative_prompt_poolers, batch_size)
    concat_latent = repeat_to_batch_size(concat_latent, batch_size)

    # 准备采样器参数
    sampler_kwargs = dict(
        dtype=dtype,
        cfg_scale=real_guidance_scale,
        cfg_rescale=guidance_rescale,
        concat_latent=concat_latent,
        positive=dict(  # 正面 prompt 的条件
            pooled_projections=prompt_poolers,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_embeds_mask,
            guidance=distilled_guidance,
            **kwargs,
        ),
        negative=dict(  # 负面 prompt 的条件
            pooled_projections=negative_prompt_poolers,
            encoder_hidden_states=negative_prompt_embeds,
            encoder_attention_mask=negative_prompt_embeds_mask,
            guidance=distilled_guidance,
            **(kwargs if negative_kwargs is None else {**kwargs, **negative_kwargs}),
        )
    )

    if sampler == 'unipc':  # 如果选择的是 unipc 采样器
        results = sample_unipc(k_model, latents, sigmas, extra_args=sampler_kwargs, disable=False, callback=callback)
    else:
        raise NotImplementedError(f'Sampler {sampler} is not supported.')

    return results
