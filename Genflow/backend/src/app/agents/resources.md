# ComfyUI Resources

This file is a lightweight mock inventory for the creative intake agent.
It is intentionally human-readable so it can be replaced by a real RAG source
later without changing the orchestration code.

## Checkpoints

- `Juggernaut_XL_-_Ragnarok_by_RunDiffusion`
  - Best for cinematic realism, fantasy realism, painterly realism, and general-purpose image generation.
- `CyberRealistic_CyberIllustrious_-_v7-0`
  - Best for polished portraits with strong detail and controlled lighting.
- `Kody`
  - Best for stylized architecture, art deco, and clean graphic composition.
- `incursiosMemeDiffusion_v27PDXL`
  - Best for energetic illustration and meme-adjacent stylized work.
- `Pony`
  - Best for anime-flavored, high-contrast, expressive character work.
- `NoobAI`
  - Best for furry, kemono, and stylized creature concepts.
- `lilithsDesire_v10`
  - Best for soft illustration, slice-of-life, and balanced composition.

## LoRAs

- `add-detail-xl`
- `zy_Detailed_Backgrounds_v1`
- `perfection style`
- `Rawfully_Stylish_v0.2`
- `great_lighting`
- `SDXLHighDetail_v5`
- `RMSDXL_Creative`
- `Pony_DetailV2.0`
- `MeMaXL_V3`
- `Star_ButterflyILL`

## Samplers

- `DPM++ 2M Karras`
- `DPM++ 3M SDE`
- `DPM++ 3M SDE Exponential`
- `Euler a`

## VAE / Auxiliary

- `sdxl_vae`
- `sdxlVaeAnimeTest_beta120000`
- `4xUltrasharp_4xUltrasharpV10`

## Retrieval Hints

- If the user asks for cinematic realism, prefer `Juggernaut_XL_-_Ragnarok_by_RunDiffusion`.
- If the user asks for anime or flat color, prefer `Pony` and keep composition explicit.
- If the user asks for architecture or graphic design, prefer `Kody` and cleaner samplers.
- If the user asks for furry/anthro subjects, prefer `NoobAI`.
- If the user asks for illustration or storybook work, prefer `lilithsDesire_v10` or `Pony`.

