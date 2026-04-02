# ComfyUI Resources

This inventory is aligned with the current gallery metadata distribution
(`spider/civitai_gallery/metadata.json`) so agent retrieval and expansion
decisions stay close to what actually exists in the wall.

## Checkpoints

- `SDXL`
  - High-frequency base family in gallery; safe default for broad compatibility.
- `epicrealismXL_vxiiAbea2t`
  - Frequent realistic branch for polished portraits and cinematic scenes.
- `SDXLFaetastic_v24`
  - Common stylized/fantasy branch in the gallery.
- `eventHorizonXL_v60`
  - Frequent sci-fi/cyber branch.
- `epicrealismXL_pureFix`
  - Realism-heavy branch with cleaner skin/lighting behavior.
- `Pony`
  - Core anime/stylized branch in gallery data.
- `cyberrealisticXL_V10DMD2`
  - Cyberpunk and high-detail character-friendly branch.
- `cyberrealisticXL_v90`
  - Alternate cyberrealistic branch found in gallery.
- `zavychromaxl_v100`
  - Frequent stylized-realism checkpoint.
- `Illustrious`
  - Illustrative branch present in gallery.
- `waiIllustriousSDXL_v160`
  - WAI/Illustrious line appearing multiple times.
- `ebara_pony_2.1`
  - Pony family variant in gallery set.
- `albedobaseXL_v21`
  - Additional frequent SDXL branch.
- `Grilled_Lamprey_SDXL_v12o7n9745`
  - Distinct SDXL variant present in gallery.
- `Juggernaut_XL_-_Ragnarok_by_RunDiffusion`
  - Keep for high-quality cinematic realism when requested.

## LoRAs

- `Kodak Portra 400 analog film stock style v2`
- `angelic golden armor`
- `ArtDeco01_CE_SDXL_64x32x300x2bOT`
- `dmd2_sdxl_4step_lora`
- `70sSci-FiMJ7SDXL`
- `VintageDrawing01-00_CE_SDXL_128OT`
- `RetroPop01_CE_SDXL`
- `50sPanavisionMovieSDXL`
- `sdxl_lightning_8step_lora`
- `skin tone style v4`
- `1990's PC_style_SDXL_v1`
- `perfection style`
- `30TechnicolorMovieMJ7SDXL`
- `psyglowxl`
- `VintageTravelPoster01a_CE_SDXL_64x32x120x2bOT`
- `add-detail-xl`
- `Minimalist_vector_art`
- `Art_Deco_Interior_2_(Buildings)_(SDXL)_(AD)`
- `detailed_notrigger`
- `Apparatus - Traction City Walkers Vehicles_epoch_11`
- `SDXLHighDetail_v5`
- `RMSDXL_Creative`
- `Pony_DetailV2.0`
- `MeMaXL_V3`
- `great_lighting`
- `zy_Detailed_Backgrounds_v1`

## Samplers

- `EULER A`
- `DPM++ 2M KARRAS`
- `DPM++ 2M`
- `DPM++ 3M SDE`
- `LCM`
- `DPM++ 2M SDE`
- `DPM++ SDE KARRAS`
- `DPM++ 3M SDE EXPONENTIAL`
- `DPM++ 2M SDE KARRAS`
- `EULER`
- `DDIM`

## VAE / Auxiliary

- `sdxl_vae`
- `sdxlVaeAnimeTest_beta120000`
- `4xUltrasharp_4xUltrasharpV10`

## Retrieval Hints

- If user intent is analog film / photo realism, prioritize `epicrealismXL_vxiiAbea2t`, `epicrealismXL_pureFix`, or `SDXL` with `Kodak Portra 400 analog film stock style v2`.
- If user intent is cyberpunk sci-fi poster, prioritize `eventHorizonXL_v60` or `cyberrealisticXL_V10DMD2` with `70sSci-FiMJ7SDXL`, `30TechnicolorMovieMJ7SDXL`, or `psyglowxl`.
- If user intent is art-deco, retro poster, or graphic layout, prioritize `ArtDeco01_CE_SDXL_64x32x300x2bOT`, `VintageTravelPoster01a_CE_SDXL_64x32x120x2bOT`, and `Minimalist_vector_art`.
- If user intent is anime/illustration, prioritize `Pony`, `ebara_pony_2.1`, `waiIllustriousSDXL_v160`, and `Pony_DetailV2.0`.
- For high detail character close-up, prefer `DPM++ 2M KARRAS` or `DPM++ 3M SDE` and combine with `add-detail-xl`, `SDXLHighDetail_v5`, or `great_lighting`.
