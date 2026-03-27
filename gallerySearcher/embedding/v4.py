import json
import sys
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data(file_path):
    """
    加载并清洗 metadata.json 数据
    """
    print(f"正在加载数据: {file_path}", flush=True)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    print(f"原始数据样本数: {len(df)}", flush=True)
    
    # --- 新增: 提取 Checkpoint(Model) 和 LoRA 信息 ---
    def extract_model(meta_str):
        if pd.isna(meta_str): return 'UNKNOWN'
        match = re.search(r'Model:\s*([^,]+)', str(meta_str))
        return match.group(1).strip() if match else 'UNKNOWN'
        
    def extract_loras(meta_str):
        if pd.isna(meta_str): return ''
        loras = re.findall(r'<lora:([^:]+):', str(meta_str))
        return ' '.join(loras)

    if 'full_metadata_string' in df.columns:
        df['model'] = df['full_metadata_string'].apply(extract_model)
        df['loras'] = df['full_metadata_string'].apply(extract_loras)
    else:
        df['model'] = 'UNKNOWN'
        df['loras'] = ''
        
    # 强化 Prompt: 将 LoRA 的名称融合进 prompt，引导 CLIP 捕获特定的概念语义
    df['prompt'] = df['prompt'].fillna('')
    df['enhanced_prompt'] = df['prompt'] + " " + df['loras']
    
    # ... (rest of the cleaning) ...
    df['negative_prompt'] = df['negative_prompt'].fillna('')
    df['clipskip'] = df['clipskip'].fillna(2).astype(float)
    df['cfgscale'] = pd.to_numeric(df['cfgscale'], errors='coerce')
    df['cfgscale'] = df['cfgscale'].fillna(df['cfgscale'].median())
    df['steps'] = pd.to_numeric(df['steps'], errors='coerce')
    df['steps'] = df['steps'].fillna(df['steps'].median())
    df['sampler'] = df['sampler'].fillna('UNKNOWN').str.upper()
    
    print("数据清洗完成。", flush=True)
    return df

def build_features(df):
    """
    构建文本、数值和类别特征
    """
    print("\n[阶段一] 正在使用 CLIP 模型提取文本特征 (这可能需要几秒钟)...", flush=True)
    model = SentenceTransformer('clip-ViT-B-32') 
    
    # 【核心改动】：完全抛弃可能带来污染的 negative_prompt，仅编码正面意图与 LoRA
    text_features = model.encode(df['enhanced_prompt'].tolist(), show_progress_bar=True)
    print(f"原始文本特征维度 (仅正向+LoRA): {text_features.shape}", flush=True)

    print("\n[阶段二] 正在处理生成参数 (数值标准化 & 类别独热编码)...", flush=True)
    scaler = StandardScaler()
    num_features = scaler.fit_transform(df[['cfgscale', 'steps', 'clipskip']])
    
    encoder_sampler = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    sampler_features = encoder_sampler.fit_transform(df[['sampler']])
    
    # 【核心改动】：将 Checkpoint (Model) 作为独立的特征进行独热编码
    encoder_model = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    model_features = encoder_model.fit_transform(df[['model']])
    
    print(f"数值特征维度: {num_features.shape}", flush=True)
    print(f"类别特征 (Sampler) 维度: {sampler_features.shape}", flush=True)
    print(f"类别特征 (Model) 维度: {model_features.shape}", flush=True)
    
    return text_features, num_features, sampler_features, model_features, scaler, encoder_sampler

def perform_two_stage_pca(text_features, num_features, sampler_features, model_features, final_dim=8):
    """
    执行两阶段 PCA 降维，并加入特征权重平衡
    """
    print(f"\n[阶段三] 执行两阶段 PCA 降维 (目标维度: {final_dim})...", flush=True)
    n_text_components = min(20, text_features.shape[0], text_features.shape[1])
    pca_text = PCA(n_components=n_text_components)
    text_reduced = pca_text.fit_transform(text_features)
    print(f"第一阶段：文本特征被压缩至 {n_text_components} 维", flush=True)
    
    # 【核心修复】：模态权重平衡！
    # 1. 消除文本特征在第一阶段降维后的数值衰减，拉回到同一起跑线
    scaler_text = StandardScaler()
    text_reduced_scaled = scaler_text.fit_transform(text_reduced)
    
    # 2. 赋予不同模态显式的业务权重 (文本语义决定画面内容，必须占主导)
    TEXT_WEIGHT = 4.0     # 强化语义特征 (包含 Prompt 和 LoRA，占主导)
    MODEL_WEIGHT = 3.0    # 强化 Checkpoint 特征 (极大地影响画风，如二次元 vs 写实)
    PARAM_WEIGHT = 0.5    # 弱化数值参数 (CFG/Steps)
    SAMPLER_WEIGHT = 0.5  # 弱化采样器特征
    
    combined_features = np.hstack([
        text_reduced_scaled * TEXT_WEIGHT, 
        model_features * MODEL_WEIGHT,
        num_features * PARAM_WEIGHT, 
        sampler_features * SAMPLER_WEIGHT
    ])
    
    n_final_components = min(final_dim, combined_features.shape[0], combined_features.shape[1])
    pca_final = PCA(n_components=n_final_components)
    pbo_space = pca_final.fit_transform(combined_features)
    explained_variance = sum(pca_final.explained_variance_ratio_) * 100
    print(f"第二阶段：最终构建的 PBO 搜索空间形状: {pbo_space.shape}", flush=True)
    print(f"最终的 {n_final_components} 维保留了原始融合空间 {explained_variance:.2f}% 的方差信息。", flush=True)
    return pbo_space, pca_text, pca_final

def simulate_pbo_retrieval(pbo_space, df, target_index=0, top_k=5):
    """
    模拟 PBO 找到最优点后，如何使用 KNN 检索
    """
    print("\n[阶段四] 模拟 PBO 预测后的 KNN 检索...", flush=True)
    predicted_optimal_point = pbo_space[target_index].reshape(1, -1) 
    
    # 【核心改动】：改用余弦相似度 (Cosine) 计算语义距离
    knn = NearestNeighbors(n_neighbors=top_k, metric='cosine')
    knn.fit(pbo_space)
    distances, indices = knn.kneighbors(predicted_optimal_point)
    print(f"\n==== 为 LLM Agent 准备的 Top {top_k} 参考 Metadata ====\n", flush=True)
    top_metadata_for_llm = []
    for i, idx in enumerate(indices[0]):
        row = df.iloc[idx]
        metadata_dict = {
            "distance": round(distances[0][i], 4),
            "id": row.get('id', 'N/A'),
            "prompt": row['prompt'][:100] + "..." if len(row['prompt']) > 100 else row['prompt'],
            "model": row.get('model', 'UNKNOWN'),
            "loras": row.get('loras', ''),
            "cfgscale": row['cfgscale'],
            "steps": row['steps'],
            "sampler": row['sampler']
        }
        top_metadata_for_llm.append(metadata_dict)
        print(f"Rank {i+1} (余弦距离: {metadata_dict['distance']}): ID {metadata_dict['id']}", flush=True)
        print(f"  - Model: {metadata_dict['model']} | LoRAs: {metadata_dict['loras']}", flush=True)
        print(f"  - Prompt: {metadata_dict['prompt']}", flush=True)
        print(f"  - CFG: {metadata_dict['cfgscale']} | Steps: {metadata_dict['steps']} | Sampler: {metadata_dict['sampler']}", flush=True)
        print("-" * 40, flush=True)
        
    print(f"\n[最终展示] 正在根据优化结果显示 Top {top_k} 匹配图片...", flush=True)
    display_images(df, indices[0].tolist(), title=f"Top {top_k} Recommended Images", filename="pbo_results_top5.png")
    return top_metadata_for_llm

def display_images(df, indices, title="Images", filename="pbo_display.png"):
    """
    使用 Matplotlib 显示多张图片，并保存到本地
    """
    print(f"正在准备显示图片: {title} (索引: {indices})", flush=True)
    n = len(indices)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 5))
    if n == 1: axes = [axes]
    fig.suptitle(title, fontsize=16)
    gallery_dir = '/Users/mgccvmacair/Myproject/Academic/Genflow/spider/civitai_gallery'
    for i, idx in enumerate(indices):
        row = df.iloc[idx]
        local_filename = row.get('local_path')
        img_url = row.get('image_url')
        try:
            img = None
            if local_filename:
                local_path = os.path.join(gallery_dir, local_filename)
                if os.path.exists(local_path): img = Image.open(local_path)
            if img is None and img_url:
                try:
                    response = requests.get(img_url, timeout=3)
                    img = Image.open(BytesIO(response.content))
                except: pass
            if img is None: raise FileNotFoundError(f"Cannot find image for ID {row.get('id')}")
            axes[i].imshow(img)
            axes[i].set_title(f"Idx: {idx}\nID: {row.get('id', 'N/A')}")
            axes[i].axis('off')
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Image Load Error\n{e}", ha='center', va='center', fontsize=8)
            axes[i].set_title(f"Idx: {idx} (Error)")
            axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    print(f"图片已保存至: {os.path.abspath(filename)}", flush=True)
    try:
        plt.show(block=False)
        plt.pause(0.5) 
        plt.close()
    except: pass

def run_pbo_loop(pbo_space, df, iterations=10, batch_size=4):
    """
    运行真实的 PBO 优化过程 (包含多样性采样与动态探索机制)
    """
    print(f"\n[阶段四] 开始 PBO 优化过程 (共 {iterations} 轮, 每轮显示 {batch_size} 张图片)...", flush=True)
    kernel = 1.0 * Matern(length_scale=1.0, nu=1.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1, n_restarts_optimizer=5)
    X_train = []
    y_train = []
    
    kappa_base = 1.5 # 基础探索系数
    consecutive_skips = 0 # 连续跳过次数
    
    current_iter = 0
    while current_iter < iterations:
        print(f"\n--- 第 {current_iter+1} / {iterations} 轮 ---", flush=True)
        
        # 动态调大探索系数 kappa
        cur_kappa = kappa_base + (2.5 * consecutive_skips)
            
        if len(X_train) < 2:
            candidate_indices = np.random.choice(len(pbo_space), batch_size, replace=False).tolist()
        else:
            gp.fit(np.array(X_train), np.array(y_train))
            mu, sigma = gp.predict(pbo_space, return_std=True)
            ucb = mu + cur_kappa * sigma
            
            # 【核心修复】：解决连续按 0 卡死的问题 (防止候选池坍缩)
            
            if consecutive_skips >= 2:
                # 方案 B：纯探索逃脱机制 (Pure Exploration Trigger)
                # 当连续跳过大于等于 2 次，说明预测得分已失效。不再看 UCB，
                # 而是直接寻找系统“最不确定 (Sigma 最大)”的地方进行随机抽样！
                print(f">>> 严重卡点检测：激活全局强探索 (跳出局部最优)...", flush=True)
                pool_size = min(len(pbo_space), 100)
                # 提取方差(不确定性)最大的那批图片
                sigma_pool_indices = np.argsort(sigma)[-pool_size:].tolist()
                np.random.shuffle(sigma_pool_indices)
                candidate_indices = sigma_pool_indices[:batch_size]
                
            else:
                # 方案 A：动态多样性采样 (Dynamic Diversity Sampling)
                # 跳过次数越多，我们强行把候选池挖得越深，强制容纳更多风格
                # 基础候选池大小为全库的 1/3，每次跳过额外扩大 50
                base_pool_size = max(50, len(pbo_space) // 3)
                top_n = min(len(pbo_space), base_pool_size + 50 * consecutive_skips)
                top_k_indices = np.argsort(ucb)[-top_n:].tolist()
                
                # 选出池子里绝对最高分的 1 张
                candidate_indices = [top_k_indices.pop(-1)]
                
                # 再挑出 3 张：保证它们既是高分，又与已挑出的图长得尽量不一样 (距离最大化)
                while len(candidate_indices) < batch_size and top_k_indices:
                    max_min_dist = -1
                    best_idx_in_pool = -1
                    for pool_idx in top_k_indices:
                        dist_to_selected = min([np.linalg.norm(pbo_space[pool_idx] - pbo_space[s]) for s in candidate_indices])
                        if dist_to_selected > max_min_dist:
                            max_min_dist = dist_to_selected
                            best_idx_in_pool = pool_idx
                    
                    candidate_indices.append(best_idx_in_pool)
                    top_k_indices.remove(best_idx_in_pool)
            
        display_images(df, candidate_indices, title=f"Round {current_iter+1} Candidates (Interactive)", filename=f"pbo_rounds.png")
        print(f"请对以下 {batch_size} 张图片提供偏好反馈 (查看弹出窗口):", flush=True)
        for idx, c_idx in enumerate(candidate_indices):
            row = df.iloc[c_idx]
            print(f"  [{idx+1}] ID: {row.get('id', 'N/A')} | Prompt: {row['prompt'][:80]}...", flush=True)
            
        # 3. 获取反馈
        while True:
            try:
                line = input(f"请输入【最喜欢】和【最不喜欢】的序号 (例如: 1 4), 或输入 0 跳过本轮: ")
                line = line.strip()
                if line == '0':
                    # 用户跳过本轮 (不占用优化轮次)
                    consecutive_skips += 1
                    for c_idx in candidate_indices:
                        X_train.append(pbo_space[c_idx])
                        y_train.append(0.0) # 全局惩罚：这些全都不行
                    print(f"已跳过本轮。累计连续跳过 {consecutive_skips} 次，已记录惩罚点 (本轮不计入总轮次)。", flush=True)
                    gp.fit(np.array(X_train), np.array(y_train)) # 强制更新模型
                    break 
                
                parts = line.split()
                if len(parts) == 2:
                    best_idx = int(parts[0]) - 1
                    worst_idx = int(parts[1]) - 1
                    if 0 <= best_idx < batch_size and 0 <= worst_idx < batch_size:
                        # 正常反馈
                        consecutive_skips = 0 # 反馈成功，清零跳过计数
                        for idx, c_idx in enumerate(candidate_indices):
                            X_train.append(pbo_space[c_idx])
                            if idx == best_idx: y_train.append(1.0)
                            elif idx == worst_idx: y_train.append(0.0)
                            else: y_train.append(0.5)
                        current_iter += 1 # 只有正常反馈才进入下一轮
                        break
                print(f"请输入两个 1 到 {batch_size} 之间的数字，或输入 0 跳过。", flush=True)
            except ValueError:
                print("请输入有效的数字或 0。", flush=True)
        
        print(f"反馈已记录。当前已收集 {len(X_train)} 个数据点。", flush=True)
    print("\n[优化结束] 正在计算最终推荐结果...", flush=True)
    gp.fit(np.array(X_train), np.array(y_train))
    final_scores = gp.predict(pbo_space)
    best_idx = np.argmax(final_scores)
    print(f"\n==== PBO 预测的【最佳生成参数组合】参考 ====", flush=True)
    row = df.iloc[best_idx]
    print(f"ID: {row.get('id', 'N/A')}", flush=True)
    print(f"Model: {row.get('model', 'UNKNOWN')} | LoRAs: {row.get('loras', '')}", flush=True)
    print(f"Prompt: {row['prompt']}", flush=True)
    print(f"CFG: {row['cfgscale']} | Steps: {row['steps']} | Sampler: {row['sampler']}", flush=True)
    print("============================================\n", flush=True)
    return best_idx

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = '/Users/mgccvmacair/Myproject/Academic/Genflow/spider/civitai_gallery/metadata.json' 
    
    try:
        df = load_and_clean_data(file_path)
        text_features, num_features, sampler_features, model_features, scaler, encoder = build_features(df)
        pbo_space, pca_text, pca_final = perform_two_stage_pca(text_features, num_features, sampler_features, model_features, final_dim=8)
        
        # 步骤 4: 随机选择一个 Target 目标图片作为参考
        target_index = np.random.randint(0, len(df))
        print(f"\n[开始前] 随机选择了索引为 {target_index} 的图片作为本次优化的参考目标...", flush=True)
        display_images(df, [target_index], title="Target Image (Goal Reference)", filename="pbo_target.png")
        input("查看完目标图片后，按回车开始 PBO 循环...")
        
        best_pbo_index = run_pbo_loop(pbo_space, df, iterations=10, batch_size=4)
        
        print("\n[最终结果] 显示 PBO 找到的最佳图片...", flush=True)
        display_images(df, [best_pbo_index], title="Final Optimized Result", filename="pbo_final.png")
        
        simulate_pbo_retrieval(pbo_space, df, target_index=best_pbo_index, top_k=5)
        
        print("\n🎉 PBO 流程演示完成！", flush=True)
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{file_path}'。", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"发生错误: {e}", flush=True)