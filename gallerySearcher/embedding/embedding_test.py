import json
import sys
import os
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
    
    # ... (rest of the cleaning) ...
    df['prompt'] = df['prompt'].fillna('')
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
    prompt_emb = model.encode(df['prompt'].tolist(), show_progress_bar=True)
    neg_prompt_emb = model.encode(df['negative_prompt'].tolist(), show_progress_bar=True)
    text_features = np.hstack([prompt_emb, neg_prompt_emb])
    print(f"原始文本特征维度 (正+负): {text_features.shape}", flush=True)

    print("\n[阶段二] 正在处理生成参数 (数值标准化 & 类别独热编码)...", flush=True)
    scaler = StandardScaler()
    num_features = scaler.fit_transform(df[['cfgscale', 'steps', 'clipskip']])
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    sampler_features = encoder.fit_transform(df[['sampler']])
    
    print(f"数值特征维度: {num_features.shape}", flush=True)
    print(f"类别特征 (Sampler) 维度: {sampler_features.shape}", flush=True)
    return text_features, num_features, sampler_features, scaler, encoder

def perform_two_stage_pca(text_features, num_features, sampler_features, final_dim=8):
    """
    执行两阶段 PCA 降维
    """
    print(f"\n[阶段三] 执行两阶段 PCA 降维 (目标维度: {final_dim})...", flush=True)
    n_text_components = min(20, text_features.shape[0], text_features.shape[1])
    pca_text = PCA(n_components=n_text_components)
    text_reduced = pca_text.fit_transform(text_features)
    print(f"第一阶段：文本特征被压缩至 {n_text_components} 维", flush=True)
    combined_features = np.hstack([text_reduced, num_features, sampler_features])
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
    knn = NearestNeighbors(n_neighbors=top_k, metric='euclidean')
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
            "cfgscale": row['cfgscale'],
            "steps": row['steps'],
            "sampler": row['sampler']
        }
        top_metadata_for_llm.append(metadata_dict)
        print(f"Rank {i+1} (距离: {metadata_dict['distance']}): ID {metadata_dict['id']}", flush=True)
        print(f"  - Prompt: {metadata_dict['prompt']}", flush=True)
        print(f"  - CFG: {metadata_dict['cfgscale']} | Steps: {metadata_dict['steps']} | Sampler: {metadata_dict['sampler']}", flush=True)
        print("-" * 40, flush=True)
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
    运行真实的 PBO 优化过程
    """
    print(f"\n[阶段四] 开始 PBO 优化过程 (共 {iterations} 轮, 每轮显示 {batch_size} 张图片)...", flush=True)
    kernel = 1.0 * Matern(length_scale=1.0, nu=1.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1, n_restarts_optimizer=5)
    X_train = []
    y_train = []
    for i in range(iterations):
        print(f"\n--- 第 {i+1} / {iterations} 轮 ---", flush=True)
        if len(X_train) < 2:
            candidate_indices = np.random.choice(len(pbo_space), batch_size, replace=False).tolist()
        else:
            gp.fit(np.array(X_train), np.array(y_train))
            mu, sigma = gp.predict(pbo_space, return_std=True)
            ucb = mu + 1.5 * sigma
            candidate_indices = np.argsort(ucb)[-batch_size:][::-1].tolist()
        display_images(df, candidate_indices, title=f"Round {i+1} Candidates (Interactive)", filename=f"pbo_round_{i+1}.png")
        print(f"请对以下 {batch_size} 张图片提供偏好反馈 (查看弹出窗口):", flush=True)
        for idx, c_idx in enumerate(candidate_indices):
            row = df.iloc[c_idx]
            print(f"  [{idx+1}] ID: {row.get('id', 'N/A')} | Prompt: {row['prompt'][:80]}...", flush=True)
        while True:
            try:
                line = input(f"请输入【最喜欢】和【最不喜欢】的序号 (例如: 1 4): ")
                parts = line.split()
                if len(parts) == 2:
                    best_idx = int(parts[0]) - 1
                    worst_idx = int(parts[1]) - 1
                    if 0 <= best_idx < batch_size and 0 <= worst_idx < batch_size:
                        break
                print(f"请输入两个 1 到 {batch_size} 之间的数字，用空格隔开。", flush=True)
            except ValueError:
                print("请输入有效的数字。", flush=True)
        for idx, c_idx in enumerate(candidate_indices):
            X_train.append(pbo_space[c_idx])
            if idx == best_idx: y_train.append(1.0)
            elif idx == worst_idx: y_train.append(0.0)
            else: y_train.append(0.5)
        print(f"反馈已记录。当前已收集 {len(X_train)} 个数据点。", flush=True)
    print("\n[优化结束] 正在计算最终推荐结果...", flush=True)
    gp.fit(np.array(X_train), np.array(y_train))
    final_scores = gp.predict(pbo_space)
    best_idx = np.argmax(final_scores)
    print(f"\n==== PBO 预测的【最佳生成参数组合】参考 ====", flush=True)
    row = df.iloc[best_idx]
    print(f"ID: {row.get('id', 'N/A')}", flush=True)
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
        text_features, num_features, sampler_features, scaler, encoder = build_features(df)
        pbo_space, pca_text, pca_final = perform_two_stage_pca(text_features, num_features, sampler_features, final_dim=8)
        
        print("\n[开始前] 显示本次优化的目标/参考图片...", flush=True)
        display_images(df, [0], title="Target Image (Goal Reference)", filename="pbo_target.png")
        input("查看完目标图片后，按回车开始 PBO 循环...")
        
        best_pbo_index = run_pbo_loop(pbo_space, df, iterations=10, batch_size=4)
        
        print("\n[最终结果] 显示 PBO 找到的最佳图片...", flush=True)
        display_images(df, [best_pbo_index], title="Final Optimized Result", filename="pbo_final.png")
        
        simulate_pbo_retrieval(pbo_space, df, target_index=best_pbo_index, top_k=3)
        
        print("\n🎉 PBO 流程演示完成！", flush=True)
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{file_path}'。", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"发生错误: {e}", flush=True)