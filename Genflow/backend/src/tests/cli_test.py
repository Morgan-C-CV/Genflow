import os
import sys
import json
from math import ceil
from io import BytesIO

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import requests
    from PIL import Image
except Exception:
    np = None
    pd = None
    plt = None
    requests = None
    Image = None

# Add src to sys.path so we can import app
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from app.agents.creative_agent import CreativeAgent

BACKEND_IMPORT_ERROR = None
try:
    from app.repositories.search_repository import SearchRepository
    from app.repositories.llm_repository import LLMRepository
    from app.services.search_service import SearchService
    from app.core.config import settings
except Exception as exc:
    SearchRepository = None
    LLMRepository = None
    SearchService = None
    settings = None
    BACKEND_IMPORT_ERROR = exc

def display_images(df, indices, title="Images", gallery_dir=None, filename="display.png", cols=4):
    n = len(indices)
    cols = max(1, min(cols, n))
    rows = int(ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.8 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)
    fig.suptitle(title, fontsize=16)
    
    for i, idx in enumerate(indices):
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        row = df.iloc[idx]
        local_filename = row.get('local_path')
        img_url = row.get('image_url')
        try:
            img = None
            if local_filename and gallery_dir:
                local_path = os.path.join(gallery_dir, local_filename)
                if os.path.exists(local_path):
                    img = Image.open(local_path)
            
            if img is None and img_url:
                try:
                    response = requests.get(img_url, timeout=5)
                    img = Image.open(BytesIO(response.content))
                except Exception as e:
                    print(f"Failed to fetch image from URL: {e}")
            
            if img is None:
                raise FileNotFoundError(f"Cannot find image for ID {row.get('id')}")
            
            ax.imshow(img)
            ax.set_title(f"#{i+1} | ID: {row.get('id', 'N/A')}", fontsize=9)
            ax.axis('off')
        except Exception as e:
            ax.text(0.5, 0.5, f"Image Load Error\n{e}", ha='center', va='center', fontsize=8)
            ax.set_title(f"#{i+1} (Error)", fontsize=9)
            ax.axis('off')

    total_slots = rows * cols
    for i in range(n, total_slots):
        r = i // cols
        c = i % cols
        axes[r][c].axis('off')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"[{title}] Images saved to: {os.path.abspath(filename)}")


def prompt_user_intent(agent: CreativeAgent):
    while True:
        user_intent = input("请先告诉 agent 你的创作意图（例如：我想生成一张关于猫的插画）：\n> ").strip()
        if user_intent:
            break
        print("创作意图不能为空。")

    plan = agent.analyze_intent(user_intent)
    print("\n[Agent ReAct] 当前判断：")
    print(plan.reasoning_summary)
    print(f"下一步动作: {plan.next_action}")

    clarified_intent = user_intent
    clarification_rounds = 0
    while plan.next_action == "ask_user" and clarification_rounds < 2:
        print("\n需要补充的信息：")
        print(agent.build_clarification_prompt(plan))
        for question in plan.clarification_questions[:3]:
            answer = input(f"{question}\n> ").strip()
            if answer:
                clarified_intent += f" | {answer}"
        clarification_rounds += 1
        plan = agent.analyze_intent(clarified_intent)
        print("\n[Agent ReAct] 更新后的判断：")
        print(plan.reasoning_summary)
        print(f"下一步动作: {plan.next_action}")

    return clarified_intent, plan


def print_wall_summary(agent: CreativeAgent, wall, df):
    print("\n" + "=" * 60)
    print("16 图初始发散墙")
    print("=" * 60)
    print(agent.describe_wall(wall))
    for group_idx, group in enumerate(wall.groups, start=1):
        print(f"\nGroup {group_idx}:")
        for idx_in_group, df_idx in enumerate(group, start=1):
            row = df.iloc[df_idx]
            prompt = str(row.get('prompt', ''))[:90].replace("\n", " ")
            print(f"  [{(group_idx - 1) * 4 + idx_in_group:02d}] ID: {row.get('id', 'N/A')} | {prompt}...")

def main():
    print("Initializing Genflow Backend CLI Test Tool...", flush=True)

    if any(dep is None for dep in [np, pd, plt, requests, Image]):
        print("One or more runtime visualization/data dependencies are missing in this environment.")
        print("The CLI test module can be imported safely, but the interactive image workflow cannot run here.")
        return

    if BACKEND_IMPORT_ERROR is not None:
        print("Backend dependencies are not fully available in this environment.")
        print(f"Import error: {BACKEND_IMPORT_ERROR}")
        print("The new agent layer can still be imported, but the full gallery search pipeline needs the missing runtime dependency.")
        return
    
    # 1. Initialize logic
    # Make sure we have the metadata loaded
    search_repo = SearchRepository()
    llm_repo = LLMRepository()
    search_service = SearchService(search_repo, llm_repo)
    creative_agent = CreativeAgent()
    
    df = search_repo.get_all_data()
    search_engine = search_repo.search_engine
    pbo_space = search_engine.pbo_space
    
    if getattr(search_engine, "pca_final", None) is None and getattr(search_engine, "text_features", None) is None:
        print("\nPBO embedding source: precomputed database (no runtime embedding).", flush=True)
    else:
        print("\nPBO embedding source: runtime embedding (slow). Set SUPABASE_URL/SUPABASE_KEY to use the precomputed v4 database.", flush=True)

    print(f"\nSuccessfully loaded {len(df)} images into the search engine.")

    # 2. ReAct-based creative intake and divergent candidate retrieval
    clarified_intent, intent_plan = prompt_user_intent(creative_agent)
    resources = creative_agent.load_resources()
    resource_recommendation = creative_agent.recommend_resources(intent_plan, resources)
    expansions = creative_agent.build_axis_expansions(
        clarified_intent,
        intent_plan,
        resources,
        recommendation=resource_recommendation,
    )

    print("\n[Resource RAG] 资源推荐：")
    print(f"- Checkpoint: {resource_recommendation.checkpoint}")
    print(f"- Sampler: {resource_recommendation.sampler}")
    print(f"- LoRAs: {', '.join(resource_recommendation.loras) if resource_recommendation.loras else 'none'}")
    print(f"- Reasoning: {resource_recommendation.reasoning_summary}")
    print("\n[Agent ReAct] 4 个发散查询已生成：")
    for i, expansion in enumerate(expansions, start=1):
        print(f"  {i}. {expansion.label}")
        print(f"     {expansion.prompt[:220]}...")

    wall = creative_agent.build_candidate_wall(search_engine, expansions, per_query_k=4, top_k=12)
    if len(wall.flat_indices) < 16:
        raise RuntimeError(f"Initial wall only produced {len(wall.flat_indices)} candidates; expected 16.")

    print_wall_summary(creative_agent, wall, df)
    display_images(
        df,
        wall.flat_indices[:16],
        title="Initial 16-Candidate Wall",
        gallery_dir=settings.GALLERY_DIR,
        filename="pbo_initial_wall.png",
        cols=4,
    )

    print("\n请选择 16 张图中最喜欢的 3 张，输入编号（例如：2 7 14）。")
    selected_raw = input("> ").strip()
    selected_numbers = []
    try:
        selected_numbers = [int(x) for x in selected_raw.split() if x.isdigit()]
    except ValueError:
        selected_numbers = []
    selected_numbers = [n for n in selected_numbers if 1 <= n <= 16][:3]
    if len(selected_numbers) < 1:
        selected_numbers = [1]
    selected_indices = [wall.flat_indices[n - 1] for n in selected_numbers]

    X_train, y_train = creative_agent.build_training_labels(
        pbo_space=pbo_space,
        selected_indices=selected_indices,
        wall=wall,
        selected_score=1.0,
        unselected_score=0.5,
    )
    consecutive_skips = 0

    print("\n初始训练集已构建：")
    print(f"- Selected positives: {len(selected_indices)}")
    print(f"- All 16 initial samples entered into GP: {len(X_train)}")

    # 3. Exploitation phase
    iterations = 8
    batch_size = 4
    
    print(f"\nStarting 4-image PBO exploitation loop ({iterations} rounds)...")
    
    current_iter = 0
    while current_iter < iterations:
        print(f"\n--- Round {current_iter+1} / {iterations} ---")
        candidate_indices = search_engine.run_pbo_round(X_train, y_train, batch_size=batch_size, consecutive_skips=consecutive_skips)
        
        # Display candidates
        filename = f"pbo_round.png"
        display_images(df, candidate_indices, title=f"Round {current_iter+1} Candidates", gallery_dir=settings.GALLERY_DIR, filename=filename, cols=4)
        
        print(f"Please provide feedback for the following {batch_size} images (Check popup window):")
        for idx, c_idx in enumerate(candidate_indices):
            row = df.iloc[c_idx]
            print(f"  [{idx+1}] ID: {row.get('id', 'N/A')} | Prompt: {row.get('prompt', '')[:80]}...")
            
        while True:
            try:
                line = input(f"Enter the numbers for [Best] and [Worst] (e.g., 1 4), or enter 0 to skip: ")
                line = line.strip()
                if line == '0':
                    consecutive_skips += 1
                    for c_idx in candidate_indices:
                        X_train.append(pbo_space[c_idx])
                        y_train.append(0.0)
                    print(f"Round skipped. Total consecutive skips: {consecutive_skips}. Penalty recorded (This round is not counted).", flush=True)
                    break
                
                parts = line.split()
                if len(parts) == 2:
                    best_idx = int(parts[0]) - 1
                    worst_idx = int(parts[1]) - 1
                    if 0 <= best_idx < batch_size and 0 <= worst_idx < batch_size:
                        consecutive_skips = 0
                        for idx, c_idx in enumerate(candidate_indices):
                            X_train.append(pbo_space[c_idx])
                            if idx == best_idx: y_train.append(1.0)
                            elif idx == worst_idx: y_train.append(0.0)
                            else: y_train.append(0.5)
                        current_iter += 1
                        break
                print(f"Please enter two numbers between 1 and {batch_size}, or enter 0 to skip.")
            except ValueError:
                print("Please enter valid numbers or 0.")
    
    # 3. Final Result and Backend Summarization
    print("\nOptimization finished. Searching for the best match using the CSR backend logic...", flush=True)
    
    # Calculate final best index from PBO (simplified)
    # Use Gaussian process logic or just find max reward
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    
    kernel = 1.0 * Matern(length_scale=1.0, nu=1.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1, n_restarts_optimizer=5)
    gp.fit(np.array(X_train), np.array(y_train))
    final_scores = gp.predict(pbo_space)
    best_discovered_index = int(np.argmax(final_scores))
    
    print(f"\n==== Optimal Image Found ====")
    row = df.iloc[best_discovered_index]
    print(f"ID: {row.get('id', 'N/A')}")
    print(f"Prompt: {row.get('prompt')}")
    
    # 4. Get similar results and skip summary
    print("\nRetrieving similar results for generation...", flush=True)
    try:
        top_results = search_repo.search_by_index(best_discovered_index, top_k=5)
        
        print("\nTop 5 Similar Results Found:")
        for i, res in enumerate(top_results):
            print(f"{i+1}. ID {res['id']} | Distance: {res['distance']:.4f} | Prompt: {res['prompt'][:60]}...")
            
        display_images(df, [best_discovered_index] + [int(df[df['id'] == r['id']].index[0]) for r in top_results], 
                      title="Best Result + Similar Recommendations", gallery_dir=settings.GALLERY_DIR, filename="pbo_final_results.png", cols=3)
        
        # 5. NEW: Generate metadata based on intent
        print("\n" + "*"*50)
        print("GENERATE NEW IMAGE PARAMETERS")
        print("*"*50)
        print("Now that we've found your style preference, what would you like to generate?")
        user_intent = input("Enter your generation intent (e.g., 'a cat drawn in crayons'): ").strip()
        
        if user_intent:
            print(f"\nGenerating metadata for: '{user_intent}'...", flush=True)
            generated_json_str = search_service.generate_image_metadata(top_results, user_intent)
            
            try:
                # Verify it's valid JSON and pretty print it
                generated_json = json.loads(generated_json_str)
                print("\nSuccessfully Generated Metadata:")
                print(json.dumps(generated_json, indent=2, ensure_ascii=False))
                
                # Save to a file for convenience
                with open("generated_metadata.json", "w", encoding="utf-8") as f:
                    json.dump(generated_json, f, indent=2, ensure_ascii=False)
                print(f"\nMetadata saved to: {os.path.abspath('generated_metadata.json')}")
            except json.JSONDecodeError:
                print("\nError: The LLM did not return a valid JSON string.")
                print("Raw Response:")
                print(generated_json_str)
        
    except Exception as e:
        print(f"Error during backend summarization: {e}")
        import traceback
        traceback.print_exc()

    input("\nPress Enter to exit the test tool.")
    plt.close('all')

if __name__ == "__main__":
    main()
