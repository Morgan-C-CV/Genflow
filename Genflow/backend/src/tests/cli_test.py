import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import requests
from PIL import Image

# Add src to sys.path so we can import app
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

# Now import the CSR layers
from app.repositories.search_repository import SearchRepository
from app.repositories.llm_repository import LLMRepository
from app.services.search_service import SearchService
from app.core.config import settings

def display_images(df, indices, title="Images", gallery_dir=None, filename="display.png"):
    n = len(indices)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 5))
    if n == 1: axes = [axes]
    fig.suptitle(title, fontsize=16)
    
    for i, idx in enumerate(indices):
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
            
            axes[i].imshow(img)
            axes[i].set_title(f"Idx: {idx}\nID: {row.get('id', 'N/A')}")
            axes[i].axis('off')
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Image Load Error\n{e}", ha='center', va='center', fontsize=8)
            axes[i].set_title(f"Idx: {idx} (Error)")
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"[{title}] Images saved to: {os.path.abspath(filename)}")

def main():
    print("Initializing Genflow Backend CLI Test Tool...", flush=True)
    
    # 1. Initialize logic
    # Make sure we have the metadata loaded
    search_repo = SearchRepository()
    llm_repo = LLMRepository()
    search_service = SearchService(search_repo, llm_repo)
    
    df = search_repo.get_all_data()
    search_engine = search_repo.search_engine
    pbo_space = search_engine.pbo_space
    
    print(f"\nSuccessfully loaded {len(df)} images into the search engine.")
    
    # Optional: Display a target image as a reference (like v4.py)
    target_index = int(np.random.randint(0, len(df)))
    display_images(df, [target_index], title="Target Image (Goal Reference)", gallery_dir=settings.GALLERY_DIR, filename="pbo_target.png")
    input(f"\nTarget image saved to 'pbo_target.png'. Press Enter to start the PBO loop...")

    # 2. Interactive PBO Loop
    iterations = 10
    batch_size = 4
    X_train = []
    y_train = []
    consecutive_skips = 0
    
    print(f"\nStarting Interactive PBO Loop ({iterations} rounds)...")
    
    current_iter = 0
    while current_iter < iterations:
        print(f"\n--- Round {current_iter+1} / {iterations} ---")
        candidate_indices = search_engine.run_pbo_round(X_train, y_train, batch_size=batch_size, consecutive_skips=consecutive_skips)
        
        # Display candidates
        filename = f"pbo_round.png"
        display_images(df, candidate_indices, title=f"Round {current_iter+1} Candidates", gallery_dir=settings.GALLERY_DIR, filename=filename)
        
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
    
    # 4. Use SearchService to get summary and similar results
    print("\nRequesting Gemini-powered summary and similarity analysis...", flush=True)
    try:
        results_data = search_service.summarize_search_results(best_discovered_index, top_k=5)
        
        print("\n" + "="*50)
        print("GEMINI SEARCH ANALYSIS")
        print("="*50)
        print(results_data["llm_summary"])
        print("="*50)
        
        print("\nTop 5 Similar Results Found:")
        for i, res in enumerate(results_data["top_results"]):
            print(f"{i+1}. ID {res['id']} | Distance: {res['distance']} | Prompt: {res['prompt'][:60]}...")
            
        display_images(df, [best_discovered_index] + [int(df[df['id'] == r['id']].index[0]) for r in results_data['top_results']], 
                      title="Best Result + Similar Recommendations", gallery_dir=settings.GALLERY_DIR, filename="pbo_final_results.png")
        
        # 5. NEW: Generate metadata based on intent
        print("\n" + "*"*50)
        print("GENERATE NEW IMAGE PARAMETERS")
        print("*"*50)
        print("Now that we've found your style preference, what would you like to generate?")
        user_intent = input("Enter your generation intent (e.g., 'a cat drawn in crayons'): ").strip()
        
        if user_intent:
            print(f"\nGenerating metadata for: '{user_intent}'...", flush=True)
            generated_json_str = search_service.generate_image_metadata(results_data["top_results"], user_intent)
            
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
