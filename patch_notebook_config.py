
import json
import re

NOTEBOOK_PATH = 'notebooks/NeuroNautilus_Training.ipynb'

def patch_notebook():
    try:
        with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        print(f"📖 Loaded {NOTEBOOK_PATH}")
        
        updated = False
        
        # We need to find the cell defining config variables
        # Look for 'TOTAL_TIMESTEPS ='
        
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                if 'TOTAL_TIMESTEPS =' in source and 'MODEL_NAME =' in source:
                    print("🔍 Found Config Cell!")
                    
                    # Update Timesteps
                    new_source = re.sub(
                        r'TOTAL_TIMESTEPS\s*=\s*[\d_e]+', 
                        'TOTAL_TIMESTEPS = 5_000_000', 
                        source
                    )
                    
                    # Update Model Name
                    new_source = re.sub(
                        r"MODEL_NAME\s*=\s*['\"].*?['\"]", 
                        "MODEL_NAME = 'ppo_neurotrader_v2'", 
                        new_source
                    )
                    
                    if new_source != source:
                        cell['source'] = new_source.splitlines(keepends=True)
                        updated = True
                        print("✅ Updated Config Cell")
                    else:
                        print("⚠️ Config appears already updated or regex mismatch.")
                        
        if updated:
            with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=2)
            print(f"💾 Saved updated notebook to {NOTEBOOK_PATH}")
        else:
            print("⚠️ No changes made to notebook.")
            
    except Exception as e:
        print(f"❌ Error patching notebook: {e}")

if __name__ == "__main__":
    patch_notebook()
