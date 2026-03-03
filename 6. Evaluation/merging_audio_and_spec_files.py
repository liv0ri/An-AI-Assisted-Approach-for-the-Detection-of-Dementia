import os
import shutil
import pandas as pd

# Define your base paths
base_path = r"6. Evaluation/adresso_corpus"
train_root = os.path.join(base_path, "Train")
test_root = os.path.join(base_path, "test-dist")
test_csv_path = os.path.join(test_root, "task1.csv")

# The sub-modalities you have
modalities = {
    "audio": "test-distaudio",
    "specto": "test-distspecto",
    "trans": "test-disttrans"
}

def migrate_with_class_labels():
    new_records = []
    
    for mod_name, test_folder_name in modalities.items():
        train_mod_path = os.path.join(train_root, mod_name)
        dest_path = os.path.join(base_path, test_folder_name)
        
        # We check both 'ad' and 'cn' folders inside each modality
        for category in ["ad", "cn"]:
            category_path = os.path.join(train_mod_path, category)
            
            if os.path.exists(category_path):
                print(f"Processing {mod_name}/{category}...")
                for file_name in os.listdir(category_path):
                    # Move the file
                    shutil.move(os.path.join(category_path, file_name), os.path.join(dest_path, file_name))
                    
                    # Only add to CSV list once 
                    if mod_name == "audio":
                        # Strip extension for the ID (e.g., 'adrso024.wav' -> 'adrso024')
                        file_id = os.path.splitext(file_name)[0]
                        new_records.append({
                            'ID': file_id, 
                            'Dx': "ProbableAD" if category == "ad" else "Control",
                        })
    # Update or Create the CSV
    if os.path.exists(test_csv_path):
        # Read the existing test CSV
        df_test = pd.read_csv(test_csv_path)
        
        # Create the new records dataframe
        df_new = pd.DataFrame(new_records)
        
        # Combine and drop duplicates 
        # We also reset the index to ensure it's a clean 0 to N range
        combined_df = pd.concat([df_test, df_new], ignore_index=True)
        
        # This removes any existing "unnamed" index columns that might have been saved before
        combined_df = combined_df.loc[:, ~combined_df.columns.str.contains('^Unnamed')]
        
        combined_df = combined_df.drop_duplicates(subset=['ID'], keep='first')
        
    else:
        raise FileNotFoundError(f"{test_csv_path} not found.")

    combined_df.to_csv(test_csv_path, index=False)
    print(f"Done! Moved files and updated {test_csv_path}")

if __name__ == "__main__":
    migrate_with_class_labels()