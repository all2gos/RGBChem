import pandas as pd
import os

df = pd.read_csv(f"qm7_demo.csv")
l = df.ID.tolist()

t = [x.split('_')[1] for x in l]
t = list(set(t))
valid_filenames = [f"dsgdb9nsd_{'0'*(6-len(x))}{x}.xyz" for x in t]

directory = 'data' 

all_files = os.listdir(directory)[1:]

# Step 5: Delete files that are not in the valid filenames list
for file in all_files:
    if file not in valid_filenames:
        os.remove(os.path.join(directory, file))
        print(f"Removed: {file}")

print("Cleanup complete.")
