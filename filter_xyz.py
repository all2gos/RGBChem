import pandas as pd
import os

if DB == 'qm7_demo.csv':
    df = pd.read_csv(f"qm7_demo.csv")
elif DB == 'qm8_demo.csv'::
    df = pd.read_csv(f"qm8_demo.csv")
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

print(f"Cleanup complete. Data folder now contain {len(os.listdir('data'))}")
