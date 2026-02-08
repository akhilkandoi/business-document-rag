from llama_index.core import SimpleDirectoryReader
import pandas as pd
import json
import os
import re

with open("./data/processed/document_labels.json") as f:
    labels=json.load(f)

print(f"Labels loaded of {len(labels)} size.\n")
print(f"Categories: {set(labels.values())}\n")

print("Text Extraction\n\n")

training_data=[]
total_chunks=0

def clean_data(text):
    text = re.sub(r'[^a-zA-Z0-9\s]',' ', text)
    text = ' '.join(text.split())
    text = text.lower()
    return text


for filename, label in labels.items():
    filepath = f"./data/{filename}"

    if not os.path.exists(filepath):
        print(f"{filename} not found")
        continue
    print(f"\n Processing: {filename}")
    print(f"Category: {label}")

    try: 
        docs = SimpleDirectoryReader(input_files=[filepath]).load_data()
        chunk_count=0
        for doc in docs:
            text_sample = doc.text[:500].strip()
            if len(text_sample) < 50:
                continue
            
            cleaned_text = clean_data(text_sample)

            training_data.append({
                "text":cleaned_text,
                "filename" : filename,
                "label":label
            })
            chunk_count+=1
        
        total_chunks+=chunk_count
        print(f"extracted {chunk_count} text chunks")
    except Exception as e:
        print(f"Error: {e}")

#creating dataset

df = pd.DataFrame(training_data)
print(f"Total data: {len(df)}")
print(df["label"].value_counts())

print("SAMPLE DATA\n\n")
print(f"""
\nFirst Trainging sample:
Category: {df.iloc[0]['label']}
Text preview: {df.iloc[0]['text'][:200]}...

""")

#save to csv
output_file = "./data/processed/training__data.csv"
df.to_csv(output_file, index=False)

print(f"Saved in {output_file}")