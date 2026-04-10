import json
import os

#files in system
pdf_files=[]
for i, filename in enumerate(os.listdir("./data")):
    if filename.endswith('.pdf'):
        pdf_files.append(filename)
        print(f"File {i}: {filename}")

if len(pdf_files) == 0:
    print("No files found. Add some PDFs.")
    exit(1)

print(f"Total Files:{len(pdf_files)}")


"""
Common Categories:

- HR           (employee handbook, benefits, polices)
- ENGINEERING  (code guidelines, API docs, tech specs)
- FINANCE       (expence polices, budgets, invoices)
- LEGAL        (contracts, terms of service)
- GENERAL      (onboarding, FAQs, misc)

"""


#assigning category
labels = {
    "benefits_guide.pdf": "HR",
    "employee_handbook.pdf": "HR",
    "engineering_handbook.pdf": "ENGINEERING",
    "finance_policies.pdf": "FINANCE",
    "onboarding_guide.pdf": "GENERAL",
    "financial_controls.pdf":"FINANCE"
}

#label distribution
from collections import Counter
category_count = Counter(labels.values())
for category, count in category_count.items():
    print(f"{category:15s}: {count} documents")


#saving in json file
os.makedirs("./data/processed", exist_ok=True)
output_file = "./data/processed/document_labels.json"
with open (output_file, "w") as f:
    json.dump(labels, f, indent=2)

print(f"""Labels saved to: {output_file}
      Total Labels: {len(labels)}""")