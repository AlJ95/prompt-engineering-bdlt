import random

import numpy as np
import re
import pandas as pd

if __name__ == "__main__":
    tax_split = []
    # Open Taxonomy File and Read every line (also remove \n)
    with open("./Projects/Prompt Engineering/math_taxonomy.txt") as file:
        for line in file.readlines():
            tax_split.append(re.sub("\n", "", line))

    # Create dict for better overview: 4 Nr. for different depths of taxonomic level and the topic name
    tax_dict = {i: [
        (re.split(r'\.|(?<=\d)\s', k)[:1 + i][-1] if len(re.findall(r'.{0}\d\.\d\s', k)) == 1 else "")
        for k in tax_split
    ] for i in range(5)}

    # Change name of 5. dict entry
    tax_dict["Category"] = tax_dict[4]
    tax_dict.pop(4)

    # Add new dict entry
    tax_dict["Superordinate Topic"] = []

    # Remove all entries with letters in Nr. columns
    for cat in range(4):
        tax_dict[cat] = [re.sub(r"\D", "", num) for num in tax_dict[cat]]

    # loop through dict and find the next higher level topic name
    records = []
    for i in range(len(tax_dict[0])):
        record = [tax_dict[key][i] for key in list(tax_dict.keys())[:-1]]
        records.append(record)

        superordinate_topic = ""
        for cat in reversed(range(4)):
            if record[cat] == "":
                continue
            if record[cat] == "0" or cat == 0:
                break

            superordinate_topic = [topic for topic in records
                                   if topic[cat - 1] == record[cat - 1] and topic[cat] in ["0", ""]
                                   ][-1][-1]

        tax_dict["Superordinate Topic"].append(superordinate_topic)

    results = {"X": [], "y": [], "text-davinci-002": []}

    grouped_sample_size = []
    tax_df = pd.DataFrame(tax_dict)
    tax_df.columns = ["Cat1", "Cat2", "Cat3", "Cat4", "Category", "Superordinate Topic"]

    for i in range(9):
        gathered = []
        for row in tax_df.query(f"Cat1=='{i + 1}' and Cat2 != '0'").iterrows():
            gathered.append(list(row[1])[-2:])
        if len(gathered) > 30:
            grouped_sample_size.append([])
            for _ in range(30):
                grouped_sample_size[-1].append(gathered.pop(np.random.randint(0, len(gathered) - 1)))
        else:
            grouped_sample_size.append(gathered)

    for group in grouped_sample_size:
        for item in group:
            item[1] = item[1].split(" and ")

