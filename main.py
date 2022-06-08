import random
import time
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from Levenshtein import distance
import openai
import os
import numpy as np
import re
import pandas as pd

if __name__ == "__main__":
    """
                            PREPARATION
    """
    PATH = "./Projects/Prompt Engineering"


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

        superordinate_topic = []
        for cat in reversed(range(4)):
            if record[cat] == "":
                continue
            if record[cat] == "0" or cat == 0:
                break
            else:
                superordinate_topic.append([topic for topic in records
                                       if topic[cat - 1] == record[cat - 1] and topic[cat] in ["0", ""]
                                       ][-1][-1])

        tax_dict["Superordinate Topic"].append(" and ".join(superordinate_topic))

    # Preparing Dataset. Choose randomly 30 topics per top-level topic (if it has more than 30 topics)
    grouped_sample_size = []
    tax_df = pd.DataFrame(tax_dict)
    tax_df.columns = ["Cat1", "Cat2", "Cat3", "Cat4", "Category", "Superordinate Topic"]

    for i in range(9):
        gathered = []
        top_level_category = tax_df.query(f"Cat1=='{i + 1}' and Cat2 == '0'").iloc[-1].loc["Category"]
        for row in tax_df.query(f"Cat1=='{i + 1}' and Cat2 != '0'").iterrows():
            gathered.append([x for x in list(row[1])[-2:]] + [top_level_category])
        if len(gathered) > 30:
            grouped_sample_size.append([])
            for _ in range(30):
                grouped_sample_size[-1].append(gathered.pop(np.random.randint(0, len(gathered) - 1)))
        else:
            grouped_sample_size.append(gathered)

    for group in grouped_sample_size:
        for item in group:
            item[1] = item[1].split(" and ")

    # Prepare 5 standard phrases to ask text-davinci-002 model
    PHRASES = [
        'The superordinate topic for "{}" is called',
        'In mathematics, the hypernym for "{}" is called',
        'In mathematics, the superordinate topic for "{}" is called',
        'Within the mathematical taxonomy, the superordinate topic for "{}" is called',
        'Mathematical topics are strongly connected, for example the hypernym of {} is called'
    ]

    """
    .
    .
                                Model Execution
    .
    .
    """
    results = {"X": [], "y": [], "text-davinci-002": [], "max-str-match-score": [], "top_level_category": []}
    # Load your API key from an environment variable or secret management service
    load_dotenv("./Projects/Prompt Engineering/.env")
    openai.api_key = os.getenv("openai")

    for tl_cat in grouped_sample_size:
        for inp in tl_cat:
            for phrase in PHRASES:
                # 60 requests per minute allowed
                time.sleep(1)

                # Extract arguments
                X = phrase.format(inp[0])
                top_level_cat = inp[2]
                y = inp[1]

                # API request
                # response = openai.Completion.create(
                #     model="text-davinci-002",
                #     prompt=X,
                #     temperature=0,
                #     max_tokens=40,
                #     stop="\n"
                # )

                # Postprocessing output
                response = re.sub("[(^\s)(\s$)\"\.\n]", "",
                                  response["choices"][0]["text"])

                # Similarity Function
                max_str_sim_len_ratio = 1 - min([distance(response, target)/max([len(response), len(target)])
                                                 for target in y])
                # Append result to dict
                results.update({
                    "X": results["X"] + [X],
                    "y": results["y"] + [y],
                    "text-davinci-002": results["text-davinci-002"] + [response],
                    "max-str-match-score": results["max-str-match-score"] + [max_str_sim_len_ratio],
                    "top_level_category": results["top_level_category"] + [top_level_cat]
                })

    results = pd.DataFrame(results)
    results["text-davinci-002"] = results["text-davinci-002"].str.replace(r"[\"\.]", "", regex=True)
    results["phrase"] = results.X.str.replace("\".*\"|(?<=hypernym of ).*(?= is called)", "{TOPIC}", regex=True)

    results.to_excel("./Projects/Prompt Engineering/results.xlsx")

    """
    .
    .
                                Model Evaluation
    .
    .
    """

    results = pd.read_excel(f"{PATH}/results.xlsx").iloc[:, 1:]

    res_by_tl_cat = results.groupby(["top_level_category", "phrase"], as_index=False).mean()
    res_by_tl_cat.to_excel(f"{PATH}/model_eval.xlsx", sheet_name="analysed")


    neat_labels = [
        'The superordinate topic\nfor "{}" is called',
        'In mathematics, the hypernym\nfor "{}" is called',
        'In mathematics, the super-\nordinate topic for "{}"\nis called',
        'Within the mathematical\ntaxonomy, the superordinate\ntopic for "{}" is called',
        'Mathematical topics are\nstrongly connected, for example\nthe hypernym of {} is called'
    ]
    colors = ["r-", "y-", "b-", "g-", "m", 'tab:pink', 'tab:olive', 'tab:cyan', 'tab:orange']

    plt.rcParams['axes.facecolor'] = 'none'
    plt.figure(figsize=(12, 5))

    for cat, color in zip(sorted(results["top_level_category"].unique().tolist()), colors):
        plt.plot(neat_labels,
                 res_by_tl_cat.sort_values(["top_level_category", "phrase"])\
                 .query(f"top_level_category == '{cat}'")["max-str-match-score"].to_list(),
                 color, label=cat)
    plt.legend(loc=(1.04,0))
    # plt.ylim((0, 1))
    plt.xticks(rotation=45 , ha='right')
    plt.ylabel("String Similarity / Length of Strings")
    plt.gcf().subplots_adjust(bottom=0.45, left=0.2, right=0.5)
    plt.savefig(f"{PATH}/Model.png")
    plt.show()
