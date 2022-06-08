# prompt-engineering-bdlt
Mini Project: Prompt Engineering GPT-3,
Module: Big Data and Language Technologies,
Institution: University Leipzig

## Motivation
Use an existing taxonomy and extract topics and their superordinate ones. 

## Methods
Take topic names, insert them into following phrases and pass those as inputs into the GPT-3 text-davinci-002 model.

        'The superordinate topic for "{}" is called',
        'In mathematics, the hypernym for "{}" is called',
        'In mathematics, the superordinate topic for "{}" is called',
        'Within the mathematical taxonomy, the superordinate topic for "{}" is called',
        'Mathematical topics are strongly connected, for example the hypernym of {} is called'

Calculate a similarity score with use of Levenshtein distance.
It does not matter if the superordinate topic is the next parent or an even higher level superordinate topic.
