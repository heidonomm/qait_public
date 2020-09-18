import os

out = open("vocabularies/act2id.txt", "w+", encoding="utf-8")

openable_entities = ["fridge", "oven", "toolbox", "door", "gate"]

with open("vocabularies/entities.txt") as f:
    all_entities = f.read().splitlines()

    for entity in all_entities:
        for openab in openable_entities:
            if openab in entity:
                out.write(f"open {entity}\n")

out.close()
