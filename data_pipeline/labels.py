# Adinkra class labels

labels = {
    "nsoroma": 0,
    "gye_nyame": 1,
    "sankofa": 2,
    "nyame_dua": 3
}


# Reverse dictionary
# Useful for prediction

label_names = {
    0: "nsoroma",
    1: "gye_nyame",
    2: "sankofa",
    3: "nyame_dua"
}

print(labels["gye_nyame"])  # Output: 1
print(label_names[1])  # Output: gye_nyame