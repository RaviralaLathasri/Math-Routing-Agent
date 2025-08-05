import pickle

jee_docs = [
    "Newton's Second Law explains how force affects motion.",
    "Ohm’s Law defines the relationship between voltage, current, and resistance.",
    "The mole concept relates the amount of substance to number of particles.",
]

with open("jee_vectorstore.pkl", "wb") as f:
    pickle.dump(jee_docs, f)
