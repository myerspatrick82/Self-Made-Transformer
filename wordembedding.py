from nltk.corpus import stopwords

# pre process
file_path = "out.txt"
print("opening file")
with open(file_path, 'r') as file:
    text = file.read()

def generate_cbows(text, window_size):
    print(len(text))
    text = text.lower()
    words = text.split()

    words = [word for word in words if word.isalpha()]

    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    cbows = []

    for i, target_word in enumerate(words):
        context_words = words[max(0, i - window_size):i] + words[i + 1:i + window_size + 1]
        if len(context_words) == window_size * 2:
            cbows.append((context_words, target_word))
    return cbows

cbows = generate_cbows(text, window_size=3)

for context_words, target_word in cbows[:10]:
    print(f'Context words: {context_words}, Target word: {target_word}')
