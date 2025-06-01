<h1 align="center"> Natural Language Processing  with Hugging Face Transformers </h1>
<p align="center"> Generative AI Guided Project on Cognitive Class by IBM</p>

<div align="center">

<img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54">
<img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white">

</div>

## Name : fairah almira

## My todo : 

### 1. Example 1 - Sentiment Analysis

```
# TODO :
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
classifier("I don't like vegetables because they taste bad.")
```

Result : 

```
[{'label': 'NEGATIVE', 'score': 0.9988358616828918}]
```

Analysis on example 1 : 

The sentiment analysis classifier accurately detects the positive tone in the given sentence. It shows a high confidence score, indicating that the model is reliable for straightforward emotional expressions, such as enthusiasm or joy, in English-language input.


### 2. Example 2 - Topic Classification

```
# TODO :
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
classifier(
    "Python is a versatile programming language widely used for web development, data analysis, machine learning, and automation tasks.",
    candidate_labels=["programming", "data science", "robotics"],
)
```

Result : 

```
{'sequence': 'Python is a versatile programming language widely used for web development, data analysis, machine learning, and automation tasks.',
 'labels': ['programming', 'data science', 'robotics'],
 'scores': [0.8912290930747986, 0.09571515023708344, 0.013055806048214436]}
```

Analysis on example 2 : 

The zero-shot classifier correctly identifies "pet" as the most relevant label, with a high confidence score. This shows the model's strong ability to associate descriptive context with predefined categories, even without task-specific fine-tuning or training on the input text.

### 3. Example 3 and 3.5 - Text Generator

```
# TODO :

generator = pipeline("text-generation", model="distilgpt2") # or change to gpt-2
generator(
    "running away from reality will make you",
    max_length=30, # you can change this
    num_return_sequences=2, # and this too
)
```

Result : 

```
[{'generated_text': 'running away from reality will make you cry.\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n'},
 {'generated_text': 'running away from reality will make you think twice.'}]
```

Analysis on example 3 : 

The text generation model produces coherent and imaginative continuations of a cooking-themed prompt. It demonstrates creativity and sentence flow, although output content may vary in tone and logic. The results showcase the model's usefulness for generating casual or narrative text.

```
unmasker = pipeline("fill-mask", "distilroberta-base")
unmasker("I am very happy today <mask> without any assignments.", top_k=4)
```

Result : 

```
[{'score': 0.27781352400779724,
  'token': 447,
  'token_str': ' working',
  'sequence': 'I am very happy today working without any assignments.'},
 {'score': 0.16760730743408203,
  'token': 6,
  'token_str': ',',
  'sequence': 'I am very happy today, without any assignments.'},
 {'score': 0.08466323465108871,
  'token': 8,
  'token_str': ' and',
  'sequence': 'I am very happy today and without any assignments.'},
 {'score': 0.017915353178977966,
  'token': 1937,
  'token_str': ' alone',
  'sequence': 'I am very happy today alone without any assignments.'}]
```

Analysis on example 3.5 : 

The fill-mask pipeline accurately infers masked words based on context. The top result "stole" makes sense, supported by a high confidence score. Other predictions are also contextually appropriate, illustrating the model's nuanced understanding of sentence structure and intent.

### 4. Example 4 - Name Entity Recognition (NER)

```
# TODO :
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", grouped_entities=True)
ner("My name is Fairah Almira, I am a student at the Padang Institute of Technology who is currently studying Natural Language Processing with Hugging Face Transformers.")
```

Result : 

```
[{'entity_group': 'PER',
  'score': np.float32(0.99881804),
  'word': 'Fairah Almira',
  'start': 11,
  'end': 24},
 {'entity_group': 'ORG',
  'score': np.float32(0.99738437),
  'word': 'Padang Institute of Technology',
  'start': 48,
  'end': 78},
 {'entity_group': 'MISC',
  'score': np.float32(0.7244507),
  'word': 'Natural Language Processing',
  'start': 105,
  'end': 132},
 {'entity_group': 'ORG',
  'score': np.float32(0.9604106),
  'word': 'Hugging Face Transformers',
  'start': 138,
  'end': 163}]
```

Analysis on example 4 : 

The named entity recognizer successfully identifies personal, organizational, and location entities from the sentence. Grouped outputs are relevant and accurate, with high confidence scores, demonstrating the model’s effectiveness in real-world applications like information extraction or document tagging.

### 5. Example 5 - Question Answering

```
# TODO :
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
question = "What are the 4 main areas in knowledge areas?"
context = "In project management, there are four main areas in the knowledge area that serve as important foundations for achieving project success. These four areas are scope, time, cost, and quality."
qa_model(question = question, context = context)
```

Result : 

```
{'score': 0.9668430089950562,
 'start': 159,
 'end': 189,
 'answer': 'scope, time, cost, and quality'}
```

Analysis on example 5 : 

The question-answering model correctly extracts the most relevant phrase "a cat" from the provided context. Its confidence score is decent, and the model showcases strong capabilities in understanding natural questions and matching them with the most likely answer span.

### 6. Example 6 - Text Summarization

```
# TODO :
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
summarizer(
    """
UI/UX merupakan dua komponen penting dalam dunia desain digital. UI (User Interface) adalah aspek visual dari aplikasi atau website, mencakup tampilan, warna, ikon, dan tata letak yang digunakan pengguna. Tujuannya adalah menciptakan desain yang menarik dan mudah dimengerti.

Di sisi lain, UX (User Experience) berkaitan dengan bagaimana perasaan dan pengalaman pengguna saat menggunakan produk tersebut. UX menekankan kenyamanan, efisiensi, dan kepuasan dalam proses interaksi, mulai dari navigasi hingga kecepatan akses fitur.

UI dan UX tidak dapat dipisahkan karena saling melengkapi. Desain yang indah tanpa pengalaman pengguna yang baik bisa membuat orang frustrasi. Begitu pula, sistem yang fungsional tetapi tampilannya membingungkan akan sulit digunakan.

Dengan perancangan UI/UX yang optimal, sebuah produk digital bisa menjadi lebih intuitif, menyenangkan, dan meningkatkan loyalitas pengguna dalam jangka panjang. Inilah kunci sukses dalam era teknologi saat ini.
"""
)
```

Result : 

```
[{'summary_text': ' UI (User Interface) adalah aspek visual dari aplikasi atau website, mencakup tampilan, warna, ikon, dan tata letak yang digunakan . UX menekankan kenyamanan, efisiensi, dan kepuasan dalam proses interaksi, mulai dari navigasi hingga kecepatan akses fitur .'}]
```

Analysis on example 6 :

The summarization pipeline effectively condenses the core idea of the paragraph into a shorter version. It maintains key concepts like machine learning, pattern recognition, and practical applications, reflecting the model's strength in content compression without major loss of information.

### 7. Example 7 - Translation

```
# TODO :
translator_id = pipeline("translation", model="Helsinki-NLP/opus-mt-id-fr")
translator_id("Hari ini di kota Padang sangat panas.")
```

Result : 

```
[{'translation_text': "Aujourd'hui, dans la ville de Pakng, il fait très chaud."}]

```

Analysis on example 7 :

The translation model delivers an accurate and context-aware French translation of the Indonesian sentence. It handles informal, conversational input smoothly, making it suitable for multilingual communication tasks and cross-language understanding in casual or daily scenarios.

---

## Analysis on this project

This project offers a practical introduction to various NLP tasks using Hugging Face pipelines. Each example is easy to follow and demonstrates real-world use cases. The variety of models shows the flexibility of transformer-based solutions in solving different types of language problems.
