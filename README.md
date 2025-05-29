# Measuring-Semantic-Similarity

In this assignment, you will be using semantic similarity to identify relevant responses to user questions in a conversational context.
The goal, for a given question, is to find the most similar question and return its corresponding response.
This involves using different text representation techniques to quantify the semantic similarity between questions.

### 📚 Data

The data you will be working with consists of real-world conversations between users and a large language model.
Each entry in the data consists of a user question and a model-generated response, identified by a unique conversation ID.

- `conversation_id`: 1983b94c7d5849aea507a9a8fb467af8
- `user_prompt`: What is the primary color of a tiger?
- `model_response`: The primary color of a tiger is orange.

The data is split into three splits.
The TRAIN and DEV splits, released with the start of the assignment, contain prompts and responses. Use these to develop your retrieval method. 
The TEST set contains only prompts. This is what you will be assessed on.

Please note that these are real conversations between LLMs and users. 
We filtered out likely-inappropriate content, but some prompts and responses may still be sensitive or offensive.

### 📝 Task

Your task is to implement a **retrieval method** that, when given a new TEST question (`user_prompt`), finds **the most similar question from the TRAIN + DEV dataset** and returns the corresponding answer of this question (`model_response`).
To achieve this, you need to first represent all questions in TRAIN and DEV in some way, and then use similarity metrics (like cosine similarity) to compare them to the given TEST question.

For the purpose of this assignment, you should **assume that the most similar question will have the most relevant response**, even if this is not always the case.
You will be assessed based on how similar the response you retrieved is to the actual response (see below).

### ⚙️ Implementation

You will need to use different text representation methods to convert user prompts into a numerical format, for which you can then calculate similarity.
There will be three Tracks.
Each track corresponds to a different text representation method.

- **Track 1: Discrete Text Representation**. Choose a discrete representation method we have seen in class, such as n-gram word or character-level representations, Count Vectorizer, or TF-IDF.

- **Track 2: Distributed Static Text Representation**. Choose a static distributed representation method we have seen in class, such as Word2vec, Doc2Vec, or pretrained embeddings like FastText.

- **Track 3 (✨BONUS✨): Open Text Representation**. In this track, you can use any combination of the two previous or another representation method. This could include methods not covered in class.

**You must develop solutions for Track 1 and Track 2**.
Track 3 is optional.

### 🏅 Assessment

Tracks will have equal weighting in the final grade.
If you submit to Track 3, **we will choose the best two tracks for evaluating your assignment**.

For each track, you will be assessed based on the BLEU score of the responses you retrieve for the TEST set relative to the actual responses (which you do not know).
The BLEU score is a word-overlap metric developed for machine translation that is used for all kinds of language generation tasks.
It measures the 1–4-gram overlap between a system's output and at least one reference (the "correct" response).
We have provided an example for how to calculate BLEU below, which you should use in developing your retrieval method using the TRAIN and DEV sets.

You will be assessed on the TEST set, for which you will not have access to the responses.
For each prompt in the TEST set, you will need to retrieve the most similar prompt from the combination of the TRAIN and DEV sets.

For each track, your submission will be a CSV file with two columns:
`conversation_id`, which is the conversation ID of the prompt in the TEST set, and
`response_id`, which is the conversation ID of the most similar prompt in the combination of the TRAIN and DEV sets.
It is extremely important that you follow this format, and return only IDs, not text responses.

You also have to submit a brief description of the methodology you used for each track (max 100 words per track).
It is very important that you stick to the "allowed" methods for each track.
We will check your code, and if you are not, you will receive a 0 for that track.

### 📥 Submission Instructions

Follow these instructions to submit your assigment on BlackBoard:

1. **File structure**: Ensure that your submission is a .zip file, and that it contains the following items with exactly these specified names:
  - `track_1_test.csv`: A CSV file with two columns (conversation_id, response_id) for Track 1.
  - `track_2_test.csv`: A CSV file with two columns (conversation_id, response_id) for Track 2.
  - `track_3_test.csv` (optional): A CSV file with two columns (conversation_id, response_id) for Track 3.
  - `description.txt`: A brief description of the methodology you used for each track (max 100 words per track).
  - `/code`: A folder containing all your code for the assignment. This code needs to be well-documented, and fully and easily reproducible by us. If your code is too large, include a README file with Google Drive link.
2. **Submission**: Upload the .zip file to the BlackBoard Assignment 2 section.
3. **Deadline**: Please refer to BlackBoard.
