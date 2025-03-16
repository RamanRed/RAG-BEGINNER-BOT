# RAG-BEGINNER-BOT

📌 Project Overview
The Beginner RAG Bot is an AI-powered system that reads a given story, stores it in a vector database, and answers user questions about the content. It utilizes retrieval-augmented generation (RAG) to fetch relevant information from stored documents and generate accurate responses using a large language model (LLM).

🛠️ Technologies Used
ChromaDB – To store and retrieve document embeddings efficiently.
LangChain – For document loading and text chunking.
Sentence Transformers – To generate vector embeddings for similarity searches.
Google Gemini API – To generate natural language answers from retrieved content.
Python – The core programming language for development.
🔹 How It Works
Load Story Documents 📖

The bot loads markdown (.md) or text files containing the story.
Preprocess & Store Data 🔍

The document is split into overlapping chunks using RecursiveCharacterTextSplitter.
Each chunk is converted into a vector embedding using Sentence Transformers.
The embeddings are stored in ChromaDB for efficient retrieval.
Query Processing 🤖

The user asks a question related to the story.
The bot retrieves the most relevant text chunks from ChromaDB.
The retrieved chunks are passed to the Gemini API, which formulates a coherent answer.
Generate AI-Powered Responses ✨

The Gemini API combines retrieved information with LLM reasoning to return an accurate response.
The user receives a contextually relevant answer!
🚀 Features
✅ Story-based Question Answering – Ask anything about the story, and get an AI-generated response.
✅ Efficient Retrieval – Uses semantic search for accurate chunk selection.
✅ Dynamic Content Processing – Updates stored data when a new story is added.
✅ Fast & Lightweight – Uses ChromaDB for quick vector searches.

🔮 Future Enhancements
🔹 Improve chunking strategy for better context retention.
🔹 Support multiple document formats like PDF and EPUB.
🔹 Add voice-based interaction for better user experience.
🔹 Optimize response generation using fine-tuned LLMs.
