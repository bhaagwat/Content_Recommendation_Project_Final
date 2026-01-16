🎬 **MovieSalah**
MovieSalah is a context-aware movie recommendation system designed to deliver personalised, intent-driven, and explainable recommendations for streaming platforms. It understands natural language queries, learns from user watch history, adapts over time, and explains why each movie is recommended.
The system combines semantic retrieval, deep learning–based collaborative filtering, large language models, and reinforcement learning to optimise both short-term relevance and long-term user engagement.


🚀 **Key Features**
Natural language movie queries
Semantic intent understanding beyond keywords
Personalised recommendations from user's watch history
Reinforcement learning–based ranking optimisation
LLM-generated explanations for transparency
Interactive Streamlit application


🧠 **System Overview**
The recommendation pipeline works as follows:
User query → LLM-based intent parsing → Semantic retrieval (vector database) → BiLSTM + attention (collaborative filtering) → Reinforcement learning ranker → Streamlit UI with explanations


🛠️ **Tech Stack**
Programming & UI: Python, Streamlit
Machine Learning: PyTorch, NumPy, Pandas
NLP & LLMs: Sentence-Transformers, LangChain, Ollama (llama3.2)
Vector Database: ChromaDB



📊 **Model Performance**
Collaborative filtering model results:
Top-10 Accuracy: ~16.8%
Top-20 Accuracy: ~25.9%
These results significantly outperform random baselines given the large movie vocabulary.



📂 **Project Structure**
collab-model/
├── agents/          # LLM & RL agents
├── data/            # Data processing
├── models/          # Model architectures
├── evaluation_results/
├── user/            # User interaction logic
├── app.py           # Streamlit app
├── config.py
└── requirements.txt




▶️ **Running the Application**
pip install -r requirements.txt
streamlit run app.py
Ollama must be running locally for query understanding and recommendation explanations.


📦 **Model & Dataset**
The trained model and full dataset are not included due to size constraints.
All model architectures, training logic, and evaluation code are provided.
Model and data can be shared separately for academic or demonstration purposes.


👤 **Author**
Bhaagwat Sharma
