# RAG_QnA_PDF_Chat_Model
This Gen_ai Retrival Augmented Generation(RAG) based application takes personal PDF as user input.
It divides the whole text document in to smaller chunks then vector embedding is computed with the help of "COHERE API" model="embed-model-V3.0" and it is stored in a langchain Vector store database FAISS.
When the user asks query the the query is transformed in to a numeric form and after semantic search the answer is found from that vector db using llm model"GEMINI-1.5-latest" of google response is generated.
This task is performed free of cost unlike OpenAI embed models here you do not have to pay for tokens,it is completely FREE.
![rag_indexing-8160f90a90a33253d0154659cf7d453f](https://github.com/DUSMANTA03/RAG_Q-A_PDF_Chat_Model/assets/112675892/07d1e06a-ba6e-43d5-a014-ee8b96a174fc)
