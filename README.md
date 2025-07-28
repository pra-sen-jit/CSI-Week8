# Gen_AI_Celebal
=======
# Loan Application Q&A Chatbot

An intelligent chatbot that uses Retrieval Augmented Generation (RAG) to answer questions about loan applications. The system combines advanced natural language processing with statistical analysis to provide comprehensive insights about loan applications, approval factors, and trends.

## Features

- 🤖 Interactive Streamlit web interface
- 🔍 Semantic search using Sentence Transformers
- 📊 Comprehensive statistical analysis of loan data
- 💡 Intelligent response generation using FLAN-T5 Large
- 🚀 Real-time data processing and analysis
- 📈 Detailed insights about approval rates, credit impact, and more
- 💬 Example questions for quick testing

## Technical Stack

The application uses the following libraries with specific versions:

```
pandas==2.1.3
numpy==1.26.2
sentence-transformers==5.0.0
faiss-cpu==1.11.0
streamlit==1.46.1
transformers==4.53.2
torch==2.7.1
accelerate==1.8.1
scikit-learn==1.3.2
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
streamlit run loan_qa_bot.py
```

2. Open your web browser and navigate to:
   - Local URL: http://localhost:8501
   - The application will automatically open in your default browser

3. Ask questions about loan applications using the text input field

## Features in Detail

### 1. Data Analysis Capabilities
- Approval rate analysis by various factors
- Credit history impact assessment
- Education level influence
- Property area statistics
- Income level analysis
- Loan amount patterns

### 2. Question Types Supported
- Approval chances and factors
- Credit history impact
- Education level influence
- Property area preferences
- Loan amount trends
- Income requirements
- Combined factor analysis

### 3. Technical Implementation
- **Semantic Search**: Uses FAISS for efficient similarity search
- **Embeddings**: Sentence-BERT (all-MiniLM-L6-v2) for text embeddings
- **Text Generation**: FLAN-T5 Large for comprehensive responses
- **Data Processing**: Pandas and NumPy for statistical analysis
- **Web Interface**: Streamlit for interactive UI

## Example Questions

1. "What factors have the biggest impact on loan approval?"
2. "How does education level and credit history affect loan approval chances?"
3. "What are the typical loan amounts for different property areas?"
4. "How does income level influence loan approval?"
5. "Which property areas have the highest loan approval rates?"

## Data Processing

The system processes loan application data with the following fields:
- Applicant Income
- Co-applicant Income
- Loan Amount
- Loan Term
- Credit History
- Education Level
- Employment Status
- Property Area
- Loan Status

## Response Generation

The chatbot uses a sophisticated response generation system that:
1. Retrieves relevant examples using semantic search
2. Performs statistical analysis of the data
3. Combines examples and statistics in a detailed prompt
4. Generates responses using the FLAN-T5 Large model
5. Enhances responses with specific data analysis when needed

## Performance Notes

- The first run may take longer as it downloads the required models
- Response generation typically takes 2-5 seconds
- The system can handle complex, multi-factor questions
- Responses are based on both statistical analysis and similar examples

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
