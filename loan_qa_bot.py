import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict
import streamlit as st
from transformers import pipeline

class LoanQABot:
    def __init__(self, data_path: str):
        """Initialize the Loan QA Bot with data and embeddings."""
        self.df = pd.read_csv(data_path)
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.setup_embeddings()
        
        # Initialize a more powerful model for text generation
        self.qa_model = pipeline(
            "text2text-generation",
            model="google/flan-t5-large",  # Using a larger model
            max_length=256  # Allowing longer responses
        )

    def setup_embeddings(self):
        """Create and store embeddings for the dataset."""
        # Create detailed text representations of each row
        self.texts = []
        for _, row in self.df.iterrows():
            text = (
                f"Loan application details: The applicant has an income of {row['ApplicantIncome']}, "
                f"with additional co-applicant income of {row['CoapplicantIncome']}. "
                f"They requested a loan amount of {row['LoanAmount']} thousand units "
                f"for a term of {row['Loan_Amount_Term']} months. "
                f"The applicant's credit history score is {row['Credit_History']}, "
                f"they are {row['Education']} educated, "
                f"{'self-employed' if row['Self_Employed']=='Yes' else 'not self-employed'}, "
                f"and the property is located in a {row['Property_Area']} area. "
                f"The loan was {'approved' if row['Loan_Status']=='Y' else 'not approved'}."
            )
            self.texts.append(text)
        
        # Create embeddings
        embeddings = self.model.encode(self.texts)
        
        # Normalize embeddings and create FAISS index
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype('float32'))

    def get_relevant_context(self, query: str, k: int = 5) -> List[str]:
        """Retrieve relevant context based on the query."""
        query_vector = self.model.encode([query])
        distances, indices = self.index.search(query_vector.astype('float32'), k)
        return [self.texts[i] for i in indices[0]]

    def analyze_data(self, query: str) -> Dict:
        """Perform comprehensive data analysis based on query topics."""
        analysis = {}
        
        # Approval Rate Analysis
        approved = self.df[self.df['Loan_Status'] == 'Y']
        total = len(self.df)
        analysis['overall_approval_rate'] = (len(approved) / total) * 100

        # Credit History Impact
        good_credit = self.df[self.df['Credit_History'] == 1.0]
        good_credit_approval = (good_credit['Loan_Status'] == 'Y').mean() * 100
        bad_credit_approval = (self.df[self.df['Credit_History'] != 1.0]['Loan_Status'] == 'Y').mean() * 100
        analysis['credit_impact'] = {
            'good_credit_approval': good_credit_approval,
            'bad_credit_approval': bad_credit_approval
        }

        # Education Impact
        grad_approval = (self.df[self.df['Education'] == 'Graduate']['Loan_Status'] == 'Y').mean() * 100
        non_grad_approval = (self.df[self.df['Education'] == 'Not Graduate']['Loan_Status'] == 'Y').mean() * 100
        analysis['education_impact'] = {
            'graduate_approval': grad_approval,
            'non_graduate_approval': non_grad_approval
        }

        # Property Area Impact
        for area in ['Urban', 'Rural', 'Semiurban']:
            rate = (self.df[self.df['Property_Area'] == area]['Loan_Status'] == 'Y').mean() * 100
            analysis.setdefault('property_impact', {})[area] = rate

        # Loan Amount Analysis
        analysis['loan_amount'] = {
            'mean': self.df['LoanAmount'].mean(),
            'median': self.df['LoanAmount'].median(),
            'by_area': self.df.groupby('Property_Area')['LoanAmount'].mean().to_dict()
        }

        # Income Analysis
        analysis['income'] = {
            'mean_applicant': self.df['ApplicantIncome'].mean(),
            'mean_coapplicant': self.df['CoapplicantIncome'].mean(),
            'approval_high_income': (self.df[self.df['ApplicantIncome'] > self.df['ApplicantIncome'].median()]['Loan_Status'] == 'Y').mean() * 100
        }

        return analysis

    def generate_response(self, query: str) -> str:
        """Generate a comprehensive response using RAG approach and data analysis."""
        # Get relevant context
        contexts = self.get_relevant_context(query)
        analysis = self.analyze_data(query)
        
        # Create a detailed prompt
        prompt = (
            "You are a loan expert assistant. Answer the following question about loan applications "
            "using both the provided examples and statistical analysis. "
            "Give a detailed, informative response.\n\n"
            "Statistical Analysis:\n"
            f"- Overall loan approval rate: {analysis['overall_approval_rate']:.1f}%\n"
            f"- Credit history impact: {analysis['credit_impact']['good_credit_approval']:.1f}% approval with good credit vs "
            f"{analysis['credit_impact']['bad_credit_approval']:.1f}% with bad credit\n"
            f"- Education impact: {analysis['education_impact']['graduate_approval']:.1f}% for graduates vs "
            f"{analysis['education_impact']['non_graduate_approval']:.1f}% for non-graduates\n"
            "- Property area approval rates: " + 
            ", ".join([f"{area}: {rate:.1f}%" for area, rate in analysis['property_impact'].items()]) + "\n"
            f"- Average loan amount: {analysis['loan_amount']['mean']:.1f}k\n\n"
            "Example Cases:\n"
        )
        
        for i, ctx in enumerate(contexts, 1):
            prompt += f"Case {i}: {ctx}\n"
        
        prompt += f"\nQuestion: {query}\nDetailed Answer:"

        # Generate initial response
        response = self.qa_model(prompt)[0]['generated_text']
        
        # If response is too short or just repeats the question, enhance it with specific analysis
        if len(response.strip()) < 50 or query.lower() in response.lower():
            response = self.enhance_response(query, analysis)
        
        return response

    def enhance_response(self, query: str, analysis: Dict) -> str:
        """Enhance response with specific analysis based on query topic."""
        query_lower = query.lower()
        response_parts = []

        if any(word in query_lower for word in ['approval', 'approved', 'chances', 'likely']):
            response_parts.append(
                f"Based on our analysis, the overall loan approval rate is {analysis['overall_approval_rate']:.1f}%. "
                f"Having good credit history significantly improves chances, with {analysis['credit_impact']['good_credit_approval']:.1f}% "
                "approval rate compared to only "
                f"{analysis['credit_impact']['bad_credit_approval']:.1f}% for those with poor credit history."
            )

        if any(word in query_lower for word in ['education', 'graduate', 'study']):
            response_parts.append(
                f"Education plays a significant role in loan approval. Graduates have a {analysis['education_impact']['graduate_approval']:.1f}% "
                f"approval rate, while non-graduates have a {analysis['education_impact']['non_graduate_approval']:.1f}% approval rate. "
                "This suggests that higher education positively influences loan approval chances."
            )

        if any(word in query_lower for word in ['property', 'area', 'location', 'urban', 'rural']):
            area_rates = analysis['property_impact']
            best_area = max(area_rates.items(), key=lambda x: x[1])[0]
            response_parts.append(
                f"Property location significantly affects loan approval. {best_area} areas have the highest approval rate at "
                f"{area_rates[best_area]:.1f}%. The approval rates are: " + 
                ", ".join([f"{area}: {rate:.1f}%" for area, rate in area_rates.items()])
            )

        if any(word in query_lower for word in ['amount', 'money', 'loan size']):
            response_parts.append(
                f"The average loan amount is {analysis['loan_amount']['mean']:.1f}k units, with variations by area: " +
                ", ".join([f"{area}: {amount:.1f}k" for area, amount in analysis['loan_amount']['by_area'].items()])
            )

        if any(word in query_lower for word in ['income', 'salary', 'earn']):
            response_parts.append(
                f"The average applicant income is {analysis['income']['mean_applicant']:.1f} units, with an additional average "
                f"co-applicant income of {analysis['income']['mean_coapplicant']:.1f} units. Applicants with above-median income "
                f"have an approval rate of {analysis['income']['approval_high_income']:.1f}%."
            )

        if not response_parts:
            response_parts.append(
                f"Based on our analysis of loan applications, the overall approval rate is {analysis['overall_approval_rate']:.1f}%. "
                "Key factors affecting approval include credit history, education, property location, and income level. "
                "Would you like to know more about any specific factor?"
            )

        return " ".join(response_parts)

def main():
    st.title("Loan Application Q&A Bot")
    st.write("Ask questions about loan applications and get AI-powered answers!")

    # Initialize bot
    if 'qa_bot' not in st.session_state:
        st.session_state.qa_bot = LoanQABot('Training Dataset.csv')

    # User input
    user_question = st.text_input("Ask your question:", "")

    if st.button("Get Answer"):
        if user_question:
            with st.spinner("Analyzing data and generating response..."):
                response = st.session_state.qa_bot.generate_response(user_question)
                st.write("Answer:", response)

    # Example questions
    st.sidebar.header("Example Questions")
    example_questions = [
        "What factors have the biggest impact on loan approval?",
        "How does education level and credit history affect loan approval chances?",
        "What are the typical loan amounts for different property areas?",
        "How does income level influence loan approval?",
        "Which property areas have the highest loan approval rates?"
    ]
    for q in example_questions:
        if st.sidebar.button(q):
            with st.spinner("Analyzing data and generating response..."):
                response = st.session_state.qa_bot.generate_response(q)
                st.write("Answer:", response)

if __name__ == "__main__":
    main()