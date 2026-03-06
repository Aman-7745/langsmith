from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

os.environ['LANGSMITH_PROJECT'] = 'Sequential LLM app'

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a concise report (max 250 words) on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate exactly 5 bullet points from the following text:\n{text}',
    input_variables=['text']
)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, timeout=45, max_retries=1)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

config = {
    'run_name' :'sequential chain',
    'tags' : ['llm app','report generation', 'summarization'],
    'metadata' : {'model' : 'gpt-4o-mini', 'temp': 0.3, 'parser' : 'stroutputparser' }
}

result = chain.invoke({'topic': 'Unemployment in India'}, config=config)

print(result)
