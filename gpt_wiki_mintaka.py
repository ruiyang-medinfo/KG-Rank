from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from typing import List, Dict, Any
from context_generator import generate_context
from tqdm import tqdm
import pandas as pd

llm = ChatOpenAI(model_name="model_type",
                 temperature=0,
                 openai_api_key="api_key")

class ExtendedConversationBufferWindowMemory(ConversationBufferWindowMemory):
    extra_variables: List[str] = []

    @property
    def memory_variables(self) -> List[str]:
        """Will only return list of extra memory variables, not including self.memory_key."""
        return self.extra_variables

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return buffer with extra variables, but exclude the default 'history' memory_key."""
        d = super().load_memory_variables(inputs)
        d.pop("history", None)  
        d.update({k: inputs.get(k) for k in self.extra_variables})
        return d


memory = ExtendedConversationBufferWindowMemory(k=0,
                                                extra_variables=["context"])

template = """
Here are some examples for output format:

Question: What is the seventh tallest mountain in North America?
Example Output:
Mount Lucania

Question: What year was the first book of the A Song of Ice and Fire series published?
Example Output:
1996

Question: How old was Taylor Swift when she won her first Grammy?
Example Output:
20

Question: Has there ever been a Christian U.S. senator?
Example Output:
Yes

The following content may help you answer your question:

Context:
{context}

Question: {input}
Answer:
"""

PROMPT = PromptTemplate(
    input_variables=["context", "input"], template=template
)

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=PROMPT,
    verbose=True,
)

cohere_api_key = "api_key"

data = pd.read_csv("path")
questions = data["Question"].tolist() 

results = []
for i in tqdm(questions, desc="Processing Questions"):
    context = generate_context(i, cohere_api_key)
    answer = conversation.predict(context=context, input=i)
    results.append(answer)

