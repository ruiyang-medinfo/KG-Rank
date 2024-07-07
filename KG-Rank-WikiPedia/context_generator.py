import requests
from bs4 import BeautifulSoup
import cohere

class DBpediaAnnotator:
    def __init__(self, max_entities=5, confidence=0.8):
        self.base_url = "https://api.dbpedia-spotlight.org/en/annotate"
        self.headers = {"accept": "application/json"}
        self.max_entities = max_entities
        self.confidence = confidence

    def annotate_text(self, text):
        params = {"text": text, "confidence": self.confidence}
        response = requests.get(self.base_url, headers=self.headers, params=params)
        if response.status_code == 200:
            data = response.json()
            if "Resources" in data:
                data["Resources"] = data["Resources"][:self.max_entities]
            return data
        else:
            return "Error: " + str(response.status_code)
    

    def extract_entities(self, query):
        annotations = self.annotate_text(query)
        entities = [] 
        if "Resources" in annotations:
            for resource in annotations["Resources"]:
                entities.append({
                "uri": resource["@URI"],
                "name": resource["@surfaceForm"]
            })
                
        return entities

class HTMLExtractor:
    @staticmethod
    def extract_entity_information(entity_uri):
        response = requests.get(entity_uri)
        if response.status_code == 200:
            return response.text
        else:
            return None

    @staticmethod
    def html_extraction(entity_info):
        soup = BeautifulSoup(entity_info, "html.parser")
        intro_text = soup.find("p").get_text()
        properties = {}
        table = soup.find("table")
        if table:
            rows = table.find_all("tr")
            for row in rows:
                cols = row.find_all("td")
                if len(cols) >= 2:
                    prop = cols[0].get_text().strip()
                    value = cols[1].get_text().strip()
                    if not HTMLExtractor.contains_url(value): 
                        properties[prop] = value
        return intro_text, properties

    @staticmethod
    def contains_url(value):
        return "http" in value or "www" in value or "dbr:" in value or "dbpedia.org" in value

class CohereReranker:
    def __init__(self, api_key):
        self.co = cohere.Client(api_key)

    def rerank(self, query, properties, top_n=10):
        results = self.co.rerank(query=query, documents=properties, top_n=top_n, model="rerank-english-v2.0")
        reranked_results = []
        for r in results:
            reranked_results.append({
                "properties": r.document["text"], 
                "relevance_score": r.relevance_score
            })
        return reranked_results

def create_context(intro, entities, reranked_results):
    context = "Entity:\n"
    for entity in entities:
        context += f"- {entity['name']}\n"
    context += "\nIntroduction:\n"
    context += f"{intro}\n\n"
    
    text = ""
    for i in reranked_results:
        property = i["properties"]
        #score = i["relevance_score"]
        text += f"- {property}\n"

    if text != "":
        context += f"Properties:\n{text}"

    return context.strip() 

def generate_context(query, cohere_api_key):
    dbpedia_annotator = DBpediaAnnotator()
    cohere_reranker = CohereReranker(cohere_api_key)
    entities = dbpedia_annotator.extract_entities(query)
    if not entities:  
        return ""  
    context = ""
    for entity in entities:
        entity_info = HTMLExtractor.extract_entity_information(entity['uri'])
        if entity_info:
            intro, properties = HTMLExtractor.html_extraction(entity_info) 
            if properties:
                properties_strings = [f"{key}: {value}" for key, value in properties.items() if not HTMLExtractor.contains_url(value)]
                reranked_results = cohere_reranker.rerank(query, properties_strings)
                entity_context = create_context(intro, [entity], reranked_results)
                context += entity_context + "\n\n"
            else:
                context += f"Entity: {entity['name']}"
    return context.strip()

