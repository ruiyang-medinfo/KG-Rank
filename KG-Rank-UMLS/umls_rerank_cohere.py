import torch
import cohere
import numpy as np
import re
import json
import requests
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

class UMLSBERT:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("GanjinZero/UMLSBert_ENG")
        self.model = AutoModel.from_pretrained("GanjinZero/UMLSBert_ENG")

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def batch_encode(self, texts, batch_size=16):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                model_output = self.model(**inputs)
                attention_mask = inputs["attention_mask"]
                batch_embeddings = self.mean_pooling(model_output, attention_mask)
            all_embeddings.extend(batch_embeddings)  
        return np.array(all_embeddings)

def get_similarity(query_vec, rel_vec):
    return cosine_similarity(query_vec, rel_vec)

class UMLS_CohereReranker:
    def __init__(self, api_key):
        self.co = cohere.Client(api_key)
    
    def rerank(self, query, rels, top_n=20):
        results = self.co.rerank(query=query, documents=rels, top_n=top_n, model="rerank-english-v2.0")
        reranked_results = []
        for r in results:
            reranked_results.append({
                "rels": r.document["text"],  
                "relevance_score": r.relevance_score
            })
        return reranked_results

class UMLS_API:
    def __init__(self, apikey, version="current"):
        self.apikey = apikey
        self.version = version
        self.search_url = f"https://uts-ws.nlm.nih.gov/rest/search/{version}"
        self.content_url = f"https://uts-ws.nlm.nih.gov/rest/content/{version}"
        self.content_suffix = "/CUI/{}/{}?apiKey={}"

    def search_cui(self, query):
        cui_results = []

        try:
            page = 1
            size = 1
            query = {"string": query, "apiKey": self.apikey, "pageNumber": page, "pageSize": size}
            r = requests.get(self.search_url, params=query)
            r.raise_for_status()
            print(r.url)
            r.encoding = 'utf-8'
            outputs = r.json()

            items = outputs["result"]["results"]

            if len(items) == 0:
                print("No results found.\n")

            for result in items:
                cui_results.append((result["ui"], result["name"]))

        except Exception as except_error:
            print(except_error)

        return cui_results

    def get_definitions(self, cui):
        try:
            suffix = self.content_suffix.format(cui, "definitions", self.apikey)
            r = requests.get(self.content_url + suffix)
            r.raise_for_status()
            r.encoding = "utf-8"
            outputs = r.json()

            return outputs["result"]
        except Exception as except_error:
            print(except_error)

    def get_relations(self, cui, pages=20):
        all_relations = []

        try:
            for page in range(1, pages + 1):
                suffix = self.content_suffix.format(cui, "relations", self.apikey) + f"&pageNumber={page}"
                r = requests.get(self.content_url + suffix)
                r.raise_for_status()
                r.encoding = "utf-8"
                outputs = r.json()

                page_relations = outputs.get("result", [])
                all_relations.extend(page_relations)

        except Exception as except_error:
            print(except_error)

        return all_relations

umls_api = UMLS_API("api_key")
umlsbert = UMLSBERT()
cohere_reranker = UMLS_CohereReranker("api_key")

def get_umls_keys(query, prompt, llm):
    umls_res = {}
    prompt = prompt.replace("{question}", query)

    try:
        keys_text = llm.predict(prompt)
        print(keys_text)
        pattern = r"\{(.*?)\}"
        matches = re.findall(pattern, keys_text.replace("\n", ""))
        if not matches:
            raise ValueError("No medical terminologies returned by the model.")
        
        keys_dict = json.loads("{" + matches[0] + "}")
        if "medical terminologies" not in keys_dict or not keys_dict["medical terminologies"]:
            raise ValueError("Model did not return expected 'medical terminologies' key.")
    except Exception as e:
        print(f"Error during model processing: {e}")
        return "" 

    for key in keys_dict["medical terminologies"][:]:
        cuis = umls_api.search_cui(key)

        if len(cuis) == 0:
            continue
        cui = cuis[0][0]
        name = cuis[0][1]

        defi = ""
        definitions = umls_api.get_definitions(cui)

        if definitions is not None:
            msh_def = None
            nci_def = None
            icf_def = None
            csp_def = None
            hpo_def = None

            for definition in definitions:
                source = definition["rootSource"]
                if source == "MSH":
                    msh_def = definition["value"]
                    break
                elif source == "NCI":
                    nci_def = definition["value"]
                elif source == "ICF":
                    icf_def = definition["value"]
                elif source == "CSP":
                    csp_def = definition["value"]
                elif source == "HPO":
                    hpo_def = definition["value"]

            defi = msh_def or nci_def or icf_def or csp_def or hpo_def

        relations = umls_api.get_relations(cui)
        rels=[]

        if relations is not None:
            relation_texts = [query] + [f"{rel.get('relatedFromIdName', '')} {rel.get('additionalRelationLabel', '').replace('_', ' ')} {rel.get('relatedIdName', '')}" for rel in relations]
            embeddings = umlsbert.batch_encode(relation_texts)

            query_embedding = embeddings[0]
            relation_embeddings = embeddings[1:]
            relation_scores = [(get_similarity([query_embedding], [rel_emb]), rel) for rel_emb, rel in zip(relation_embeddings, relations)]
            relation_scores.sort(key=lambda x: x[0],reverse=True)
            top_rels = relation_scores[:200]

            top_rel_texts = [f"{rel[1].get('relatedFromIdName', '')} {rel[1].get('additionalRelationLabel', '').replace('_', ' ')} {rel[1].get('relatedIdName', '')}" for rel in top_rels]
            if top_rel_texts:
                reranked_rel_texts = cohere_reranker.rerank(query, top_rel_texts)
            else:
                reranked_rel_texts = [] 

            for result in reranked_rel_texts:
                rel = result["rels"]  
                score = result["relevance_score"]
                rels.append((rel, score))

        umls_res[cui] = {"name": name, "definition": defi, "rels": rels}

    context = "" 
    for k, v in umls_res.items():
      name = v["name"]
      definition = v["definition"]
      rels = v["rels"]
      rels_text = ""
      for rel in rels:
          relation_description, relevance_score = rel
          rels_text += f"({relation_description})\n"
      text = f"Name: {name}\nDefinition: {definition}\n"
      if rels_text != "":
          text += f"Relations: \n{rels_text}"
          
      context += text + "\n"
    if context != "":
      context = context[:-1]
    return context