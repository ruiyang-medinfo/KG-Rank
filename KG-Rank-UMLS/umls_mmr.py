import torch
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

    def batch_encode(self, texts, batch_size=16):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
            all_embeddings.extend(batch_embeddings)
        return np.array(all_embeddings)

def get_similarity(query_vec, rel_vec):
    return cosine_similarity(query_vec, rel_vec)

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

def most_similar_relation(query_embedding, relation_embeddings):
    similarities = [get_similarity([query_embedding], [rel_emb]) for rel_emb in relation_embeddings]
    index = similarities.index(max(similarities))
    return index

def calculate_rerank_scores(query_embedding, relation_embeddings, rerank_relations_indices, base_weight=0.1, delta_weight=0.01):
    scores = []
    for i, rel_emb in enumerate(relation_embeddings):
        if i not in rerank_relations_indices:
            query_similarity = get_similarity([query_embedding], [rel_emb])
            
            if rerank_relations_indices:
                rel_similarities = [get_similarity([relation_embeddings[j]], [rel_emb]) for j in rerank_relations_indices]
                avg_rel_similarity = sum(rel_similarities) / len(rel_similarities)
            else:
                avg_rel_similarity = 0
            
            weight_factor = base_weight + delta_weight * len(rerank_relations_indices)
            
            score = query_similarity - weight_factor * avg_rel_similarity
            scores.append((i, score))
    return scores


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

        if relations:
            relation_texts = [query] + [f"{rel.get('relatedFromIdName', '')} {rel.get('additionalRelationLabel', '').replace('_', ' ')} {rel.get('relatedIdName', '')}" for rel in relations]
            embeddings = umlsbert.batch_encode(relation_texts)

            query_embedding = embeddings[0]
            relation_embeddings = embeddings[1:]

            rerank_relations_indices = [most_similar_relation(query_embedding, relation_embeddings)]

            while len(rerank_relations_indices) < 20 and len(relation_embeddings) > len(rerank_relations_indices):
                rerank_scores = calculate_rerank_scores(query_embedding, relation_embeddings, rerank_relations_indices)
                rerank_scores.sort(key=lambda x: x[1], reverse=True)

                for score in rerank_scores:
                    if score[0] not in rerank_relations_indices:
                        rerank_relations_indices.append(score[0])
                        break

            rerank_relations = [relations[i] for i in rerank_relations_indices]

            for rel in rerank_relations:
                related_from_id_name = rel.get("relatedFromIdName")
                additional_relation_label = rel.get("additionalRelationLabel").replace("_", " ")
                related_id_name = rel.get("relatedIdName")

                rels.append((related_from_id_name, additional_relation_label, related_id_name))

        umls_res[cui] = {"name": name, "definition": defi, "rels": rels}

    context = ""
    for k, v in umls_res.items():
      name = v["name"]
      definition = v["definition"]
      rels = v["rels"]
      rels_text = ""
      for rel in rels:
          rel_0 = rel[0] if rel[0] is not None else ""
          rel_1 = rel[1] if rel[1] is not None else ""
          rel_2 = rel[2] if rel[2] is not None else ""
          rels_text += "(" + rel_0 + "," + rel_1 + "," + rel_2 + ")\n"
      text = f"Name: {name}\nDefinition: {definition}\n"
      if rels_text != "":
          text += f"Relations: \n{rels_text}"

      context += text + "\n"
    if context != "":
      context = context[:-1]
    return context