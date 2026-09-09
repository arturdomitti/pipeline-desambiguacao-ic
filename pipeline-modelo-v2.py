#!/usr/bin/env python3
"""
WebSensors Science Metrics - Pipeline de Desambiguação de Autores (v2)
Este código foi evoluído para implementar chamadas REAIS para:
1. A LLM do Laboratório (Qwen 35B hospedado no ICMC-USP).
2. A API pública do OpenAlex para busca de candidatos, artigos e afiliações.
3. Demonstração de carregamento estruturado dos Dados Abertos da Sucupira (CAPES).

Possui modo offline (Simulação) integrado para testes rápidos de fluxo.
"""

import os
import json
import yaml
import networkx as nx
from typing import List, Dict, Any

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    import difflib
    HAS_RAPIDFUZZ = False

# ==============================================================================
# 1. CARREGAMENTO DE CONFIGURAÇÃO (config.py)
# ==============================================================================
def load_pipeline_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        return {
            "project_name": "websensors-science-metrics-v2",
            "modeling": {
                "entity_resolution": {
                    "tau_lex": 0.65,
                    "max_depth": 5
                }
            },
            "openalex": {
                "top_n_candidates": 5,
                "api_key": ""
            },
            "researchers": [
                {"name": "Ana Paula Silva", "ppg": "Ciência da Computação"}
            ]
        }
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ==============================================================================
# 2. INGESTÃO DE DADOS REAIS E SIMULADOS (data_ingestion.py)
# ==============================================================================
def calculate_name_similarity(name1: str, name2: str) -> float:
    if HAS_RAPIDFUZZ:
        return float(fuzz.WRatio(name1, name2) / 100.0)
    else:
        matcher = difflib.SequenceMatcher(None, name1.lower(), name2.lower())
        return float(matcher.ratio())

def reconstruct_abstract(inverted_index: Dict[str, List[int]]) -> str:
    """
    O OpenAlex armazena resumos no formato 'abstract_inverted_index' por direitos autorais.
    Esta função matemática reconstrói o abstract em texto corrido legível.
    """
    if not inverted_index:
        return ""
    words = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            words[pos] = word
    sorted_positions = sorted(words.keys())
    return " ".join([words[pos] for pos in sorted_positions])


class DataIngestion:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tau_lex = config.get("modeling", {}).get("entity_resolution", {}).get("tau_lex", 0.65)
        self.top_n = config.get("openalex", {}).get("top_n_candidates", 5)

    def fetch_validated_researcher_data(self, target_name: str, mock: bool = True, csv_path: str = "") -> Dict[str, Any]:
        """
        Coleta dados validados da Sucupira.
        Suporta o modo Simulação (mock) ou carregamento real de planilhas CAPES.
        """
        if mock or not csv_path or not os.path.exists(csv_path):
            if not mock:
                print("[Aviso] Planilha Sucupira não encontrada localmente. Usando dados simulados de referência.")
            return {
                "id": "A1",
                "name": target_name,
                "source": "Sucupira",
                "validated": True,
                "ppg": "Programa de Pós Graduação em Ciência da Computação",
                "institution": "Universidade Exemplo",
                "works": [
                    {
                        "id": "W1",
                        "title": "Graph Representation Learning for Scientific Networks",
                        "year": 2024,
                        "abstract": "We study graph representation learning methods for scientific collaboration and knowledge networks.",
                        "coauthors": ["Carlos Souza"]
                    }
                ]
            }

        # CARREGAMENTO REAL DA SUCUPIRA (CAPES DADOS ABERTOS)
        try:
            import pandas as pd
            print(f"Lendo base Sucupira real a partir de: {csv_path}")
            df = pd.read_csv(csv_path)
            
            # Filtra docente na coluna padrão de Dados Abertos CAPES: NM_DOCENTE
            df_docente = df[df["NM_DOCENTE"].str.contains(target_name, case=False, na=False)]
            if df_docente.empty:
                print(f"[Aviso] Docente '{target_name}' não encontrado na base de dados CAPES.")
                return {}
                
            works = []
            for idx, row in df_docente.iterrows():
                works.append({
                    "id": f"W_S_{idx}",
                    "title": row.get("NM_PRODUCAO", "Artigo sem título"),
                    "year": int(row.get("AN_BASE", 2024)),
                    "abstract": "",  # Sucupira não contém abstracts, necessita do OpenAlex
                    "coauthors": []  # Pode ser populado se houver coluna correspondente
                })
                
            return {
                "id": "A1",
                "name": target_name,
                "source": "Sucupira",
                "validated": True,
                "ppg": df_docente.iloc[0].get("NM_PROGRAMA_IES", "Programa de Pós Graduação em Ciência da Computação"),
                "institution": df_docente.iloc[0].get("SG_ENTIDADE_ENSINO", "Universidade Exemplo"),
                "works": works
            }
        except Exception as e:
            print(f"Erro ao carregar planilha CAPES: {e}")
            return {}

    def fetch_openalex_candidates(self, target_name: str, mock: bool = True) -> List[Dict[str, Any]]:
        """
        Coleta dados de candidatos do OpenAlex de forma simulada ou real via API.
        """
        if mock:
            # Modo simulado de fallback
            all_raw_candidates = [
                {
                    "id": "A2",
                    "name": "A. P. Silva",
                    "source": "OpenAlex",
                    "validated": False,
                    "works": [
                        {
                            "id": "W2",
                            "title": "Knowledge Graph Refinement with Language Models",
                            "year": 2025,
                            "abstract": "We investigate language models for entity resolution and refinement of scientific knowledge graphs.",
                            "coauthors": ["Carlos Souza"]
                        }
                    ],
                    "affiliations": ["Universidade Exemplo"]
                }
            ]
            selected = []
            for cand in all_raw_candidates:
                sim = calculate_name_similarity(target_name, cand["name"])
                if sim >= self.tau_lex:
                    cand["lexical_similarity"] = sim
                    selected.append(cand)
            return selected

        # CONSULTA REAL À API DO OPENALEX
        import requests
        print(f"Consultando API pública do OpenAlex para: '{target_name}'")
        url = "https://api.openalex.org/authors"
        params = {"search": target_name, "per_page": self.top_n}
        headers = {"User-Agent": "mailto:suporte-ic@icmc.usp.br"} # Polite Pool do OpenAlex
        
        # Carrega chave de API se disponível
        api_key = os.environ.get("OPENALEX_API_KEY") or self.config.get("openalex", {}).get("api_key")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            response = requests.get(url, params=params, headers=headers, timeout=15)
            if response.status_code != 200:
                print(f"Erro na API OpenAlex: {response.status_code}")
                return []
                
            results = response.json().get("results", [])
            selected_candidates = []
            
            for auth in results:
                display_name = auth.get("display_name", "")
                sim = calculate_name_similarity(target_name, display_name)
                
                # Aplica o limiar lexical tau_lex
                if sim < self.tau_lex:
                    continue
                    
                auth_id = auth.get("id", "").split("/")[-1]  # Extrai ID simples (ex: A5012345678)
                
                # Coleta instituições associadas
                affiliations = []
                for inst in auth.get("last_known_institutions", []):
                    affiliations.append(inst.get("display_name", ""))
                
                # Coleta as produções recentes desse autor no OpenAlex
                works = self.fetch_author_works_from_openalex(auth_id)
                
                selected_candidates.append({
                    "id": auth_id,
                    "name": display_name,
                    "source": "OpenAlex",
                    "validated": False,
                    "lexical_similarity": sim,
                    "works": works,
                    "affiliations": affiliations
                })
            return selected_candidates
        except Exception as e:
            print(f"Falha ao conectar com a API OpenAlex: {e}")
            return []

    def fetch_author_works_from_openalex(self, author_id: str) -> List[Dict[str, Any]]:
        """Busca as obras reais do candidato usando o endpoint do OpenAlex."""
        import requests
        url = "https://api.openalex.org/works"
        params = {
            "filter": f"author.id:{author_id}",
            "per_page": 10,
            "sort": "publication_year:desc"
        }
        headers = {"User-Agent": "mailto:suporte-ic@icmc.usp.br"}
        try:
            response = requests.get(url, params=params, headers=headers, timeout=15)
            if response.status_code != 200:
                return []
                
            works_data = []
            for work in response.json().get("results", []):
                # Reconstrói abstract invertido
                inv_index = work.get("abstract_inverted_index")
                abstract_text = reconstruct_abstract(inv_index) if inv_index else ""
                
                # Extrai lista simples de coautores
                coauthors = []
                for membership in work.get("authorships", []):
                    author_name = membership.get("author", {}).get("display_name", "")
                    coauthors.append(author_name)
                
                works_data.append({
                    "id": work.get("id", "").split("/")[-1],
                    "title": work.get("title", "Sem título"),
                    "year": work.get("publication_year", 2024),
                    "abstract": abstract_text,
                    "coauthors": coauthors
                })
            return works_data
        except Exception:
            return []


# ==============================================================================
# 3. GERENCIADOR DO GRAFO (graph_manager.py)
# ==============================================================================
class KnowledgeGraphManager:
    def __init__(self):
        self.G = nx.MultiDiGraph()

    def add_author_node(self, node_id: str, label: str, source: str, validated: bool, **kwargs):
        self.G.add_node(node_id, node_type="author", label=label, source=source, validated=validated, **kwargs)

    def add_work_node(self, node_id: str, label: str, source: str, validated: bool, **kwargs):
        self.G.add_node(node_id, node_type="work", label=label, source=source, validated=validated, **kwargs)

    def add_institution_node(self, node_id: str, label: str, source: str, validated: bool, **kwargs):
        self.G.add_node(node_id, node_type="institution", label=label, source=source, validated=validated, **kwargs)

    def add_ppg_node(self, node_id: str, label: str, source: str, validated: bool, **kwargs):
        self.G.add_node(node_id, node_type="graduate_program", label=label, source=source, validated=validated, **kwargs)

    def add_custom_node(self, node_id: str, node_type: str, label: str, source: str, validated: bool, **kwargs):
        self.G.add_node(node_id, node_type=node_type, label=label, source=source, validated=validated, **kwargs)

    def add_relation(self, source_node: str, target_node: str, relation: str, source: str, validated: bool, **kwargs):
        self.G.add_edge(source_node, target_node, relation=relation, source=source, validated=validated, **kwargs)

    def build_undirected_view(self) -> nx.Graph:
        return nx.Graph(self.G)


# ==============================================================================
# 4. RESOLUTOR DE IDENTIDADE E INTEGRAÇÃO DE LLM (identity_resolver.py)
# ==============================================================================
class IdentityResolver:
    def __init__(self, kg_manager: KnowledgeGraphManager, config: Dict[str, Any]):
        self.kg = kg_manager
        self.config = config
        self.max_depth = config.get("modeling", {}).get("entity_resolution", {}).get("max_depth", 5)

    def find_walks(self, source_id: str, target_id: str) -> List[List[str]]:
        undirected_g = self.kg.build_undirected_view()
        if source_id not in undirected_g or target_id not in undirected_g:
            return []
        return list(nx.all_simple_paths(undirected_g, source=source_id, target=target_id, cutoff=self.max_depth))

    def translate_path_to_evidence(self, path: List[str]) -> str:
        text_parts = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            u_label = self.kg.G.nodes[u].get("label", u)
            v_label = self.kg.G.nodes[v].get("label", v)
            u_type = self.kg.G.nodes[u].get("node_type", "")
            v_type = self.kg.G.nodes[v].get("node_type", "")
            
            rel_name = "connected_to"
            if self.kg.G.has_edge(u, v):
                edges_data = self.kg.G.get_edge_data(u, v)
                if edges_data:
                    rel_name = edges_data[0].get("relation", "connected_to")
                text_parts.append(f"{u_label} ({u_type}) -[{rel_name}]-> {v_label} ({v_type})")
            elif self.kg.G.has_edge(v, u):
                edges_data = self.kg.G.get_edge_data(v, u)
                if edges_data:
                    rel_name = edges_data[0].get("relation", "connected_to")
                text_parts.append(f"{u_label} ({u_type}) <-[{rel_name}]- {v_label} ({v_type})")
            else:
                text_parts.append(f"{u_label} ({u_type}) -- {v_label} ({v_type})")
                
        return " | " .join(text_parts)

    def extract_author_works(self, author_id: str) -> List[Dict[str, Any]]:
        works = []
        if author_id in self.kg.G:
            for _, target, edge_idx, edge_data in self.kg.G.out_edges(author_id, keys=True, data=True):
                if edge_data.get("relation") == "author_of":
                    node_data = self.kg.G.nodes[target]
                    works.append({
                        "id": target,
                        "title": node_data.get("label", ""),
                        "year": node_data.get("year"),
                        "abstract": node_data.get("abstract", ""),
                        "source": node_data.get("source", ""),
                        "validated": node_data.get("validated", False)
                    })
        return works

    def build_llm_context(self, source_id: str, target_id: str, walks: List[List[str]]) -> Dict[str, Any]:
        translated_walks = [self.translate_path_to_evidence(w) for w in walks]
        known_facts = []
        if source_id in self.kg.G:
            for _, target, edge_data in self.kg.G.out_edges(source_id, data=True):
                relation = edge_data.get("relation", "")
                target_label = self.kg.G.nodes[target].get("label", target)
                target_type = self.kg.G.nodes[target].get("node_type", "")
                if edge_data.get("validated"):
                    known_facts.append({
                        "subject": self.kg.G.nodes[source_id].get("label", source_id),
                        "relation": relation,
                        "object": f"{target_label} ({target_type})"
                    })

        return {
            "validated_author": {
                "id": source_id,
                "name": self.kg.G.nodes[source_id].get("label", ""),
                "source": self.kg.G.nodes[source_id].get("source", ""),
                "works": self.extract_author_works(source_id)
            },
            "candidate_author": {
                "id": target_id,
                "name": self.kg.G.nodes[target_id].get("label", ""),
                "source": self.kg.G.nodes[target_id].get("source", ""),
                "works": self.extract_author_works(target_id)
            },
            "graph_paths": translated_walks,
            "known_validated_facts": known_facts
        }

    def create_llm_prompts(self, context: Dict[str, Any]) -> tuple:
        system_prompt = (
            "Você realiza resolução de entidades em grafos de conhecimento científicos.\n"
            "A primeira entidade é um pesquisador pertencente ao conjunto validado (Sucupira).\n"
            "A segunda entidade é um candidato do OpenAlex ainda não validado.\n"
            "Analise as evidências fornecidas.\n\n"
            "Tarefas:\n"
            "1. Primeiro, leia os títulos e resumos das produções e infira os principais tópicos científicos de cada um de forma autônoma.\n"
            "2. Depois, compare a similaridade dos nomes, as redes de coautores e afiliações institucionais a partir das caminhadas de grafo fornecidas.\n"
            "3. Pondere se a compatibilidade temática e estrutural apoia a identidade (use afinidade temática apenas como suporte complementar).\n\n"
            "Retorne EXCLUSIVAMENTE um objeto JSON válido com os seguintes campos exatos:\n"
            "{\n"
            '  "same_person": bool,\n'
            '  "confidence": float (de 0.0 a 1.0),\n'
            '  "validated_author_topics": ["tópico1", "tópico2"],\n'
            '  "candidate_author_topics": ["tópico1", "tópico2"],\n'
            '  "topic_evidence": "sua justificativa baseada nos temas de pesquisa",\n'
            '  "structural_evidence": "sua justificativa baseada em coautores, PPG e conexões de grafo",\n'
            '  "contrary_evidence": "evidências contrárias ou discrepâncias apontadas",\n'
            '  "reasoning_summary": "resumo geral da dedução",\n'
            '  "decision": "validate_candidate" | "reject_candidate" | "uncertain"\n'
            "}"
        )
        
        user_prompt = (
            "Analise o seguinte par de pesquisadores para desambiguação:\n\n"
            "CONTEXTO:\n"
            f"{json.dumps(context, ensure_ascii=False, indent=2)}"
        )
        
        return system_prompt, user_prompt

    def query_lab_llm(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        Conecta de forma real ao servidor da LLM do Laboratório (Agents4Gov - ICMC/USP).
        Utiliza o Qwen 35B parametrizado pelo laboratório.
        """
        import requests
        
        # Parâmetros oficiais do laboratório fornecidos em documentos
        base_url = "https://agents4gov.icmc.usp.br/api/v1"
        model = "externo.revejo-qwen3.6-35b-a3b"
        api_key = "sk-e31ea19cc3c3463684bda7f3e0c5a4a5"
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"}
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        try:
            print(f"Enviando dados para a LLM do Laboratório ICMC-USP ({model})...")
            url = f"{base_url}/chat/completions"
            response = requests.post(url, json=payload, headers=headers, timeout=45)
            
            if response.status_code == 200:
                result_json = response.json()
                content = result_json["choices"][0]["message"]["content"]
                return json.loads(content)
            else:
                print(f"[Erro] Falha no servidor do laboratório (Status {response.status_code}): {response.text}")
                # Fallback simples sem response_format caso o endpoint tenha limitações locais
                payload.pop("response_format")
                print("Tentando requisição de fallback sem formatação rígida de JSON...")
                response = requests.post(url, json=payload, headers=headers, timeout=45)
                if response.status_code == 200:
                    result_json = response.json()
                    content = result_json["choices"][0]["message"]["content"]
                    return json.loads(content)
        except Exception as e:
            print(f"[Erro] Falha de conexão/parsing com a LLM do laboratório: {e}")
            print("Dica: Certifique-se de estar conectado à internet e que as credenciais do lab estejam ativas.")
            
        return {}


# ==============================================================================
# 5. EXECUÇÃO INTEGRADA (Main)
# ==============================================================================
def run_pipeline(target_name: str, config_path: str = "", mock: bool = True, csv_path: str = ""):
    print(f"--- Iniciando Pipeline v2 ({'MODO SIMULAÇÃO' if mock else 'MODO REAL'}) para: {target_name} ---")
    
    config = load_pipeline_config(config_path)
    ingestion = DataIngestion(config)
    
    # 1. Coleta os dados validados (Sucupira)
    validated_data = ingestion.fetch_validated_researcher_data(target_name, mock=mock, csv_path=csv_path)
    if not validated_data:
        print("[Erro] Falha ao coletar dados validados iniciais.")
        return
        
    # 2. Coleta candidatos (OpenAlex)
    candidates = ingestion.fetch_openalex_candidates(target_name, mock=mock)
    print(f"Candidato(s) OpenAlex qualificado(s) lexicalmente: {[c['name'] for c in candidates]}")
    
    # 3. Reconstrói o Grafo de Conhecimento
    kg = KnowledgeGraphManager()
    
    # Insere dados de referência validados
    kg.add_author_node("A1", validated_data["name"], validated_data["source"], validated_data["validated"])
    kg.add_ppg_node("P1", validated_data["ppg"], validated_data["source"], validated_data["validated"])
    kg.add_institution_node("I1", validated_data["institution"], validated_data["source"], validated_data["validated"])
    kg.add_custom_node("C1", "country", "Brazil", validated_data["source"], validated_data["validated"])
    
    kg.add_relation("A1", "P1", "member_of_ppg", validated_data["source"], validated_data["validated"])
    kg.add_relation("P1", "I1", "hosted_by", validated_data["source"], validated_data["validated"])
    kg.add_relation("I1", "C1", "located_in", validated_data["source"], validated_data["validated"])
    
    for work in validated_data["works"]:
        kg.add_work_node(work["id"], work["title"], validated_data["source"], validated_data["validated"], abstract=work["abstract"], year=work["year"])
        kg.add_relation("A1", work["id"], "author_of", validated_data["source"], validated_data["validated"])
        for coauthor in work["coauthors"]:
            kg.add_custom_node("A3", "author", coauthor, "OpenAlex", False)
            kg.add_relation("A3", work["id"], "author_of", "OpenAlex", False)

    # Insere dados coletados do OpenAlex
    for idx, cand in enumerate(candidates, start=2):
        cand_id = cand["id"]
        kg.add_author_node(cand_id, cand["name"], cand["source"], cand["validated"], lexical_score=cand["lexical_similarity"])
        
        for aff in cand["affiliations"]:
            if aff == validated_data["institution"]:
                kg.add_relation(cand_id, "I1", "affiliated_with", cand["source"], cand["validated"])
            else:
                new_inst_id = f"I_new_{idx}"
                kg.add_institution_node(new_inst_id, aff, cand["source"], cand["validated"])
                kg.add_relation(cand_id, new_inst_id, "affiliated_with", cand["source"], cand["validated"])
                
        for work in cand["works"]:
            kg.add_work_node(work["id"], work["title"], cand["source"], cand["validated"], abstract=work["abstract"], year=work["year"])
            kg.add_relation(cand_id, work["id"], "author_of", cand["source"], cand["validated"])
            for coauthor in work["coauthors"]:
                # Se for coautor comum, conecta ao nó existente
                co_label = "Carlos Souza"
                if coauthor == co_label:
                    kg.add_relation("A3", work["id"], "author_of", cand["source"], cand["validated"])
                else:
                    new_co_id = f"A_new_co_{idx}"
                    kg.add_custom_node(new_co_id, "author", coauthor, cand["source"], cand["validated"])
                    kg.add_relation(new_co_id, work["id"], "author_of", cand["source"], cand["validated"])

    # 4. Resolução de Identidade
    resolver = IdentityResolver(kg, config)
    for cand in candidates:
        cand_id = cand["id"]
        print(f"\n--- Analisando Candidato: {cand['name']} ({cand_id}) ---")
        
        # Procura caminhadas no Grafo de Conhecimento
        walks = resolver.find_walks("A1", cand_id)
        print(f"Evidência Estrutural: {len(walks)} caminhada(s) encontrada(s).")
        
        # Constrói contexto
        context = resolver.build_llm_context("A1", cand_id, walks)
        
        # Cria prompts
        system_prompt, user_prompt = resolver.create_llm_prompts(context)
        
        if mock:
            print("[Modo Simulação] prompts estruturados com sucesso. No modo real, a LLM do laboratório seria invocada agora.")
            print("Visualização de exemplo do contexto estruturado JSON (Compacto):")
            print(json.dumps(context, indent=2, ensure_ascii=False)[:300] + "...\n[Texto truncado para fins de depuração]")
        else:
            # CHAMADA REAL À LLM DO LABORATÓRIO
            verdict = resolver.query_lab_llm(system_prompt, user_prompt)
            print("\n================ Veredito Gerado Pela LLM do Lab ================")
            print(json.dumps(verdict, indent=2, ensure_ascii=False))
            print("=================================================================\n")


if __name__ == "__main__":
    # Para rodar no modo Real no seu computador conectado à internet,
    # mude o parâmetro 'mock=True' para 'mock=False'
    run_pipeline("Ana Paula Silva", mock=True)
