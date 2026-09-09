import os
import json
import requests
import difflib
import pandas as pd
import networkx as nx
from pprint import pprint
from openai import OpenAI

# Calcula a similaridade lexical entre o nome do docente validado (Sucupira) e do candidato (OpenAlex) usando a razão da SequenceMatcher
def calculate_name_similarity(name1: str, name2: str) -> float:
    matcher = difflib.SequenceMatcher(None, name1.lower(), name2.lower())
    return float(matcher.ratio())

# Reconstitui o resumo em texto corrido a partir do 'abstract_inverted_index' fornecido pela API do OpenAlex devido a restrições de direitos autorais
def reconstruct_abstract(inverted_index: dict) -> str:
    if not inverted_index:
        return ""
    words = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            words[pos] = word
    sorted_positions = sorted(words.keys())
    return " ".join([words[pos] for pos in sorted_positions])

# Coleta até 5 publicações mais recentes de um determinado autor no OpenAlex, incluindo título, ano, resumo e coautores
def fetch_author_works_from_openalex(author_id: str, max_works: int = 5):
    url = "https://api.openalex.org/works"
    params = {
        "filter": f"author.id:{author_id}",
        "per_page": max_works,
        "sort": "publication_year:desc"
    }
    headers = {"User-Agent": "mailto:arturdomitti@usp.br"}

    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        if response.status_code != 200:
            return []
            
        works_data = []
        for work in response.json().get("results", []):
            inv_index = work.get("abstract_inverted_index")
            abstract_text = reconstruct_abstract(inv_index) if inv_index else ""
            
            coauthors = [
                m.get("author", {}).get("display_name", "")
                for m in work.get("authorships", [])
            ]
            
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

# Consulta o OpenAlex em busca de autores candidatos, filtrando apenas aqueles com similaridade de nome igual ou superior ao limiar tau_lex (0.65)
def fetch_openalex_candidates(target_name: str, tau_lex: float = 0.65, top_n: int = 5):
    print(f"\nConsultando API do OpenAlex para: '{target_name}'...")
    url = "https://api.openalex.org/authors"
    params = {"search": target_name, "per_page": top_n}
    headers = {"User-Agent": "mailto:arturdomitti@usp.br"}
    candidates = []

    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        if response.status_code != 200:
            print(f"Erro na API OpenAlex: Status {response.status_code}")
            return []

        data = response.json()
        if not isinstance(data, dict):
            return []

        raw_results = data.get("results")
        if not raw_results or not isinstance(raw_results, list):
            print("Nenhum resultado retornado pelo OpenAlex.")
            return []

        for auth in raw_results:
            if not isinstance(auth, dict):
                continue

            display_name = auth.get("display_name") or ""
            if not display_name:
                continue
                
            sim = calculate_name_similarity(target_name, display_name)
            if sim < tau_lex:
                continue

            raw_id = auth.get("id") or ""
            auth_id = raw_id.split("/")[-1] if raw_id else ""

            raw_insts = auth.get("last_known_institutions") or []
            affiliations = []
            if isinstance(raw_insts, list):
                for inst in raw_insts:
                    if isinstance(inst, dict) and inst.get("display_name"):
                        affiliations.append(inst.get("display_name"))

            works = fetch_author_works_from_openalex(auth_id) if auth_id else []

            candidates.append({
                "id": auth_id,
                "name": display_name,
                "source": "OpenAlex",
                "validated": False,
                "lexical_similarity": sim,
                "works": works,
                "affiliations": affiliations
            })
            
        return candidates
    except Exception as e:
        print(f"Erro ao processar dados do OpenAlex: {e}")
        return []

# Lê a base de Dados Abertos da Sucupira/CAPES (CSV) e aplica filtros combinados de Programa de Pós-Graduação e município do campus
def load_sucupira_docents(csv_path: str, ppg_filter: str = "COMPUTAÇÃO", city_filter: str = "SÃO CARLOS"):
    try:
        df = pd.read_csv(csv_path, sep=";", encoding="iso-8859-1", low_memory=False)
        
        if "NM_PROGRAMA_IES" in df.columns and ppg_filter:
            df = df[df["NM_PROGRAMA_IES"].str.contains(ppg_filter, case=False, na=False)]
            
        if "NM_MUNICIPIO_PROGRAMA_IES" in df.columns and city_filter:
            df = df[df["NM_MUNICIPIO_PROGRAMA_IES"].str.contains(city_filter, case=False, na=False)]
        elif "SG_ENTIDADE_ENSINO" in df.columns:
            df = df[df["SG_ENTIDADE_ENSINO"].isin(["USP", "UFSCAR"])]
            
        print(f"Total de registros encontrados em São Carlos para '{ppg_filter}': {len(df)}")
        return df
        
    except Exception as e:
        print(f"Erro ao ler CSV da Sucupira: {e}")
        return None

# Identifica o tipo de relação existente entre dois nós adjacentes no grafo direcionado (MultiDiGraph), checando ambos os sentidos da aresta
def relation_between(G, u, v):
    relations = []
    if G.has_edge(u, v):
        for _, attrs in G[u][v].items():
            relations.append(attrs["relation"])
    if G.has_edge(v, u):
        for _, attrs in G[v][u].items():
            relations.append(attrs["relation"])
    return sorted(set(relations))

# Converte uma caminhada (lista de IDs de nós) em formato de evidência textual estruturada, especificando os atributos das entidades e os tipos de relações que as conectam
def path_to_evidence(G, path):
    parts = []
    for i, node_id in enumerate(path):
        node = G.nodes[node_id]

        node_text = (
            f"{node_id} "
            f"[{node['node_type']}] "
            f"\"{node['label']}\" "
            f"source={node['source']} "
            f"validated={node['validated']}"
        )
        parts.append(node_text)

        if i < len(path) - 1:
            next_node = path[i+1]
            relations = relation_between(G, node_id, next_node)
            parts.append(f"relation={','.join(relations)}")

    return " | ".join(parts)

# Percorre as arestas de saída de um nó de autor e retorna a lista de publicações científicas às quais ele está vinculado pela relação 'author_of'
def works_of_author(G, author_id):
    works = []
    for _, target, edge_data in G.out_edges(author_id, data=True):
        if edge_data.get("relation") == "author_of":
            node = G.nodes[target]
            works.append({
                "id": target,
                "title": node.get("label"),
                "year": node.get("year"),
                "abstract": node.get("abstract", ""),
                "source": node.get("source"),
                "validated": node.get("validated")
            })
    return works

# Constrói o Grafo de Conhecimento dinâmico unindo os dados validados do docente (Sucupira) com os dados coletados do candidato, suas afiliações, produções e coautores (OpenAlex)
def build_dynamic_graph(validated_docent_name: str, candidate: dict):
    G = nx.MultiDiGraph()

    def add_node(node_id, node_type, label, source, validated, **attrs):
        G.add_node(node_id, node_type=node_type, label=label, source=source, validated=validated, **attrs)

    def add_edge(source_node, target_node, relation, source, validated, **attrs):
        G.add_edge(source_node, target_node, relation=relation, source=source, validated=validated, **attrs)

    add_node("A1", "author", validated_docent_name, "Sucupira", True)
    add_node("P1", "graduate_program", "Programa de Pós-Graduação em Computação", "Sucupira", True)
    add_node("I1", "institution", "Universidade Federal de São Carlos", "Sucupira", True)
    
    add_edge("A1", "P1", "member_of_ppg", "Sucupira", True)
    add_edge("P1", "I1", "hosted_by", "Sucupira", True)

    cand_id = "A2"
    add_node(cand_id, "author", candidate["name"], "OpenAlex", False)

    # Conecta o candidato à instituição de São Carlos se houver correspondência, ou cria um novo nó de instituição externa
    for aff in candidate.get("affiliations", []):
        if "São Carlos" in aff or "UFSCar" in aff or "USP" in aff:
            add_edge(cand_id, "I1", "affiliated_with", "OpenAlex", False)
        else:
            inst_id = f"I_{abs(hash(aff)) % 10000}"
            add_node(inst_id, "institution", aff, "OpenAlex", False)
            add_edge(cand_id, inst_id, "affiliated_with", "OpenAlex", False)

    # Popula os trabalhos do candidato e adiciona coautores como nós secundários no grafo
    for i, work in enumerate(candidate.get("works", []), start=1):
        work_id = f"W2_{i}"
        add_node(work_id, "work", work["title"], "OpenAlex", False, year=work["year"], abstract=work["abstract"])
        add_edge(cand_id, work_id, "author_of", "OpenAlex", False)
        
        for coauthor in work.get("coauthors", []):
            if coauthor != candidate["name"]:
                co_id = f"A_co_{abs(hash(coauthor)) % 10000}"
                if not G.has_node(co_id):
                    add_node(co_id, "author", coauthor, "OpenAlex", False)
                add_edge(co_id, work_id, "author_of", "OpenAlex", False)

    return G, "A1", cand_id

# Fluxo de execução principal: carrega amostra de docentes, busca candidatos, gera o grafo de conhecimento e consulta a LLM para resolução de entidade
if __name__ == "__main__":
    csv_filename = "br-capes-colsucup-docente-2024-2025-12-01.csv"
    
    print("--- LENDO BASE DA SUCUPIRA (CAPES) ---")
    df_docentes = load_sucupira_docents(csv_filename, ppg_filter="COMPUTAÇÃO", city_filter="SÃO CARLOS")
    
    if df_docentes is not None and not df_docentes.empty:
        coluna_nome = "NM_DOCENTE" if "NM_DOCENTE" in df_docentes.columns else df_docentes.columns[0]
        docentes_amostra = df_docentes[coluna_nome].dropna().unique()[:5]
        
        print(f"\nAmostra de docentes selecionados para desambiguação: {list(docentes_amostra)}\n")
        
        BASE_URL = "https://agents4gov.icmc.usp.br/api/v1"
        MODEL = "externo.revejo-qwen3.6-35b-a3b"
        API_KEY = "sk-e31ea19cc3c3463684bda7f3e0c5a4a5"
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

        SYSTEM_PROMPT = """
        Você realiza resolução de entidades em grafos de conhecimento.

        A primeira entidade é um pesquisador pertencente ao conjunto validado.
        A segunda entidade é um candidato ainda não validado.

        Analise as evidências fornecidas.

        Primeiro infira os principais tópicos de pesquisa de cada autor usando os títulos e resumos das produções.

        Depois analise conjuntamente nome, produções, coautores, instituições, vínculos e caminhos encontrados no grafo.

        Similaridade temática é uma evidência adicional.
        Ela não deve ser utilizada isoladamente para confirmar ou rejeitar uma identidade.

        Retorne exclusivamente um objeto JSON válido.

        O objeto deve conter os campos:
        same_person, confidence, validated_author_topics, candidate_author_topics, topic_evidence, structural_evidence, contrary_evidence, reasoning_summary, decision

        decision deve assumir um dos valores: validate_candidate, reject_candidate, uncertain
        """.strip()

        for docente in docentes_amostra:
            print("=" * 70)
            print(f"PROCESSANDO DOCENTE VALIDADO: {docente}")
            print("=" * 70)
            
            candidatos = fetch_openalex_candidates(docente, tau_lex=0.65)
            print(f"Total de candidatos encontrados no OpenAlex: {len(candidatos)}\n")
            
            for c in candidatos:
                print(f"--> Analisando Candidato OpenAlex: {c['name']} (ID: {c['id']})")
                
                # Monta o grafo específico para a dupla (Docente Validado vs Candidato)
                G, source_id, target_id = build_dynamic_graph(docente, c)
                
                # Converte para visão não direcionada para encontrar caminhadas/conexões no grafo
                UG = nx.Graph(G)
                MAX_DEPTH = 5
                paths = list(nx.all_simple_paths(UG, source=source_id, target=target_id, cutoff=MAX_DEPTH))
                
                # Formata evidências e estrutura o contexto em JSON para o prompt da LLM
                path_evidence = [path_to_evidence(G, p) for p in paths]
                validated_author_works = works_of_author(G, source_id)
                candidate_author_works = works_of_author(G, target_id)

                context = {
                    "validated_author": {
                        "id": source_id,
                        "name": docente,
                        "source": "Sucupira",
                        "works": validated_author_works
                    },
                    "candidate_author": {
                        "id": target_id,
                        "name": c["name"],
                        "source": "OpenAlex",
                        "works": candidate_author_works
                    },
                    "graph_paths": path_evidence,
                    "known_validated_facts": [
                        {"subject": source_id, "relation": "member_of_ppg", "object": "P1"},
                        {"subject": "P1", "relation": "hosted_by", "object": "I1"}
                    ]
                }

                USER_PROMPT = f"Analise o seguinte par de pesquisadores:\n\nCONTEXTO:\n{json.dumps(context, ensure_ascii=False, indent=2)}".strip()

                print("Enviando evidências para a LLM...")
                try:
                    response = client.chat.completions.create(
                        model=MODEL,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": USER_PROMPT}
                        ],
                        temperature=0
                    )
                    content = response.choices[0].message.content
                    result = json.loads(content)
                    print("\n--- DECISÃO DA LLM ---")
                    pprint(result)
                    print("-" * 50)
                except Exception as e:
                    print(f"Erro ao consultar a LLM: {e}\n")
    else:
        print("\n[Aviso] Não foi possível carregar a lista de docentes. Verifique o nome/caminho do arquivo CSV.")