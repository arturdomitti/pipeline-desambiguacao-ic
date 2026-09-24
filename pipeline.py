import os
import re
import json
import uuid
import argparse
import traceback
import requests
import difflib
import pandas as pd
import networkx as nx
from pprint import pprint
from openai import OpenAI

# ==============================================================================
# 1. FUNÇÕES AUXILIARES DE PROCESSAMENTO DE DADOS E API
# ==============================================================================

def calculate_name_similarity(name1: str, name2: str) -> float:
    """Calcula a similaridade lexical entre o nome validado (Sucupira) e do candidato (OpenAlex)."""
    matcher = difflib.SequenceMatcher(None, name1.lower(), name2.lower())
    return float(matcher.ratio())


def fetch_author_works_from_openalex(author_id: str, max_works: int = 10, max_year: int = 2024) -> list:
    """Coleta as publicações de um determinado autor no OpenAlex até o ano limite (ex: 2024)."""
    clean_id = author_id.split("/")[-1] if "/" in str(author_id) else author_id
    url = "https://api.openalex.org/works"
    
    params = {
        "filter": f"author.id:{clean_id},from_publication_date:1900-01-01,to_publication_date:{max_year}-12-31",
        "per_page": max_works,
        "sort": "publication_date:desc"
    }
    headers = {"User-Agent": "mailto:arturdomitti@usp.br"}

    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        if response.status_code != 200:
            return []

        results = response.json().get("results", [])
        works_data = []
        for work in results:
            coauthors = [
                m.get("author", {}).get("display_name", "")
                for m in work.get("authorships", [])
            ]

            works_data.append({
                "title": work.get("title", "Sem título"),
                "year": work.get("publication_year", None),
                "coauthors": coauthors
            })
        return works_data
    except Exception:
        return []


def fetch_openalex_candidates(target_name: str, tau_lex: float = 0.65, top_n: int = 5, max_year: int = 2024) -> list:
    """Consulta o OpenAlex em busca de autores candidatos, filtrando por similaridade lexical >= tau_lex."""
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

            works = fetch_author_works_from_openalex(auth_id, max_works=10, max_year=max_year) if auth_id else []

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


def load_sucupira_validated_dataset(autores_csv: str, producoes1_csv: str, producoes2_csv: str, ppg_filter: str = "COMPUTAÇÃO|MATEMÁTICA COMPUTACIONAL"):
    """Carrega e cruza as bases da Sucupira mapeando as colunas da tabela original."""
    try:
        print("--- CARREGANDO TABELA DE AUTORES DA PRODUÇÃO ---")
        df_autores = pd.read_csv(
            autores_csv,
            sep=";",
            encoding="iso-8859-1",
            low_memory=False,
            on_bad_lines='skip'
        )

        if "NM_PROGRAMA_IES" in df_autores.columns and ppg_filter:
            df_autores = df_autores[df_autores["NM_PROGRAMA_IES"].str.contains(ppg_filter, case=False, na=False)]

        if "SG_ENTIDADE_ENSINO" in df_autores.columns:
            df_autores = df_autores[df_autores["SG_ENTIDADE_ENSINO"].str.contains("USP|UFSCAR", case=False, na=False, regex=True)]

        if "TP_AUTOR" in df_autores.columns:
            df_autores = df_autores[df_autores["TP_AUTOR"].str.upper() == "DOCENTE"]

        print(f"Registros de docentes filtrados: {len(df_autores)}")

        print("\n--- CARREGANDO TABELA DE PRODUÇÕES (2 partes) ---")
        df_producoes_p1 = pd.read_csv(
            producoes1_csv,
            sep=";",
            encoding="iso-8859-1",
            low_memory=False,
            on_bad_lines='skip'
        )
        df_producoes_p2 = pd.read_csv(
            producoes2_csv,
            sep=";",
            encoding="iso-8859-1",
            low_memory=False,
            on_bad_lines='skip'
        )

        df_producoes = pd.concat([df_producoes_p1, df_producoes_p2], ignore_index=True)
        del df_producoes_p1, df_producoes_p2

        col_key_autores = "ID_ADD_PRODUCAO_INTELECTUAL" if "ID_ADD_PRODUCAO_INTELECTUAL" in df_autores.columns else "ID_PRODUCAO_INTELECTUAL"
        col_key_producoes = "ID_ADD_PRODUCAO_INTELECTUAL" if "ID_ADD_PRODUCAO_INTELECTUAL" in df_producoes.columns else "ID_PRODUCAO_INTELECTUAL"

        if col_key_autores not in df_autores.columns or col_key_producoes not in df_producoes.columns:
            print("\n[Erro de Estrutura] Colunas de ligação não encontradas.")
            return None

        colunas_possiveis_titulo = ["NM_PRODUCAO", "DS_TITULO", "NM_TITULO", "NM_PRODUCAO_INTELECTUAL"]
        coluna_titulo = next((c for c in colunas_possiveis_titulo if c in df_producoes.columns), df_producoes.columns[0])

        df_autores[col_key_autores] = df_autores[col_key_autores].astype(str).str.strip().str.replace(".0", "", regex=False)
        df_producoes[col_key_producoes] = df_producoes[col_key_producoes].astype(str).str.strip().str.replace(".0", "", regex=False)
        df_producoes_subset = df_producoes[[col_key_producoes, coluna_titulo]].drop_duplicates()

        df_merged = pd.merge(
            df_autores,
            df_producoes_subset,
            left_on=col_key_autores,
            right_on=col_key_producoes,
            how="inner"
        )

        print(f"Total de trabalhos validados vinculados aos docentes: {len(df_merged)}")
        return df_merged

    except Exception as e:
        print(f"Erro ao carregar/cruzar bases da Sucupira: {e}")
        traceback.print_exc()
        return None


def fetch_validated_docent_works(df_merged, docente_name: str, max_works: int = 10, max_year: int = 2024) -> list:
    """Extrai os trabalhos validados do docente limitando até max_year."""
    coluna_autor = "NM_AUTOR" if "NM_AUTOR" in df_merged.columns else "NM_DOCENTE"

    colunas_possiveis_titulo = ["NM_PRODUCAO", "DS_TITULO", "NM_TITULO", "NM_PRODUCAO_INTELECTUAL"]
    coluna_titulo = next((c for c in colunas_possiveis_titulo if c in df_merged.columns), None)
    if coluna_titulo is None:
        raise KeyError("Nenhuma coluna de título encontrada em df_merged.")

    colunas_possiveis_ano = ["AN_BASE_REVISION", "AN_BASE", "NU_ANO_PRODUCAO", "AN_PRODUCAO"]
    coluna_ano = next((c for c in colunas_possiveis_ano if c in df_merged.columns), None)

    df_docente = df_merged[df_merged[coluna_autor].str.upper() == docente_name.upper()].copy()

    if coluna_ano:
        df_docente[coluna_ano] = pd.to_numeric(df_docente[coluna_ano], errors="coerce").fillna(0).astype(int)
        df_docente = df_docente[df_docente[coluna_ano] <= max_year]
        df_docente = df_docente.sort_values(by=coluna_ano, ascending=False)

    df_docente = df_docente.drop_duplicates(subset=[coluna_titulo])

    works = []
    for _, row in df_docente.head(max_works).iterrows():
        titulo = row[coluna_titulo]
        if pd.isna(titulo):
            continue
        ano = int(row[coluna_ano]) if coluna_ano and pd.notna(row[coluna_ano]) and row[coluna_ano] != 0 else "N/A"
        works.append({
            "title": str(titulo),
            "year": ano,
            "source": "Sucupira",
            "validated": True
        })
    return works


# ==============================================================================
# 2. GRAFOS DE CONHECIMENTO E EXTRAÇÃO DE CAMINHOS
# ==============================================================================

def relation_between(G, u, v):
    relations = []
    if G.has_edge(u, v):
        for _, attrs in G[u][v].items():
            relations.append(attrs["relation"])
    if G.has_edge(v, u):
        for _, attrs in G[v][u].items():
            relations.append(attrs["relation"])
    return sorted(set(relations))


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


def works_of_author(G, author_id):
    works = []
    for _, target, edge_data in G.out_edges(author_id, data=True):
        if edge_data.get("relation") == "author_of":
            node = G.nodes[target]
            works.append({
                "title": node.get("label"),
                "year": node.get("year"),
                "source": node.get("source"),
                "validated": node.get("validated")
            })
    return works


def build_dynamic_graph(validated_docent_name: str, validated_works: list, candidate: dict):
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

    for i, work in enumerate(validated_works, start=1):
        work_id = f"W1_{i}"
        add_node(work_id, "work", work["title"], "Sucupira", True, year=work["year"])
        add_edge("A1", work_id, "author_of", "Sucupira", True)

    cand_id = "A2"
    add_node(cand_id, "author", candidate["name"], "OpenAlex", False)

    for aff in candidate.get("affiliations", []):
        if "São Carlos" in aff or "UFSCar" in aff or "USP" in aff:
            add_edge(cand_id, "I1", "affiliated_with", "OpenAlex", False)
        else:
            inst_id = f"I_{uuid.uuid4().hex[:8]}"
            add_node(inst_id, "institution", aff, "OpenAlex", False)
            add_edge(cand_id, inst_id, "affiliated_with", "OpenAlex", False)

    for i, work in enumerate(candidate.get("works", []), start=1):
        work_id = f"W2_{i}"
        add_node(work_id, "work", work["title"], "OpenAlex", False, year=work["year"])
        add_edge(cand_id, work_id, "author_of", "OpenAlex", False)

        for coauthor in work.get("coauthors", []):
            if coauthor != candidate["name"]:
                co_id = f"A_co_{uuid.uuid4().hex[:8]}"
                if not G.has_node(co_id):
                    add_node(co_id, "author", coauthor, "OpenAlex", False)
                add_edge(co_id, work_id, "author_of", "OpenAlex", False)

    return G, "A1", cand_id


# ==============================================================================
# 3. PIPELINE DE EXECUÇÃO PRINCIPAL
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Pipeline de Desambiguação Sucupira-OpenAlex")
    parser.add_argument("--autores", default="br-capes-colsucup-prod-autor-2021a2024-2025-12-01-bibliografica-artpe-2024.csv", help="Caminho do CSV de autores da Sucupira")
    parser.add_argument("--producoes1", default="br-capes-colsucup-producao-2021a2024-2025-12-01-bibliografica-artpe-p1.csv", help="Caminho da parte 1 de produções")
    parser.add_argument("--producoes2", default="br-capes-colsucup-producao-2021a2024-2025-12-01-bibliografica-artpe-p2.csv", help="Caminho da parte 2 de produções")
    parser.add_argument("--sample-size", type=int, default=5, help="Quantidade de docentes na amostra")
    parser.add_argument("--model", default="cortex-icmc", help="Modelo LLM")
    parser.add_argument("--output", default="teste_desambiguacao_ccmc.xlsx", help="Arquivo Excel de saída")

    args = parser.parse_args()

    # Busca a chave de API da variável de ambiente ou do arquivo .env
    api_key = os.getenv("AGENTS4GOV_API_KEY")
    if not api_key:
        api_key = input("Insira a sua API Key do Agents4Gov: ").strip()

    if not api_key:
        raise ValueError("API Key não pode ser vazia!")

    BASE_URL = "https://agents4gov.icmc.usp.br/api/v1"
    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    print("--- INICIANDO PIPELINE DE DESAMBIGUAÇÃO SUCUPIRA-OPENALEX ---")
    df_merged = load_sucupira_validated_dataset(
        autores_csv=args.autores,
        producoes1_csv=args.producoes1,
        producoes2_csv=args.producoes2,
        ppg_filter="CIÊNCIAS DA COMPUTAÇÃO E MATEMÁTICA COMPUTACIONAL"
    )

    if df_merged is None or df_merged.empty:
        print("\n[Erro] Não foi possível carregar a base da Sucupira.")
        return

    coluna_autor = "NM_AUTOR" if "NM_AUTOR" in df_merged.columns else "NM_DOCENTE"
    docentes_amostra = df_merged[coluna_autor].dropna().unique()[:args.sample_size]

    print(f"\nTotal de docentes selecionados: {len(docentes_amostra)}")
    print(f"Lista de docentes: {list(docentes_amostra)}\n")

    experiment_results = []

    SYSTEM_PROMPT = """
    Você realiza resolução de entidades em grafos de conhecimento.

    A primeira entidade é um pesquisador pertencente ao conjunto validado.
    A segunda entidade é um candidato ainda não validado.

    Analise as evidências fornecidas.

    Primeiro infira os principais tópicos de pesquisa de cada autor usando exclusivamente os títulos das produções.

    Depois analise conjuntamente nome, produções, coautores, instituições, vínculos e caminhos encontrados no grafo.

    Similaridade temática é uma evidência adicional.
    Ela não deve ser utilizada isoladamente para confirmar ou rejeitar uma identidade.

    Retorne exclusivamente um objeto JSON válido.
    Responda aos campos textuais do JSON em Português do Brasil.

    O objeto deve conter os campos:
    same_person, confidence, validated_author_topics, candidate_author_topics, topic_evidence, structural_evidence, contrary_evidence, reasoning_summary, decision

    decision deve assumir um dos valores: validate_candidate, reject_candidate, uncertain
    """.strip()

    for idx, docente in enumerate(docentes_amostra, start=1):
        print("=" * 70)
        print(f"[{idx}/{len(docentes_amostra)}] PROCESSANDO DOCENTE VALIDADO: {docente}")
        print("=" * 70)

        validated_works = fetch_validated_docent_works(df_merged, docente, max_works=10, max_year=2024)
        candidatos = fetch_openalex_candidates(docente, tau_lex=0.65, max_year=2024)
        print(f"Total de candidatos encontrados no OpenAlex: {len(candidatos)}\n")

        for c in candidatos:
            print(f"--> Analisando Candidato OpenAlex: {c['name']} (ID: {c['id']})")

            G, source_id, target_id = build_dynamic_graph(docente, validated_works, c)

            UG = nx.Graph(G)
            MAX_DEPTH = 5
            paths = list(nx.all_simple_paths(UG, source=source_id, target=target_id, cutoff=MAX_DEPTH))

            path_evidence = [path_to_evidence(G, p) for p in paths]
            candidate_author_works = works_of_author(G, target_id)

            context = {
                "validated_author": {
                    "name": docente,
                    "source": "Sucupira",
                    "works": validated_works
                },
                "candidate_author": {
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
                    model=args.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_PROMPT}
                    ],
                    temperature=0
                )
                content = response.choices[0].message.content
                cleaned_content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.MULTILINE)
                result = json.loads(cleaned_content)

                trabalhos_sucupira_str = "\n".join(
                    f"{w['title']} ({w['year']})" for w in validated_works
                )
                trabalhos_openalex_str = "\n".join(
                    f"{w['title']} ({w['year']})" for w in candidate_author_works
                )

                row = {
                    "Docente_Sucupira": docente,
                    "Candidato_OpenAlex": c["name"],
                    "OpenAlex_ID": c["id"],
                    "Similaridade_Lexical": round(c["lexical_similarity"], 2),
                    "Decisao_Modelo": result.get("decision"),
                    "Mesma_Pessoa": result.get("same_person"),
                    "Confianca_LLM": result.get("confidence"),
                    "Topicos_Sucupira": ", ".join(result.get("validated_author_topics", [])),
                    "Topicos_OpenAlex": ", ".join(result.get("candidate_author_topics", [])),
                    "Obras_Sucupira": trabalhos_sucupira_str,
                    "Obras_OpenAlex": trabalhos_openalex_str,
                    "Evidencia_Estrutural": result.get("structural_evidence"),
                    "Evidencia_Tematica": result.get("topic_evidence"),
                    "Evidencia_Contraria": result.get("contrary_evidence"),
                    "Resumo_Raciocinio": result.get("reasoning_summary"),
                }
                experiment_results.append(row)

                print("\n--- DECISÃO DA LLM ---")
                pprint(result)
                print("-" * 50)
            except Exception as e:
                print(f"Erro ao consultar a LLM: {e}\n")

    # Exporta resultados
    df_out = pd.DataFrame(experiment_results)
    csv_output = args.output.replace(".xlsx", ".csv")

    df_out.to_excel(args.output, index=False)
    df_out.to_csv(csv_output, index=False, sep=";", encoding="utf-8-sig")

    print("\n" + "="*70)
    print("EXPERIMENTO CONCLUÍDO COM SUCESSO!")
    print(f"Planilha Excel gerada: {args.output}")
    print(f"Arquivo CSV gerado: {csv_output}")
    print("="*70)


if __name__ == "__main__":
    main()