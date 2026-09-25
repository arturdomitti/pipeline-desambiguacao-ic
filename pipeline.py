import os
import re
import json
import uuid
import time
import argparse
import traceback
import requests
import difflib
import pandas as pd
import networkx as nx
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()  # carrega variáveis de um arquivo .env na mesma pasta, se existir
except ImportError:
    pass  # se python-dotenv não estiver instalado, segue usando só variáveis de ambiente do sistema


# ==============================================================================
# CONFIGURAÇÕES E GERENCIAMENTO DE CHECKPOINTS (CACHE)
# ==============================================================================

CACHE_DIR = ".pipeline_cache"
MAILTO = "arturdomitti@usp.br"  # usado no "polite pool" do OpenAlex para aumentar o rate limit
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY")  # gratuito em openalex.org/settings/api; eleva o orçamento diário de $0.10 (sem key, compartilhado por IP) para $1 (por conta)

def ensure_cache_dir():
    """Garante a existência do diretório de estado/cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)

def load_checkpoint(filename: str):
    """Carrega dados salvos de um passo anterior se o arquivo existir."""
    filepath = os.path.join(CACHE_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def save_checkpoint(filename: str, data):
    """Salva estado intermediário em formato JSON para persistência rápida."""
    ensure_cache_dir()
    filepath = os.path.join(CACHE_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def append_llm_result(result_row: dict, filename="llm_results_checkpoint.jsonl"):
    """Salva incrementalmente cada resposta da LLM em arquivo JSONL."""
    ensure_cache_dir()
    filepath = os.path.join(CACHE_DIR, filename)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(result_row, ensure_ascii=False) + "\n")

def load_llm_results_checkpoint(filename="llm_results_checkpoint.jsonl") -> dict:
    """Carrega os vereditos já processados para evitar reconsultar a LLM."""
    filepath = os.path.join(CACHE_DIR, filename)
    processed = {}
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    # Chave única formada por: Docente_ID_Candidato
                    key = f"{row['Docente_Sucupira']}_{row['OpenAlex_ID']}"
                    processed[key] = row
    return processed


# ==============================================================================
# FUNÇÕES DE COLETA DE DADOS (SUCUPIRA & OPENALEX)
# ==============================================================================

def calculate_name_similarity(name1: str, name2: str) -> float:
    matcher = difflib.SequenceMatcher(None, name1.lower(), name2.lower())
    return float(matcher.ratio())


def fetch_with_retry(url, params, headers, max_retries=5, timeout=15):
    """
    Faz requests.get com retry e backoff exponencial em caso de 429 (rate limit)
    ou erro de rede. Retorna o objeto Response em caso de sucesso, ou None se
    todas as tentativas falharem.
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            if response.status_code == 429:
                wait = 2 ** attempt  # 1, 2, 4, 8, 16s
                time.sleep(wait)
                continue
            return response
        except requests.exceptions.RequestException:
            wait = 2 ** attempt
            time.sleep(wait)
    return None


def fetch_author_works_from_openalex(author_id: str, max_works: int = 10, max_year: int = 2024) -> list:
    clean_id = author_id.split("/")[-1] if "/" in str(author_id) else author_id
    url = "https://api.openalex.org/works"

    params = {
        "filter": f"author.id:{clean_id},from_publication_date:1900-01-01,to_publication_date:{max_year}-12-31",
        "per_page": max_works,
        "sort": "publication_date:desc",
        "mailto": MAILTO,
    }
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY
    headers = {"User-Agent": f"IC-WebSensors ({MAILTO})"}

    response = fetch_with_retry(url, params, headers)
    if response is None or response.status_code != 200:
        return []

    try:
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
    url = "https://api.openalex.org/authors"
    params = {"search": target_name, "per_page": top_n, "mailto": MAILTO}
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY
    headers = {"User-Agent": f"IC-WebSensors ({MAILTO})"}
    candidates = []

    response = fetch_with_retry(url, params, headers)
    if response is None or response.status_code != 200:
        return []

    try:
        data = response.json()
        raw_results = data.get("results", []) if isinstance(data, dict) else []

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
            affiliations = [inst.get("display_name") for inst in raw_insts if isinstance(inst, dict) and inst.get("display_name")]

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
        print(f"Erro ao buscar candidatos no OpenAlex: {e}")
        return []


# ==============================================================================
# PIPELINE BASEADA EM STEPS (ETAPAS)
# ==============================================================================

def step_1_load_sucupira(autores_csv, prod1_csv, prod2_csv, ppg_filter, sample_size) -> list:
    """STEP 1: Carrega dados validados da Sucupira e seleciona a amostra de docentes."""
    cache_key = f"step1_docentes_sample_{sample_size}.json"
    cached_data = load_checkpoint(cache_key)
    if cached_data:
        print("--> [STEP 1] Carregado do Cache (Sucupira).")
        return cached_data

    print("\n[STEP 1] Lendo arquivos CSV da Sucupira...")
    df_autores = pd.read_csv(autores_csv, sep=";", encoding="iso-8859-1", low_memory=False, on_bad_lines='skip')

    if "NM_PROGRAMA_IES" in df_autores.columns and ppg_filter:
        df_autores = df_autores[df_autores["NM_PROGRAMA_IES"].str.contains(ppg_filter, case=False, na=False)]
    if "SG_ENTIDADE_ENSINO" in df_autores.columns:
        df_autores = df_autores[df_autores["SG_ENTIDADE_ENSINO"].str.contains("USP|UFSCAR", case=False, na=False, regex=True)]
    if "TP_AUTOR" in df_autores.columns:
        df_autores = df_autores[df_autores["TP_AUTOR"].str.upper() == "DOCENTE"]

    df_p1 = pd.read_csv(prod1_csv, sep=";", encoding="iso-8859-1", low_memory=False, on_bad_lines='skip')
    df_p2 = pd.read_csv(prod2_csv, sep=";", encoding="iso-8859-1", low_memory=False, on_bad_lines='skip')
    df_producoes = pd.concat([df_p1, df_p2], ignore_index=True)

    col_key_autores = "ID_ADD_PRODUCAO_INTELECTUAL" if "ID_ADD_PRODUCAO_INTELECTUAL" in df_autores.columns else "ID_PRODUCAO_INTELECTUAL"
    col_key_producoes = "ID_ADD_PRODUCAO_INTELECTUAL" if "ID_ADD_PRODUCAO_INTELECTUAL" in df_producoes.columns else "ID_PRODUCAO_INTELECTUAL"

    colunas_possiveis_titulo = ["NM_PRODUCAO", "DS_TITULO", "NM_TITULO", "NM_PRODUCAO_INTELECTUAL"]
    coluna_titulo = next((c for c in colunas_possiveis_titulo if c in df_producoes.columns), df_producoes.columns[0])

    colunas_possiveis_ano = ["AN_BASE_REVISION", "AN_BASE", "NU_ANO_PRODUCAO", "AN_PRODUCAO"]
    coluna_ano = next((c for c in colunas_possiveis_ano if c in df_producoes.columns), None)

    df_autores[col_key_autores] = df_autores[col_key_autores].astype(str).str.strip().str.replace(".0", "", regex=False)
    df_producoes[col_key_producoes] = df_producoes[col_key_producoes].astype(str).str.strip().str.replace(".0", "", regex=False)

    # --- CORREÇÃO DO KEYERROR AQUI ---
    # Garante que a coluna de ano seja incluída na mesclagem (merge)
    cols_to_use = [col_key_producoes, coluna_titulo]
    if coluna_ano and coluna_ano in df_producoes.columns:
        cols_to_use.append(coluna_ano)

    df_producoes_subset = df_producoes[cols_to_use].drop_duplicates()

    df_merged = pd.merge(df_autores, df_producoes_subset, left_on=col_key_autores, right_on=col_key_producoes, how="inner")

    coluna_autor = "NM_AUTOR" if "NM_AUTOR" in df_merged.columns else "NM_DOCENTE"
    docentes = df_merged[coluna_autor].dropna().unique()[:sample_size]

    dataset = []
    for doc in docentes:
        df_doc = df_merged[df_merged[coluna_autor].str.upper() == doc.upper()].copy()

        # Trata o ano de forma segura verificando se a coluna realmente existe no DataFrame mesclado
        has_year = coluna_ano and coluna_ano in df_doc.columns
        if has_year:
            df_doc[coluna_ano] = pd.to_numeric(df_doc[coluna_ano], errors="coerce").fillna(0).astype(int)
            df_doc = df_doc[df_doc[coluna_ano] <= 2024].sort_values(by=coluna_ano, ascending=False)

        df_doc = df_doc.drop_duplicates(subset=[coluna_titulo])
        works = []
        for _, r in df_doc.head(10).iterrows():
            ano_val = int(r[coluna_ano]) if has_year and pd.notna(r[coluna_ano]) and r[coluna_ano] != 0 else "N/A"
            works.append({
                "title": str(r[coluna_titulo]),
                "year": ano_val
            })

        dataset.append({"docente_name": doc, "validated_works": works})

    save_checkpoint(cache_key, dataset)
    print(f"--> [STEP 1] Concluído e salvo: {len(dataset)} docentes na amostra.")
    return dataset


def step_2_fetch_candidates(dataset_docentes, sample_size: int) -> list:
    """STEP 2: Consulta a API do OpenAlex usando o tamanho da amostra na chave do cache."""
    cache_key = f"step2_openalex_candidates_sample_{sample_size}.json"
    cached_data = load_checkpoint(cache_key)
    if cached_data:
        print(f"--> [STEP 2] Carregado do Cache ({len(cached_data)} docentes).")
        return cached_data

    print("\n[STEP 2] Consultando candidatos no OpenAlex...")
    enriched_dataset = []

    for item in dataset_docentes:
        doc_name = item["docente_name"]
        print(f"  - Buscando para: {doc_name}")
        cands = fetch_openalex_candidates(doc_name, tau_lex=0.65, top_n=5, max_year=2024)
        enriched_dataset.append({
            "docente_name": doc_name,
            "validated_works": item["validated_works"],
            "candidates": cands
        })

    save_checkpoint(cache_key, enriched_dataset)
    print("--> [STEP 2] Concluído e salvo.")
    return enriched_dataset


def step_3_build_graph_contexts(enriched_dataset, sample_size: int) -> list:
    """STEP 3: Constrói os grafos usando o tamanho da amostra na chave do cache."""
    cache_key = f"step3_graph_contexts_sample_{sample_size}.json"
    cached_data = load_checkpoint(cache_key)
    if cached_data:
        print(f"--> [STEP 3] Carregado do Cache ({len(cached_data)} pares).")
        return cached_data

    print("\n[STEP 3] Gerando Grafos e Extraindo Evidências Estruturais...")
    eval_pairs = []

    for item in enriched_dataset:
        doc_name = item["docente_name"]
        val_works = item["validated_works"]

        for c in item["candidates"]:
            # Constrói o grafo
            G = nx.MultiDiGraph()
            G.add_node("A1", node_type="author", label=doc_name, source="Sucupira", validated=True)
            G.add_node("P1", node_type="graduate_program", label="PPG Computação", source="Sucupira", validated=True)
            G.add_node("I1", node_type="institution", label="UFSCar / USP", source="Sucupira", validated=True)
            G.add_edge("A1", "P1", relation="member_of_ppg", source="Sucupira", validated=True)
            G.add_edge("P1", "I1", relation="hosted_by", source="Sucupira", validated=True)

            for i, w in enumerate(val_works, start=1):
                w_id = f"W1_{i}"
                G.add_node(w_id, node_type="work", label=w["title"], source="Sucupira", validated=True, year=w["year"])
                G.add_edge("A1", w_id, relation="author_of", source="Sucupira", validated=True)

            cand_id = "A2"
            G.add_node(cand_id, node_type="author", label=c["name"], source="OpenAlex", validated=False)

            for aff in c.get("affiliations", []):
                if "São Carlos" in aff or "UFSCar" in aff or "USP" in aff:
                    G.add_edge(cand_id, "I1", relation="affiliated_with", source="OpenAlex", validated=False)
                else:
                    inst_id = f"I_{uuid.uuid4().hex[:8]}"
                    G.add_node(inst_id, node_type="institution", label=aff, source="OpenAlex", validated=False)
                    G.add_edge(cand_id, inst_id, relation="affiliated_with", source="OpenAlex", validated=False)

            cand_works = []
            for i, w in enumerate(c.get("works", []), start=1):
                w_id = f"W2_{i}"
                G.add_node(w_id, node_type="work", label=w["title"], source="OpenAlex", validated=False, year=w["year"])
                G.add_edge(cand_id, w_id, relation="author_of", source="OpenAlex", validated=False)
                cand_works.append({"title": w["title"], "year": w["year"]})

            # Extração de Caminhos no Grafo
            UG = nx.Graph(G)
            paths = list(nx.all_simple_paths(UG, source="A1", target=cand_id, cutoff=5))

            path_evidences = []
            for p in paths:
                parts = []
                for idx_node, n_id in enumerate(p):
                    nd = G.nodes[n_id]
                    parts.append(f"{n_id} [{nd['node_type']}] \"{nd['label']}\"")
                path_evidences.append(" | ".join(parts))

            context = {
                "validated_author": {"name": doc_name, "source": "Sucupira", "works": val_works},
                "candidate_author": {"name": c["name"], "source": "OpenAlex", "works": cand_works},
                "graph_paths": path_evidences
            }

            eval_pairs.append({
                "docente_name": doc_name,
                "candidate": c,
                "validated_works": val_works,
                "candidate_works": cand_works,
                "context": context
            })

    save_checkpoint(cache_key, eval_pairs)
    print(f"--> [STEP 3] Concluído e salvo: {len(eval_pairs)} pares para avaliação.")
    return eval_pairs


def step_4_process_llm_with_resilience(eval_pairs, client, model_name) -> list:
    """STEP 4: Chama a LLM com salvamento incremental para suportar falhas/reboots."""
    print("\n[STEP 4] Enviando requisições para a LLM com Resiliência/Checkpoint...")

    already_processed = load_llm_results_checkpoint()
    results = list(already_processed.values())

    SYSTEM_PROMPT = """
    Você realiza resolução de entidades em grafos de conhecimento.
    Retorne exclusivamente um objeto JSON válido contendo:
    same_person, confidence, validated_author_topics, candidate_author_topics, topic_evidence, structural_evidence, contrary_evidence, reasoning_summary, decision
    decision deve assumir um dos valores: validate_candidate, reject_candidate, uncertain
    """.strip()

    for idx, pair in enumerate(eval_pairs, start=1):
        doc_name = pair["docente_name"]
        cand = pair["candidate"]
        cand_id = cand["id"]
        pair_key = f"{doc_name}_{cand_id}"

        if pair_key in already_processed:
            print(f"  [{idx}/{len(eval_pairs)}] Pulo (Já processado no Checkpoint): {doc_name} <-> {cand['name']}")
            continue

        print(f"  [{idx}/{len(eval_pairs)}] Consultando LLM: {doc_name} <-> {cand['name']} (ID: {cand_id})...")

        USER_PROMPT = f"Analise o seguinte par de pesquisadores:\n\nCONTEXTO:\n{json.dumps(pair['context'], ensure_ascii=False, indent=2)}".strip()

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT}
                ],
                temperature=0
            )
            content = response.choices[0].message.content
            cleaned_content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.MULTILINE)
            res = json.loads(cleaned_content)

            trabalhos_sucupira_str = "\n".join([f"{w['title']} ({w['year']})" for w in pair["validated_works"]])
            trabalhos_openalex_str = "\n".join([f"{w['title']} ({w['year']})" for w in pair["candidate_works"]])

            row = {
                "Docente_Sucupira": doc_name,
                "Candidato_OpenAlex": cand["name"],
                "OpenAlex_ID": cand_id,
                "Similaridade_Lexical": round(cand["lexical_similarity"], 2),
                "Decisao_Modelo": res.get("decision"),
                "Mesma_Pessoa": res.get("same_person"),
                "Confianca_LLM": res.get("confidence"),
                "Topicos_Sucupira": ", ".join(res.get("validated_author_topics", [])),
                "Topicos_OpenAlex": ", ".join(res.get("candidate_author_topics", [])),
                "Obras_Sucupira": trabalhos_sucupira_str,
                "Obras_OpenAlex": trabalhos_openalex_str,
                "Evidencia_Estrutural": res.get("structural_evidence"),
                "Evidencia_Tematica": res.get("topic_evidence"),
                "Evidencia_Contraria": res.get("contrary_evidence"),
                "Resumo_Raciocinio": res.get("reasoning_summary")
            }

            # Salva IMEDIATAMENTE no arquivo de checkpoint incremental
            append_llm_result(row)
            results.append(row)
            print("    -> Sucesso e salvo no Checkpoint!")

        except Exception as e:
            print(f"    -> Erro na requisição LLM: {e}. O progresso anterior permanece seguro.")

    print(f"--> [STEP 4] Concluído. Total de avaliações finalizadas: {len(results)}")
    return results


def step_5_export_reports(results, output_path):
    """STEP 5: Exporta os resultados finais consolidando a execução."""
    print("\n[STEP 5] Exportando relatórios finais...")
    if not results:
        print("Aviso: Nenhum resultado para exportar.")
        return

    df_out = pd.DataFrame(results)
    csv_output = output_path.replace(".xlsx", ".csv")

    try:
        df_out.to_excel(output_path, index=False)
        print(f" Planilha Excel gerada: {output_path}")
    except Exception as e:
        print(f"Aviso ao salvar Excel (necessário openpyxl): {e}")

    df_out.to_csv(csv_output, index=False, sep=";", encoding="utf-8-sig")
    print(f" Arquivo CSV gerado: {csv_output}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Pipeline Resiliente de Desambiguação")
    parser.add_argument("--autores", default="br-capes-colsucup-prod-autor-2021a2024-2025-12-01-bibliografica-artpe-2024.csv")
    parser.add_argument("--producoes1", default="br-capes-colsucup-producao-2021a2024-2025-12-01-bibliografica-artpe-p1.csv")
    parser.add_argument("--producoes2", default="br-capes-colsucup-producao-2021a2024-2025-12-01-bibliografica-artpe-p2.csv")
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--model", default="cortex-icmc")
    parser.add_argument("--output", default="teste_desambiguacao_ccmc.xlsx")
    parser.add_argument("--reset-cache", action="store_true", help="Apaga todos os checkpoints locais antes de iniciar")

    args = parser.parse_args()

    if args.reset_cache:
        import shutil
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
            print("Cache resetado!")

    api_key = os.getenv("AGENTS4GOV_API_KEY")
    if not api_key:
        api_key = input("Insira a sua API Key do Agents4Gov: ").strip()

    client = OpenAI(api_key=api_key, base_url="https://agents4gov.icmc.usp.br/api/v1")

    # Execução encadeada dos Steps com tolerância a falhas
    dataset = step_1_load_sucupira(args.autores, args.producoes1, args.producoes2, "CIÊNCIAS DA COMPUTAÇÃO E MATEMÁTICA COMPUTACIONAL", args.sample_size)
    enriched_dataset = step_2_fetch_candidates(dataset, args.sample_size)
    eval_pairs = step_3_build_graph_contexts(enriched_dataset, args.sample_size)
    llm_results = step_4_process_llm_with_resilience(eval_pairs, client, args.model)
    step_5_export_reports(llm_results, args.output)

    print("\n" + "="*70)
    print("PIPELINE EXECUTADO COM SUCESSO E RESILIÊNCIA!")
    print("="*70)

if __name__ == "__main__":
    main()