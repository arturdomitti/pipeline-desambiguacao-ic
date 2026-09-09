# Resolução de Entidades em Grafos de Conhecimento com LLM

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arturdomitti/pipeline-desambiguacao-ic/blob/main/pipeline.ipynb)

Este repositório contém a implementação da pipeline inicial de desambiguação de entidades, desenvolvida como parte do meu projeto de Iniciação Científica (IC).

O sistema integra dados provenientes da plataforma Sucupira/CAPES para identificar pesquisadores validados e utiliza a API do OpenAlex para recuperar candidatos correspondentes. Além da análise de similaridade lexical de nomes e das caminhadas pelo Grafo de Conhecimento (evidência estrutural), a pipeline utiliza a LLM do laboratório para realizar a inferência dos tópicos de pesquisa a partir de títulos e resumos das publicações, resolvendo as ambiguidades das entidades.

---

## Arquitetura da Pipeline

1. **Ingestão dos Dados Validados (Sucupira):** Leitura e filtragem dos docentes cadastrados na base de Dados Abertos da CAPES. Atualmente, utiliza o filtro de Programas de Pós-Graduação (PPGs) de Ciência da Computação em São Carlos (USP/UFSCar).
2. **Busca de Candidatos (OpenAlex):** Consulta à API do OpenAlex para recuperação de autores candidatos com base no limiar de similaridade lexical (tau_lex >= 0.65).
3. **Construção do Grafo Dinâmico:** Estruturação em multígrafo direcionado (`MultiDiGraph` via `NetworkX`) representando relações de afiliação, vínculos com o PPG, produções científicas e coautoria.
4. **Extração de Caminhadas:** Algoritmo de busca por caminhos simples (`all_simple_paths`) delimitados por profundidade limite para geração de evidências estruturais.
5. **Inferência LLM (JSON Estruturado):** Submissão do contexto JSON para o servidor de LLM do laboratório (`agents4gov`), obtendo o veredito desambiguado em formato JSON estrito (`validate_candidate`, `reject_candidate` ou `uncertain`).

---

## Tecnologias Utilizadas

* **Python 3.10+**
* **NetworkX:** Construção, manipulação e travessia do Grafo de Conhecimento.
* **Pandas:** Leitura, manipulação e filtragem da base de dados da CAPES.
* **OpenAI API Client:** Interface de comunicação com o servidor de LLM do laboratório (`externo.revejo-qwen3.6-35b-a3b`).
* **Requests & Difflib / RapidFuzz:** Requisições HTTP e cálculo de similaridade lexical entre nomes de pesquisadores.

---

## Como Executar Localmente

### Pré-requisitos
* **Python 3.10 ou superior** instalado.
* O arquivo CSV oficial de docentes da CAPES (`br-capes-colsucup-docente-*.csv`) baixado na raiz do projeto.

### Passo a Passo

1. **Clone o repositório:**
   git clone https://github.com/arturdomitti/pipeline-desambiguacao-ic.git
   cd pipeline-desambiguacao-ic

2. **Crie e ative o ambiente virtual:**
   - Linux / WSL / macOS:
     python3 -m venv .venv
     source .venv/bin/activate
   - Windows (PowerShell):
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1

3. **Instale as dependências do projeto:**
   pip install -r requirements.txt

4. **Execute o script principal:**
   python pipeline.py

---

## Execução Via Google Colab

Para executar a pipeline diretamente pelo navegador sem necessidade de configuração de ambiente local:

1. Clique no botão **"Open in Colab"** no topo deste README.
2. Na célula correspondente, faça o upload do arquivo CSV da Sucupira.
3. Execute as células em sequência para visualizar os grafos montados e os vereditos emitidos pela LLM em tempo real.