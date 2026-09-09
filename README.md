# Resolução de entidades em grafos de conhecimento com LLM

Este repositório contém a implementação da pipeline inicial de desambiguação de entidades (Checkpoint 1), desenvolvida como parte do meu projeto de Iniciação Científica.

O código integra dados provenientes do Sucupira/CAPES para identificar os pesquisadores validados e utiliza o OpenAlex para achar os possíveis candidatos correspondentes a esses autores. Além da similaridade lexical e das caminhadas pelo KG (evidência estrutural), ele também utiliza a LLM do laboratório para fazer inferência dos tópicos de pesquisa dos pesquisadores para resolver as entidades.

## Arquitetura da Pipeline

1. **Ingestão dos Dados Validados (Sucupira):** Leitura e filtragem dos docentes cadastrados. Estou utilizando o filtro de Programas de Pós-Graduação (PPGs) de Ciência da Computação em São Carlos (USP/UFSCar).

2. **Busca de Candidatos (OpenAlex):** Consulta da API do OpenAlex para recuperação de autores candidatos com base no limiar de similaridade lexical ($\tau_{lex} \ge 0.65$).

3. **Construção do Grafo Dinâmico:** Estruturação de (`MultiDiGraph`) representando relações de afiliação, vínculos de pós-graduação, produções científicas e coautoria.

4. **Extração de Caminhadas:** Algoritmo de busca por caminhos simples (`all_simple_paths`) delimitados por profundidade para geração de evidências estruturais.

5. **Inferência LLM (JSON Estruturado):** Submissão do contexto JSON para o servidor do laboratório, obtendo veredito desambiguado em formato estruturado (`validate_candidate`, `reject_candidate` ou `uncertain`).

## Tecnologias Utilizadas

* **Python 3.10+**

* **NetworkX:** Construção e travessia do Grafo de Conhecimento.

* **Pandas:** Leitura e filtragem da base de dados Abertos da CAPES.

* **OpenAI API:** Interface de comunicação com o servidor de LLM do laboratório (`qwen3.6-35b`).

* **Requests & RapidFuzz/Difflib:** Requisições HTTP e cálculo de similaridade lexical de nomes.

## Como Executar Localmente

1. **Clone o repositório:**

   ```bash
   git clone https://github.com/arturdomitti/pipeline-desambiguacao-ic.git
   cd pipeline-desambiguacao-ic
   ```
