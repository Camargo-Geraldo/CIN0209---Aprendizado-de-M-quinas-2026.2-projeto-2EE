# Previsão de inadimplência em cartões de crédito

Projeto de classificação da disciplina CIN0209 — Aprendizado de Máquina 1, CIn/UFPE, 2026.2. Desenvolvido por Ontje Simon Rath e Geraldo Camargo.

## Arquivos

- [Notebook da Entrega 2](CIN0209_Entrega2_modelo_CreditCardDefault.ipynb) — preparação, comparação de modelos, curvas de validação e avaliação final.
- [Slides do seminário](Seminario_CIN0209_final.ppt) — apresentação em PowerPoint.
- [Módulo de apoio](am1_util.py) e [divisão da Entrega 1](divisao_e1.json) — arquivos utilizados pelo notebook.
- Os CSVs reúnem as tabelas de validação e teste, as predições do modelo escolhido e os registros de avaliação.

## Protocolo

A busca e a seleção usaram validação cruzada estratificada com cinco folds. Após selecionar as configurações, os modelos foram reajustados nos 25.470 registros de desenvolvimento e avaliados nos 4.495 registros reservados para o teste. O Naive Bayes Gaussiano, escolhido pela acurácia balanceada na validação cruzada, obteve 69,66% no teste.

O notebook carrega a base Default of Credit Card Clients do UCI Machine Learning Repository. Para executá-lo no Colab, siga a seção de configuração do notebook e coloque os arquivos auxiliares no caminho indicado por `PASTA`.
