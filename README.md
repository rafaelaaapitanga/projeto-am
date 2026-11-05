# TabM vs. Gradient Boosting em Dados Tabulares

Projeto final da disciplina de **Aprendizagem de Máquina** (UFPE, 2026.1), em grupo.
Avaliamos o **TabM** (*Parameter-Efficient Ensembling for Tabular Data*, ICLR 2025) contra os três
principais modelos de gradient boosting (**LightGBM**, **CatBoost** e **XGBoost**) em 9 datasets
do benchmark **TabArena-v0.1**.

> Este repositório parte do template da disciplina. Aqui estão a implementação do modelo do grupo,
> os experimentos e a análise dos resultados.

## Resumo dos resultados

Média sobre os 9 datasets, nos conjuntos de **teste** (split 70/30 estratificado, seed 42):

| Modelo   | AUC (OvO)       | Acurácia        | G-Mean          | Tempo (s, CPU) |
|----------|-----------------|-----------------|-----------------|----------------|
| **TabM** | **0,873 ± 0,093** | **0,816 ± 0,119** | **0,694 ± 0,121** | 642,7          |
| LightGBM | 0,851 ± 0,081   | 0,804 ± 0,109   | 0,580 ± 0,245   | 0,89           |
| CatBoost | 0,854 ± 0,084   | 0,805 ± 0,112   | 0,581 ± 0,244   | 2,57           |
| XGBoost  | 0,835 ± 0,094   | 0,797 ± 0,112   | 0,580 ± 0,247   | 1,11           |

**Principais observações**

- O TabM obteve a melhor média em AUC, acurácia e G-Mean entre os quatro modelos.
- O custo é alto: em CPU, foi cerca de **250× mais lento** que o CatBoost (642 s contra 2,6 s).
- Os melhores resultados relativos do TabM apareceram em **datasets pequenos** e em problemas **multiclasse**.
  Em datasets grandes, os quatro modelos ficaram praticamente empatados (AUC próximo de 0,92 a 0,93).
- O XGBoost teve o maior AUC no treino, mas o menor no teste, sinal de sobreajuste (*overfitting*) nesses experimentos.

**Como ler estes números com cuidado**

- O desvio-padrão (≈ 0,09) vem principalmente da diferença entre datasets, e as médias dos modelos
  diferem por pouco (≈ 0,02 de AUC).
- Usamos um único split (seed 42) e não aplicamos teste estatístico. Os resultados indicam uma tendência,
  não uma superioridade comprovada.

## O que é o TabM

Redes neurais costumam perder para árvores de decisão em dados tabulares e variam bastante com a
inicialização aleatória. Um *ensemble* tradicional reduz essa variância, mas exige treinar N redes
independentes, e o custo cresce linearmente.

O TabM propõe um **ensemble implícito**: um MLP simples em que **k = 32 sub-redes** compartilham a maior
parte dos pesos, são treinadas ao mesmo tempo e têm suas previsões **combinadas por média**:

```
ŷ = (1/k) · Σ hᵢ(x),   i = 1..k
```

A ideia é obter qualidade próxima à de um ensemble com custo próximo ao de uma única rede.

**Pipeline do modelo:** features tabulares → embedding numérico (`none`, `pbld`, `plr`, ou `lt` na análise de sensibilidade)
→ MLP com pesos compartilhados → k previsões → média → probabilidades por classe.

**Forma de aprendizado:** aprendizado supervisionado (dados rotulados). A retropropagação ocorre
simultaneamente nas k sub-redes, o que incentiva representações complementares: os erros de cada
sub-rede tendem a se cancelar, e o resultado fica mais robusto.

## Hiperparâmetros

| Hiperparâmetro | Faixa           | Papel                                   |
|----------------|-----------------|-----------------------------------------|
| `k` (cabeças)  | 32              | Número de membros do ensemble           |
| `lr`           | [1e-4, 1e-2]    | Taxa de aprendizado                     |
| `n_epochs`     | [50, 200]       | Número de épocas de treinamento         |
| `dropout`      | [0,0, 0,5]      | Regularização (evita overfitting)       |
| `weight_decay` | [1e-6, 1e-2]    | Regularização (evita overfitting)       |

**Ajuste de hiperparâmetros:** busca com **Optuna** (métrica: AUC em validação cruzada), com 50 *trials* por dataset
(menos trials em `connect-4`).

**Análise de sensibilidade:** comparamos o tipo de embedding numérico, o número de cabeças `k` (de 8 a 64)
e o número de épocas (de 50 a 300). Nos boxplots, o embedding `pbld` teve a maior mediana de AUC,
`k` intermediário-alto (≈ 40 a 56) tendeu a medianas maiores que valores baixos, e mais épocas não trouxeram
ganho visível.

## Metodologia

- **Datasets:** 9 do TabArena-v0.1, sendo 3 pequenos (n < 1k), 3 médios (1k a 10k) e 3 grandes (> 10k).
- **Split:** 70/30 estratificado por classe, `seed = 42`.
- **Métricas:** AUC (One-vs-One), acurácia, G-Mean, entropia cruzada e tempo total.
- **Modelos comparados:** TabM, LightGBM, CatBoost e XGBoost (via `pytabkit`, com *tuned defaults*).
- **Ambiente:** Python 3.11.15, `pytabkit` 1.7.3, `torch` 2.7.1.

| Dataset           | N      | Regime |
|-------------------|--------|--------|
| blood-transfusion | 748    | small  |
| diabetes          | 768    | small  |
| balance-scale     | 625    | small  |
| credit-g          | 1.000  | medium |
| phoneme           | 5.404  | medium |
| cmc               | 1.473  | medium |
| adult             | 48.842 | large  |
| bank-marketing    | 45.211 | large  |
| connect-4         | 67.557 | large  |

**Melhor AUC em validação cruzada na busca Optuna (TabM):**

| Dataset           | AUC (CV) | Dataset        | AUC (CV) | Dataset   | AUC (CV) |
|-------------------|----------|----------------|----------|-----------|----------|
| balance-scale     | 0,9880   | cmc            | 0,7298   | adult     | 0,9141   |
| blood-transfusion | 0,7617   | credit-g       | 0,7793   | bank-marketing | 0,9392 |
| diabetes          | 0,8187   | phoneme        | 0,9475   | connect-4 | 0,9259   |

## Aplicações e limitações

**Aplicações práticas:** análise de crédito e *scoring*, detecção de fraude, previsão de *churn*,
aplicações médicas e de diagnóstico, sistemas de recomendação.

**Limitações:**

- Custo computacional alto (nos nossos testes, em CPU, centenas de vezes mais lento que os modelos de árvores).
- É uma caixa-preta: não tem interpretabilidade nativa das features.
- Tende a exigir GPU para ser viável em produção.
- Menos eficiente em datasets muito grandes sem aceleração.

**Quando usar:** qualidade é prioridade, há GPU disponível e o dataset é pequeno ou médio.
**Quando evitar:** latência restrita, datasets grandes sem GPU ou interpretabilidade obrigatória.

## Como reproduzir

Pré-requisitos: Python 3.11 ou superior e [`uv`](https://docs.astral.sh/uv/) (recomendado) ou `pip`.

```bash
# clonar o repositório
git clone https://github.com/rafaelaaapitanga/projeto-am.git
cd projeto-am

# opção A: uv (recomendado)
uv sync

# opção B: pip
pip install -e .

# smoke test: verifica que os baselines e os gráficos executam
pytest tests/test_pipeline.py -v

# experimento completo (9 datasets, com o TabM)
python -m src.pipeline.run_all --include-group-model --seed 42
```

Os resultados por dataset e por modelo são gravados em CSV na pasta `results/`.
As análises e gráficos estão nos notebooks da pasta `notebooks/`.

## Estrutura do repositório

```
projeto-am/
|- data/            # carregamento dos datasets (OpenML / TabArena)
|- src/
|   |- models/      # baselines (LightGBM, XGBoost, CatBoost) e modelo do grupo (TabM)
|   |- pipeline/    # split, tuning (Optuna), avaliação, regimes, orquestrador
|   |- reports/     # tabelas e gráficos
|- notebooks/       # EDA, demos e análise de resultados
|- model_cards/     # documentação do modelo
|- results/         # resultados em CSV
|- docs/            # documentação de infraestrutura
|- tests/           # smoke test
|- entregaveis/     # relatório, slides e rubrica da disciplina
```

## Equipe

Projeto desenvolvido em grupo na disciplina de Aprendizagem de Máquina (UFPE, 2026.1),
sob orientação do Prof. Leandro Maciel Almeida:

- Aline Acioly
- Rafaela Pitanga
- Vlademir Melo

## Referência

Gorishniy, Y. et al. **TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling.**
ICLR 2025. [arXiv:2410.24210](https://arxiv.org/abs/2410.24210)
