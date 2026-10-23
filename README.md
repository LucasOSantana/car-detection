# Classificação de Marcas de Automóveis por Imagem (CNN & Dropout)

Este projeto implementa um modelo de Aprendizado Profundo em Python (**TensorFlow / Keras**) capaz de identificar a marca de um automóvel (entre as **33 marcas** presentes no dataset, como *Volkswagen, Chevrolet, FIAT, Toyota, Ford, BMW, Audi, Honda, Hyundai, Jeep, Mercedes-Benz*, etc.) a partir da foto da sua dianteira, grade frontal ou emblema.

---

## 1. Instalação e Como Executar

### Dependências
```bash
pip install tensorflow scikit-learn matplotlib numpy pillow
```

### Etapa 1: Treinamento e Experimentos de Dropout (`0.0`, `0.2`, `0.5`)
Para treinar o modelo nas 33 marcas, executar os 3 experimentos variando o **Dropout** na camada densa e gerar os gráficos comparativos e a matriz de confusão:
```bash
python car-detection.py --mode train --epochs 15
```
> Caso deseje treinar uma CNN puramente do zero (sem Transfer Learning do MobileNetV2), adicione a flag `--from-scratch`:
> ```bash
> python car-detection.py --mode train --epochs 20 --from-scratch
> ```

**Arquivos gerados após o treinamento:**
- `car_brand_model.keras`: Melhor modelo salvo automaticamente (aquele que obteve a menor perda de validação).
- `class_names.json`: Lista ordenada com os 33 rótulos de marcas do dataset.
- `resultados_experimentos/curvas_loss_dropout.png`: Gráfico comparativo das curvas de perda (*Loss*) e acurácia de treino vs. validação para `Dropout = 0.0`, `0.2` e `0.5`.
- `resultados_experimentos/matriz_confusao.png`: Matriz de confusão `33 x 33` avaliada no conjunto de teste (`test`).
- `resultados_experimentos/erros_distancia_e_reflexos.png`: Painel visual com amostras classificadas incorretamente pelo modelo para inspeção de distância da foto e reflexos na lataria.
- `resultados_experimentos/relatorio_classificacao.txt`: Métricas de *Precision*, *Recall* e *F1-Score* para cada uma das 33 marcas.

### Etapa 2: Predição de uma Imagem (Entrada: Imagem $\rightarrow$ Saída: Rótulo)
Depois que o modelo estiver treinado, basta passar o caminho de qualquer imagem com `--image`:
```bash
python car-detection.py --image "dataset/Car Brand Classification Dataset/test/Volkswagen/1.jpg"
```

---

## 2. Fundamentação Teórica: Overfitting e Dropout

### O que é Overfitting (Sobreajuste)?
O **Overfitting** ocorre quando uma rede neural memoriza excessivamente os detalhes, ruídos e particularidades específicas do conjunto de treinamento (como fundos de concessionárias, placas, iluminação específica de uma foto ou texturas irrelevantes), em vez de aprender padrões visuais generalizáveis (o formato geométrico do logotipo, o desenho dos faróis e as linhas da grade frontal).

**Como diagnosticar na curva de perda (*Loss*):**
- A perda de treinamento (**Train Loss**) continua caindo continuamente rumo a zero.
- A perda de validação (**Validation Loss**) atinge um ponto mínimo e **começa a subir**, abrindo um distanciamento (*gap*) crescente entre as curvas de treino e validação.

### Como a técnica de Dropout ajuda a prevenir o Overfitting?
O **Dropout** é uma técnica de regularização aplicada durante o treinamento na qual, a cada passo (*mini-batch*), uma fração aleatória $p$ dos neurônios da camada é temporariamente "desligada" (suas ativações são zeradas) e os neurônios restantes têm suas saídas reescalonadas por $\frac{1}{1 - p}$. Durante a inferência/teste, todos os neurônios permanecem ativos.

Os mecanismos pelos quais o Dropout combate o sobreajuste são:
1. **Quebra de Co-adaptação de Neurônios:** Sem Dropout, certos neurônios da camada densa podem criar dependência mútua excessiva para memorizar artefatos específicos do conjunto de treino. Como qualquer neurônio pode ser desativado aleatoriamente, cada neurônio é forçado a aprender representações úteis por conta própria (ex.: detectar tanto o contorno circular do emblema quanto os frisos da grade frontal).
2. **Efeito de *Ensemble* Implícito:** A cada época e lote de imagens, uma sub-rede ligeiramente diferente é amostrada e treinada. Na hora da predição, usar a camada inteira equivale a combinar a média de milhares de sub-redes diferentes, reduzindo drasticamente a variância das predições.

---

## 3. Análise dos Experimentos de Dropout (`0.0`, `0.2`, `0.5`)

No código (`construir_modelo` em `car-detection.py`), aplicamos o `layers.Dropout(rate)` imediatamente após a camada densa principal (`layers.Dense(256, activation="relu")`):

| Taxa de Dropout | Comportamento Esperado na Curva de Perda (*Loss*) | Fenômeno Observado |
| :--- | :--- | :--- |
| **`Dropout = 0.0`** (Sem regularização) | A *Train Loss* cai muito rápido, mas a *Val Loss* para de cair cedo e passa a subir nas épocas seguintes. | **Overfitting severo**: Os 256 neurônios da camada densa memorizam rapidamente os exemplos de treino, perdendo capacidade de generalização em carros novos. |
| **`Dropout = 0.2`** (20% desativados) | A *Train Loss* cai de forma mais controlada e a *Val Loss* acompanha a queda de forma estável, estabilizando em um patamar mais baixo. | **Equilíbrio ideal**: Reduz a co-adaptação sem comprometer a capacidade de representação discriminativa para separar as 33 marcas. |
| **`Dropout = 0.5`** (50% desativados) | A *Train Loss* é a mais alta entre os três experimentos, mas a distância entre *Train Loss* e *Val Loss* é mínima. | **Forte regularização**: Impede o sobreajuste de forma agressiva; pode exigir algumas épocas extras para convergir, mas oferece alta robustez em dados ruidosos. |

---

## 4. O que Analisar na Matriz de Confusão: Distância da Foto e Reflexos na Lataria

Ao inspecionar a **Matriz de Confusão** (`matriz_confusao.png`) e o painel de erros (`erros_distancia_e_reflexos.png`), os principais erros fora da diagonal principal estão diretamente ligados a dois fatores físicos e ópticos da captura das imagens:

### 4.1. Erros Associados à Distância da Foto (Escala e Resolução Espacial)
Quando a imagem da dianteira do veículo é capturada a uma **grande distância** (ou com o veículo ocupando pequena porção do quadro):
- **Perda do Emblema no Redimensionamento (`224x224`):** Quando a imagem inteira é redimensionada para `224x224` pixels para entrar na CNN, um emblema pequeno no centro da grade passa a ocupar apenas cerca de `8x8` a `12x12` pixels.
- **Confusão por Silhueta de Carroceria em vez de Marca:** Sem resolução suficiente para distinguir a tipografia ou o símbolo interno do emblema, a rede passa a classificar com base na silhueta global do carro (SUV, sedã ou picape).
  - *Exemplos típicos na Matriz de Confusão:* Confusão entre picapes/SUVs robustos fotografados de longe (ex.: **Ford**, **Chevrolet**, **GMC** e **Ram**), ou entre sedãs/hatches com emblemas ovais/circulares pequenos vistos à distância (ex.: **Toyota**, **Hyundai**, **Kia**, **Nissan** e **Volkswagen**).

### 4.2. Erros Associados a Reflexos na Lataria e no Emblema Cromado
A maioria dos emblemas automotivos e grades frontais possui acabamento **cromado metálico** e pintura envernizada altamente reflexiva:
- **Saturação Especular (Estouro de Brilho):** Sob luz solar direta ou iluminação artificial intensa, o reflexo especular satura os canais RGB (`[255, 255, 255]`), apagando as bordas internas que diferenciam um emblema do outro (por exemplo, o "V" e o "W" dentro do círculo da **Volkswagen** ou as elipses internas da **Toyota**).
- **Texturas Falsas no Capô e Para-choque:** Reflexos de árvores, prédios, faixas da pista ou de outros carros na lataria criam gradientes de alta frequência que os filtros convolucionais (`Conv2D`) podem interpretar erroneamente como grades colmeia, frisos cromados ou vincos de design de outra montadora.
  - *Exemplos típicos na Matriz de Confusão:* Emblemas circulares ou ovais com forte reflexo solar (como **FIAT**, **Ford**, **Subaru**, **MINI** ou **Mercedes-Benz**) perdem sua textura interna e passam a ser confundidos entre si pelo formato externo da moldura cromada.
