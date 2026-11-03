"""
Sistema de Detecção e Classificação de Marcas de Automóveis por Imagem
======================================================================

Funcionalidades:
1. Treinamento e Experimentos (--mode train):
   - Carrega as 15 marcas do dataset em `dataset/marcas` (gerado por `preparar_dataset.py`)
   - Executa experimentos variando a taxa de Dropout na camada densa: [0.0, 0.2, 0.5]
   - Cada experimento treina em duas fases: (1) só a cabeça, com o backbone congelado;
     (2) fine-tuning do topo do backbone com learning rate baixo
   - Gera e salva o gráfico comparativo das curvas de perda (Loss de Treino vs. Validação)
   - Gera e salva a Matriz de Confusão no conjunto de teste e um painel com erros de predição
     (útil para analisar erros por distância da foto ou reflexos na lataria/emblema)
   - Salva o melhor modelo treinado (`car_brand_model.keras`) e os rótulos (`class_names.json`)

2. Inferência / Predição:
   - --image caminho/da/imagem.jpg: classifica uma imagem e retorna o rótulo da marca.
   - --pasta caminho/da/pasta: classifica todas as imagens da pasta e gera uma tabela CSV.

Exemplos de uso:
   python car-detection.py --mode train
   python car-detection.py --mode train --backbone efficientnetv2s --checkpoint-dir checkpoints
   python car-detection.py --image "caminho/da/foto.jpg"
   python car-detection.py --pasta fotos_proprias
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models

# Configurações padrão de diretório e hiperparâmetros
DATASET_DIR = Path("dataset") / "marcas"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"

MODEL_PATH = Path("car_brand_model.keras")
CLASSES_PATH = Path("class_names.json")
OUTPUT_DIR = Path("resultados_experimentos")
CHECKPOINT_DIR = Path("checkpoints")

IMG_SIZE = (224, 224)
BATCH_SIZE = 64
DROPOUT_RATES = [0.0, 0.2, 0.5]
SEED = 42

# Treino em duas fases
LR_FASE_CABECA = 1e-3
LR_FASE_FINE_TUNING = 1e-5
FRACAO_DESCONGELADA = 0.3  # Fração final das camadas do backbone liberada no fine-tuning

# Fundo branco: as imagens do DVM-CAR têm o fundo removido (branco)
COR_FUNDO = 255.0

BACKBONES = {
    # nome: (construtor, função de pré-processamento ou None se já embutida no modelo)
    "efficientnetv2b0": (tf.keras.applications.EfficientNetV2B0, None),
    "efficientnetv2s": (tf.keras.applications.EfficientNetV2S, None),
    "convnext_tiny": (tf.keras.applications.ConvNeXtTiny, None),
    "mobilenetv2": (
        tf.keras.applications.MobileNetV2,
        tf.keras.applications.mobilenet_v2.preprocess_input,
    ),
}
BACKBONE_PADRAO = "efficientnetv2b0"


def carregar_datasets(img_size: tuple[int, int] = IMG_SIZE, batch_size: int = BATCH_SIZE):
    """Carrega os conjuntos de treino, validação e teste com todas as marcas."""
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Diretório de treino não encontrado em: {TRAIN_DIR.resolve()}"
        )

    print(f"[INFO] Carregando dataset a partir de: {DATASET_DIR.resolve()}")

    def carregar(pasta: Path) -> tf.data.Dataset:
        # batch_size=None: imagens individuais, para o cache/shuffle abaixo funcionarem
        # por imagem. crop_to_aspect_ratio evita esticar imagens não quadradas.
        return tf.keras.utils.image_dataset_from_directory(
            pasta,
            image_size=img_size,
            batch_size=None,
            label_mode="int",
            shuffle=False,
            crop_to_aspect_ratio=True,
        )

    train_ds = carregar(TRAIN_DIR)
    val_ds = carregar(VAL_DIR)
    test_ds = carregar(TEST_DIR)

    class_names = train_ds.class_names
    print(f"[INFO] Total de marcas (classes) detectadas ({len(class_names)}): {class_names}")

    # Salva os nomes das classes para uso posterior na predição de imagens avulsas
    with open(CLASSES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    # Cache em uint8 (~4x menos RAM que float32), shuffle por imagem a cada época e prefetch
    autotune = tf.data.AUTOTUNE

    def para_uint8(img, rotulo):
        return tf.cast(img, tf.uint8), rotulo

    train_ds = (
        train_ds.map(para_uint8, num_parallel_calls=autotune)
        .cache()
        .shuffle(4096, seed=SEED, reshuffle_each_iteration=True)
        .batch(batch_size)
        .prefetch(autotune)
    )
    val_ds = val_ds.map(para_uint8, num_parallel_calls=autotune).cache().batch(batch_size).prefetch(autotune)
    test_ds = test_ds.map(para_uint8, num_parallel_calls=autotune).cache().batch(batch_size).prefetch(autotune)

    return train_ds, val_ds, test_ds, class_names


def construir_modelo(
    num_classes: int,
    dropout_rate: float = 0.2,
    usar_transfer_learning: bool = True,
    backbone: str = BACKBONE_PADRAO,
) -> tf.keras.Model:
    """
    Constrói o modelo de classificação de marcas de automóveis.
    Destaca a variação da camada de Dropout logo após a camada Densa (Dense).
    O modelo é retornado sem compilar: a compilação acontece em cada fase do treino.
    """
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="imagem_entrada")

    # Data Augmentation: ângulo, distância e enquadramento (flip, rotação, zoom, translação)
    # e brilho/contraste para simular reflexos e iluminação de rua.
    # Áreas novas criadas pelas transformações são preenchidas com branco, como o fundo do dataset.
    x = layers.RandomFlip("horizontal", name="aug_flip")(inputs)
    x = layers.RandomRotation(0.08, fill_mode="constant", fill_value=COR_FUNDO, name="aug_rotation")(x)
    x = layers.RandomZoom(0.15, fill_mode="constant", fill_value=COR_FUNDO, name="aug_zoom")(x)
    x = layers.RandomTranslation(0.1, 0.1, fill_mode="constant", fill_value=COR_FUNDO, name="aug_translation")(x)
    x = layers.RandomBrightness(0.2, name="aug_brightness")(x)
    x = layers.RandomContrast(0.2, name="aug_contrast")(x)

    if usar_transfer_learning:
        construtor, preprocessar = BACKBONES[backbone]
        if preprocessar is not None:
            x = preprocessar(x)
        # Os backbones EfficientNetV2/ConvNeXt já normalizam a entrada (0-255) internamente
        base_model = construtor(
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
            include_top=False,
            weights="imagenet",
        )
        base_model.name = "backbone"
        base_model.trainable = False  # Fase 1: pesos convolucionais congelados
        # training=False mantém as BatchNorm em modo de inferência também no fine-tuning
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    else:
        # Arquitetura CNN Convolucional clássica (treinada do zero)
        x = layers.Rescaling(1.0 / 255.0, name="normalizacao")(x)
        x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
        x = layers.GlobalAveragePooling2D()(x)

    # =========================================================================
    # CAMADA DENSA + EXPERIMENTO DE DROPOUT (0.0, 0.2, 0.5)
    # =========================================================================
    x = layers.Dense(256, activation="relu", name="camada_densa")(x)
    x = layers.Dropout(rate=dropout_rate, name=f"dropout_{str(dropout_rate).replace('.', '_')}")(x)

    # Saída em float32 para estabilidade numérica com mixed precision
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32", name="saida_marcas")(x)

    return models.Model(
        inputs=inputs,
        outputs=outputs,
        name=f"CarBrandClassifier_Dropout_{dropout_rate}",
    )


def compilar(model: tf.keras.Model, learning_rate: float):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_accuracy"),
        ],
    )


def descongelar_topo_backbone(model: tf.keras.Model, fracao: float = FRACAO_DESCONGELADA):
    """Libera para treino a fração final das camadas do backbone (exceto BatchNorm)."""
    base_model = model.get_layer("backbone")
    base_model.trainable = True
    corte = int(len(base_model.layers) * (1 - fracao))
    for i, camada in enumerate(base_model.layers):
        camada.trainable = i >= corte and not isinstance(camada, layers.BatchNormalization)
    liberadas = sum(1 for c in base_model.layers if c.trainable)
    print(f"[INFO] Fine-tuning: {liberadas} de {len(base_model.layers)} camadas do backbone liberadas.")


def concatenar_historicos(fase1: dict, fase2: dict) -> dict:
    historico = {
        chave: [float(v) for v in fase1[chave]] + [float(v) for v in fase2[chave]]
        for chave in fase1
        if chave in fase2
    }
    historico["inicio_fine_tuning"] = len(fase1["loss"])
    return historico


def plotar_curvas_loss_dropout(historicos: dict[float, dict], output_dir: Path):
    """
    Plota as curvas de perda (Loss de Treino e Validação) e Acurácia para cada
    taxa de Dropout (0.0, 0.2, 0.5), permitindo visualizar o Overfitting.
    A linha vertical pontilhada marca o início do fine-tuning de cada experimento.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    cores = {0.0: "#e74c3c", 0.2: "#2ecc71", 0.5: "#3498db"}

    for ax, metrica, nome in (
        (axes[0], "loss", "Loss"),
        (axes[1], "accuracy", "Acc"),
    ):
        for rate, hist in historicos.items():
            cor = cores.get(rate, None)
            epocas = range(1, len(hist[metrica]) + 1)
            ax.plot(
                epocas,
                hist[metrica],
                linestyle="--",
                color=cor,
                alpha=0.75,
                label=f"Treino {nome} (Dropout={rate})",
            )
            ax.plot(
                epocas,
                hist[f"val_{metrica}"],
                linestyle="-",
                linewidth=2.2,
                color=cor,
                label=f"Validação {nome} (Dropout={rate})",
            )
            if "inicio_fine_tuning" in hist:
                ax.axvline(hist["inicio_fine_tuning"] + 0.5, color=cor, linestyle=":", alpha=0.6)
        ax.set_xlabel("Época (linha pontilhada vertical = início do fine-tuning)")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)

    axes[0].set_title("Curvas de Perda (Loss) por Taxa de Dropout", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Perda (Sparse Categorical Crossentropy)")
    axes[1].set_title("Curvas de Acurácia por Taxa de Dropout", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Acurácia")

    plt.tight_layout()
    caminho_grafico = output_dir / "curvas_loss_dropout.png"
    plt.savefig(caminho_grafico, dpi=200)
    plt.close(fig)
    print(f"[INFO] Gráfico das curvas de Loss salvo em: {caminho_grafico.resolve()}")


def avaliar_e_gerar_matriz_confusao(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    class_names: list[str],
    output_dir: Path,
    descricao_modelo: str = "",
):
    """
    Avalia o modelo no conjunto de teste (top-1 e top-3), plota a Matriz de Confusão
    das marcas e salva exemplos de imagens classificadas incorretamente para análise
    de distância da foto e reflexos na lataria.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print("\n[INFO] Avaliando modelo no conjunto de teste...")

    y_true = []
    y_pred = []
    acertos_top3 = 0
    imagens_erro = []

    for batch_imgs, batch_labels in test_ds:
        preds = model.predict(batch_imgs, verbose=0)
        pred_classes = np.argmax(preds, axis=1)
        pred_confs = np.max(preds, axis=1)
        top3 = np.argsort(preds, axis=1)[:, -3:]

        rotulos = batch_labels.numpy()
        y_true.extend(rotulos)
        y_pred.extend(pred_classes)
        acertos_top3 += int(np.sum(np.any(top3 == rotulos[:, None], axis=1)))

        # Coleta até 12 exemplos de erros para inspeção visual (distância / reflexos)
        if len(imagens_erro) < 12:
            for img, real_idx, pred_idx, conf in zip(
                batch_imgs.numpy(), rotulos, pred_classes, pred_confs
            ):
                if real_idx != pred_idx and len(imagens_erro) < 12:
                    imagens_erro.append(
                        (
                            img.astype("uint8"),
                            class_names[real_idx],
                            class_names[pred_idx],
                            float(conf),
                        )
                    )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc_top1 = float(np.mean(y_true == y_pred))
    acc_top3 = acertos_top3 / len(y_true)
    print(f"[RESULTADO TESTE] Top-1: {acc_top1:.4f} | Top-3: {acc_top3:.4f} ({len(y_true)} imagens)")

    # Relatório de métricas por marca
    relatorio = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0
    )
    caminho_relatorio = output_dir / "relatorio_classificacao.txt"
    with open(caminho_relatorio, "w", encoding="utf-8") as f:
        f.write(f"RELATÓRIO DE CLASSIFICAÇÃO - {len(class_names)} MARCAS DE AUTOMÓVEIS\n")
        f.write("=" * 60 + "\n")
        if descricao_modelo:
            f.write(f"Modelo: {descricao_modelo}\n")
        f.write(f"Acurácia Top-1 (teste): {acc_top1:.4f}\n")
        f.write(f"Acurácia Top-3 (teste): {acc_top3:.4f}\n\n")
        f.write(relatorio)
    print(f"[INFO] Relatório de classificação salvo em: {caminho_relatorio.resolve()}")

    # Matriz de Confusão (N x N), normalizada por linha: % de cada marca real
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    cm_pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(cm_pct, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="% das imagens da marca real")

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=f"Matriz de Confusão (%) - Identificação de Marcas de Automóveis ({len(class_names)} Classes)",
        ylabel="Marca Real (Ground Truth)",
        xlabel="Marca Prevista pelo Modelo",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=10)
    plt.setp(ax.get_yticklabels(), fontsize=10)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                ax.text(
                    j,
                    i,
                    f"{cm_pct[i, j]:.0f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if cm_pct[i, j] > 50 else "black",
                )

    plt.tight_layout()
    caminho_cm = output_dir / "matriz_confusao.png"
    plt.savefig(caminho_cm, dpi=200)
    plt.close(fig)
    print(f"[INFO] Matriz de confusão salva em: {caminho_cm.resolve()}")

    # Painel de análise de erros (foco em distância da foto e reflexos na lataria)
    if imagens_erro:
        cols = 4
        rows = int(np.ceil(len(imagens_erro) / cols))
        fig_err, axes_err = plt.subplots(rows, cols, figsize=(15, 3.8 * rows))
        axes_err = np.array(axes_err).reshape(-1)

        for idx, (img, real_lbl, pred_lbl, conf) in enumerate(imagens_erro):
            axes_err[idx].imshow(img)
            axes_err[idx].set_title(
                f"Real: {real_lbl}\nPrevisto: {pred_lbl} ({conf * 100:.1f}%)",
                fontsize=9,
                color="#c0392b",
            )
            axes_err[idx].axis("off")

        for idx in range(len(imagens_erro), len(axes_err)):
            axes_err[idx].axis("off")

        fig_err.suptitle(
            "Exemplos de Erros de Classificação (Análise de Distância da Foto e Reflexos na Lataria)",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()
        caminho_erros = output_dir / "erros_distancia_e_reflexos.png"
        plt.savefig(caminho_erros, dpi=200)
        plt.close(fig_err)
        print(f"[INFO] Painel de análise de erros salvo em: {caminho_erros.resolve()}")


def treinar_experimento(
    rate: float,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    num_classes: int,
    epochs: int,
    epochs_ft: int,
    usar_transfer_learning: bool,
    backbone: str,
    caminho_checkpoint: Path,
) -> dict:
    """
    Treina um experimento de Dropout e salva o melhor modelo (menor val_loss entre
    todas as épocas das duas fases) em `caminho_checkpoint`. Retorna o histórico.
    """
    tf.keras.utils.set_random_seed(SEED)  # Mesma inicialização para comparação justa
    model = construir_modelo(
        num_classes=num_classes,
        dropout_rate=rate,
        usar_transfer_learning=usar_transfer_learning,
        backbone=backbone,
    )

    # O mesmo ModelCheckpoint nas duas fases guarda o melhor val_loss global
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        caminho_checkpoint, monitor="val_loss", save_best_only=True, verbose=1
    )

    def callbacks(paciencia: int):
        return [
            checkpoint,
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=paciencia, restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7, verbose=1
            ),
        ]

    print(f"\n[FASE 1] Treinando a cabeça (backbone congelado) - até {epochs} épocas")
    compilar(model, LR_FASE_CABECA)
    fase1 = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks(3), verbose=1)

    if not usar_transfer_learning or epochs_ft <= 0:
        historico = {chave: [float(v) for v in valores] for chave, valores in fase1.history.items()}
    else:
        print(f"\n[FASE 2] Fine-tuning do topo do backbone - até {epochs_ft} épocas")
        descongelar_topo_backbone(model)
        compilar(model, LR_FASE_FINE_TUNING)  # Recompilar é obrigatório para o trainable valer
        fase2 = model.fit(
            train_ds, validation_data=val_ds, epochs=epochs_ft, callbacks=callbacks(5), verbose=1
        )
        historico = concatenar_historicos(fase1.history, fase2.history)

    print(
        f"[RESULTADO Dropout={rate}] Menor Val Loss: {min(historico['val_loss']):.4f} | "
        f"Maior Val Acc: {max(historico['val_accuracy']):.4f}"
    )
    return historico


def executar_experimentos_treinamento(
    epochs: int = 10,
    epochs_ft: int = 20,
    usar_transfer_learning: bool = True,
    backbone: str = BACKBONE_PADRAO,
    checkpoint_dir: Path = CHECKPOINT_DIR,
):
    """
    Executa os 3 experimentos variando o Dropout na camada densa (0.0, 0.2 e 0.5),
    compara as curvas de perda (Loss) e salva o melhor modelo para inferência.

    Cada experimento concluído fica salvo em `checkpoint_dir` (modelo + histórico).
    Se o treino for interrompido (ex.: Colab desconectou), rodar de novo pula os
    experimentos já concluídos.
    """
    if tf.config.list_physical_devices("GPU"):
        # Mixed precision acelera bastante em GPUs como a T4 do Colab
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("[INFO] GPU detectada: mixed precision (float16) ativada.")

    train_ds, val_ds, test_ds, class_names = carregar_datasets()
    num_classes = len(class_names)

    tipo = backbone if usar_transfer_learning else "from_scratch"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    historicos: dict[float, dict] = {}
    for rate in DROPOUT_RATES:
        print("\n" + "=" * 70)
        print(f"[EXPERIMENTO] {tipo} com Dropout = {rate} na camada densa")
        print("=" * 70)

        caminho_modelo = checkpoint_dir / f"{tipo}_dropout_{rate}.keras"
        caminho_historico = checkpoint_dir / f"{tipo}_dropout_{rate}_historico.json"

        if caminho_modelo.exists() and caminho_historico.exists():
            print(f"[INFO] Experimento já concluído, carregando de: {caminho_historico.resolve()}")
            with open(caminho_historico, encoding="utf-8") as f:
                historicos[rate] = json.load(f)
            continue

        historicos[rate] = treinar_experimento(
            rate, train_ds, val_ds, num_classes, epochs, epochs_ft,
            usar_transfer_learning, backbone, caminho_modelo,
        )
        # O histórico só é gravado ao fim: sem ele, o experimento é refeito do zero
        with open(caminho_historico, "w", encoding="utf-8") as f:
            json.dump(historicos[rate], f, indent=2)
        tf.keras.backend.clear_session()

    # Salva os gráficos comparativos de Loss (Dropout 0.0 vs 0.2 vs 0.5)
    plotar_curvas_loss_dropout(historicos, OUTPUT_DIR)

    # O melhor experimento é o de menor val_loss; o checkpoint guarda a sua melhor época
    melhor_dropout = min(historicos, key=lambda r: min(historicos[r]["val_loss"]))
    melhor_modelo = tf.keras.models.load_model(checkpoint_dir / f"{tipo}_dropout_{melhor_dropout}.keras")
    melhor_modelo.save(MODEL_PATH)
    print(
        f"\n[SUCESSO] Melhor modelo ({tipo}, Dropout = {melhor_dropout}) salvo em: {MODEL_PATH.resolve()}"
    )
    avaliar_e_gerar_matriz_confusao(
        melhor_modelo, test_ds, class_names, OUTPUT_DIR,
        descricao_modelo=f"{tipo}, Dropout = {melhor_dropout}",
    )


def carregar_modelo_treinado() -> tuple[tf.keras.Model, list[str]]:
    if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
        raise FileNotFoundError(
            f"Modelo treinado ({MODEL_PATH}) ou arquivo de classes ({CLASSES_PATH}) não encontrado.\n"
            "Execute primeiro o treinamento com: python car-detection.py --mode train"
        )

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        class_names: list[str] = json.load(f)

    return tf.keras.models.load_model(MODEL_PATH), class_names


def preparar_imagem(img_path: Path) -> np.ndarray:
    """
    Prepara uma foto para o modelo do mesmo jeito que as imagens de treino:
    corrige a rotação do celular (EXIF) e completa a foto até ficar quadrada com
    fundo branco, sem esticar e sem cortar as laterais do carro.
    """
    with Image.open(img_path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img = ImageOps.pad(img, IMG_SIZE, color=(255, 255, 255))
        return np.asarray(img, dtype=np.float32)


def classificar(model: tf.keras.Model, class_names: list[str], img_path: Path) -> list[tuple[str, float]]:
    """Retorna as marcas ordenadas da mais para a menos provável, com a probabilidade."""
    probabilidades = model.predict(preparar_imagem(img_path)[None, ...], verbose=0)[0]
    indices_ordenados = np.argsort(probabilidades)[::-1]
    return [(class_names[int(i)], float(probabilidades[int(i)])) for i in indices_ordenados]


def prever_imagem(caminho_imagem: str, top_k: int = 3) -> str:
    """
    Recebe o caminho de uma imagem de veículo/emblema, carrega o modelo treinado
    e retorna o rótulo da marca identificada.
    """
    img_path = Path(caminho_imagem)
    if not img_path.exists():
        raise FileNotFoundError(f"Imagem não encontrada: {img_path.resolve()}")

    model, class_names = carregar_modelo_treinado()
    ranking = classificar(model, class_names, img_path)
    rotulo_previsto, confianca = ranking[0]

    print("\n" + "=" * 50)
    print(f"Imagem analisada : {img_path.name}")
    print(f"Marca prevista   : {rotulo_previsto} ({confianca * 100:.2f}% de confiança)")
    print("-" * 50)
    print(f"Top-{top_k} probabilidades:")
    for pos, (marca, prob) in enumerate(ranking[:top_k], start=1):
        print(f"  {pos}. {marca:<18} -> {prob * 100:.2f}%")
    print("=" * 50)

    return rotulo_previsto


def prever_pasta(caminho_pasta: str, top_k: int = 3) -> Path:
    """
    Classifica todas as imagens de uma pasta (ex.: as 10 fotos próprias) e salva
    uma tabela CSV em `resultados_experimentos/fotos_proprias.csv`.

    Se o nome do arquivo começar com o nome da marca (ex.: `Volkswagen_1.jpg`),
    a coluna "real" e o acerto são preenchidos automaticamente. A coluna
    "observacao" fica em branco para anotar distância, reflexos etc.
    """
    pasta = Path(caminho_pasta)
    imagens = sorted(
        p for p in pasta.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not imagens:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em: {pasta.resolve()}")

    model, class_names = carregar_modelo_treinado()
    marcas_por_tamanho = sorted(class_names, key=len, reverse=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    caminho_csv = OUTPUT_DIR / "fotos_proprias.csv"
    acertos = avaliadas = 0
    with open(caminho_csv, "w", newline="", encoding="utf-8-sig") as f:
        escritor = csv.writer(f, delimiter=";")
        escritor.writerow(["foto", "real", "previsto", "confianca_%", f"top{top_k}", "acertou", "observacao"])
        for img_path in imagens:
            ranking = classificar(model, class_names, img_path)
            previsto, confianca = ranking[0]
            real = next(
                (m for m in marcas_por_tamanho if img_path.stem.lower().startswith(m.lower())), ""
            )
            acertou = "" if not real else ("sim" if real == previsto else "não")
            if real:
                avaliadas += 1
                acertos += real == previsto
            top = ", ".join(f"{m} ({p * 100:.0f}%)" for m, p in ranking[:top_k])
            escritor.writerow([img_path.name, real, previsto, f"{confianca * 100:.1f}", top, acertou, ""])
            print(f"{img_path.name:<30} -> {previsto:<15} ({confianca * 100:5.1f}%)  {acertou}")

    if avaliadas:
        print(f"\n[RESULTADO] Acertos: {acertos}/{avaliadas}")
    print(f"[INFO] Tabela salva em: {caminho_csv.resolve()}")
    return caminho_csv


def main():
    parser = argparse.ArgumentParser(
        description="Classificador de Marcas de Automóveis por Imagem (Experimentos de Dropout e Predição)"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "predict"],
        default="predict",
        help="Modo de execução: 'train' para rodar os experimentos (Dropout 0.0, 0.2, 0.5) ou 'predict' para classificar uma imagem.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Caminho da imagem do automóvel/emblema para identificar a marca.",
    )
    parser.add_argument(
        "--pasta",
        type=str,
        default=None,
        help="Pasta com várias imagens: classifica todas e gera resultados_experimentos/fotos_proprias.csv.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Máximo de épocas da fase 1 (só a cabeça) em cada experimento (padrão: 10).",
    )
    parser.add_argument(
        "--epochs-ft",
        type=int,
        default=20,
        help="Máximo de épocas da fase 2 (fine-tuning do backbone); 0 desativa (padrão: 20).",
    )
    parser.add_argument(
        "--backbone",
        choices=sorted(BACKBONES),
        default=BACKBONE_PADRAO,
        help=f"Rede pré-treinada usada como extrator de características (padrão: {BACKBONE_PADRAO}).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=CHECKPOINT_DIR,
        help="Pasta dos checkpoints de cada experimento; no Colab, use uma pasta do Drive (padrão: checkpoints).",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Se informado, treina uma CNN do zero em vez de usar Transfer Learning (só a fase 1).",
    )

    args = parser.parse_args()

    # Se o usuário passar --image ou --pasta, executa diretamente a inferência
    if args.image:
        rotulo = prever_imagem(args.image)
        print(f"Rótulo de saída: {rotulo}")
        return
    if args.pasta:
        prever_pasta(args.pasta)
        return

    if args.mode == "train":
        executar_experimentos_treinamento(
            epochs=args.epochs,
            epochs_ft=args.epochs_ft,
            usar_transfer_learning=not args.from_scratch,
            backbone=args.backbone,
            checkpoint_dir=args.checkpoint_dir,
        )
    else:
        parser.print_help()
        print(
            "\n[DICA] Para treinar o modelo e rodar os experimentos de Dropout (0.0, 0.2, 0.5):\n"
            "  python car-detection.py --mode train\n\n"
            "Para prever a marca de um carro a partir de uma imagem (após treinado):\n"
            '  python car-detection.py --image "caminho/da/foto.jpg"\n'
            "  python car-detection.py --pasta fotos_proprias"
        )


if __name__ == "__main__":
    main()
