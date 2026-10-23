"""
Sistema de Detecção e Classificação de Marcas de Automóveis por Imagem
======================================================================

Funcionalidades:
1. Treinamento e Experimentos (--mode train):
   - Carrega todas as 33 marcas do dataset em `dataset/Car Brand Classification Dataset`
   - Executa experimentos variando a taxa de Dropout na camada densa: [0.0, 0.2, 0.5]
   - Gera e salva o gráfico comparativo das curvas de perda (Loss de Treino vs. Validação)
   - Gera e salva a Matriz de Confusão no conjunto de teste e um painel com erros de predição
     (útil para analisar erros por distância da foto ou reflexos na lataria/emblema)
   - Salva o melhor modelo treinado (`car_brand_model.keras`) e os rótulos (`class_names.json`)

2. Inferência / Predição (--image caminho/da/imagem.jpg):
   - Carrega o modelo treinado, recebe uma imagem como entrada e retorna o rótulo da marca.

Exemplos de uso:
   python car-detection.py --mode train --epochs 15
   python car-detection.py --image "dataset/Car Brand Classification Dataset/test/Volkswagen/1.jpg"
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models

# Configurações padrão de diretório e hiperparâmetros
DATASET_DIR = Path("dataset") / "Car Brand Classification Dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"

MODEL_PATH = Path("car_brand_model.keras")
CLASSES_PATH = Path("class_names.json")
OUTPUT_DIR = Path("resultados_experimentos")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DROPOUT_RATES = [0.0, 0.2, 0.5]
SEED = 42


def carregar_datasets(img_size: tuple[int, int] = IMG_SIZE, batch_size: int = BATCH_SIZE):
    """Carrega os conjuntos de treino, validação e teste com todas as 33 marcas."""
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Diretório de treino não encontrado em: {TRAIN_DIR.resolve()}"
        )

    print(f"[INFO] Carregando dataset a partir de: {DATASET_DIR.resolve()}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        seed=SEED,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        seed=SEED,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        seed=SEED,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,
    )

    class_names = train_ds.class_names
    print(f"[INFO] Total de marcas (classes) detectadas ({len(class_names)}): {class_names}")

    # Salva os nomes das classes para uso posterior na predição de imagens avulsas
    with open(CLASSES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    # Otimização de pipeline de I/O com cache e prefetch
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=autotune)
    val_ds = val_ds.prefetch(buffer_size=autotune)
    test_ds = test_ds.prefetch(buffer_size=autotune)

    return train_ds, val_ds, test_ds, class_names


def construir_modelo(
    num_classes: int,
    dropout_rate: float = 0.2,
    usar_transfer_learning: bool = True,
) -> tf.keras.Model:
    """
    Constrói o modelo de classificação de marcas de automóveis.
    Destaca a variação da camada de Dropout logo após a camada Densa (Dense).
    """
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="imagem_entrada")

    # Data Augmentation leve para lidar com variações de ângulo, distância e brilho/reflexo
    x = layers.RandomFlip("horizontal", name="aug_flip")(inputs)
    x = layers.RandomRotation(0.08, name="aug_rotation")(x)
    x = layers.RandomZoom(0.1, name="aug_zoom")(x)

    if usar_transfer_learning:
        # Extrator de características pré-treinado (MobileNetV2) ideal para 33 classes
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
            include_top=False,
            weights="imagenet",
        )
        base_model.trainable = False  # Congela os pesos convolucionais base
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

    outputs = layers.Dense(num_classes, activation="softmax", name="saida_marcas")(x)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name=f"CarBrandClassifier_Dropout_{dropout_rate}",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plotar_curvas_loss_dropout(historicos: dict[float, dict], output_dir: Path):
    """
    Plota as curvas de perda (Loss de Treino e Validação) e Acurácia para cada
    taxa de Dropout (0.0, 0.2, 0.5), permitindo visualizar o Overfitting.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    cores = {0.0: "#e74c3c", 0.2: "#2ecc71", 0.5: "#3498db"}

    # Subplot 1: Curvas de Perda (Loss) - Foco principal do experimento
    ax_loss = axes[0]
    for rate, hist in historicos.items():
        cor = cores.get(rate, None)
        epocas = range(1, len(hist["loss"]) + 1)
        ax_loss.plot(
            epocas,
            hist["loss"],
            linestyle="--",
            color=cor,
            alpha=0.75,
            label=f"Treino Loss (Dropout={rate})",
        )
        ax_loss.plot(
            epocas,
            hist["val_loss"],
            linestyle="-",
            linewidth=2.2,
            color=cor,
            label=f"Validação Loss (Dropout={rate})",
        )

    ax_loss.set_title("Curvas de Perda (Loss) por Taxa de Dropout", fontsize=13, fontweight="bold")
    ax_loss.set_xlabel("Época")
    ax_loss.set_ylabel("Perda (Sparse Categorical Crossentropy)")
    ax_loss.legend()
    ax_loss.grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: Curvas de Acurácia
    ax_acc = axes[1]
    for rate, hist in historicos.items():
        cor = cores.get(rate, None)
        epocas = range(1, len(hist["accuracy"]) + 1)
        ax_acc.plot(
            epocas,
            hist["accuracy"],
            linestyle="--",
            color=cor,
            alpha=0.75,
            label=f"Treino Acc (Dropout={rate})",
        )
        ax_acc.plot(
            epocas,
            hist["val_accuracy"],
            linestyle="-",
            linewidth=2.2,
            color=cor,
            label=f"Validação Acc (Dropout={rate})",
        )

    ax_acc.set_title("Curvas de Acurácia por Taxa de Dropout", fontsize=13, fontweight="bold")
    ax_acc.set_xlabel("Época")
    ax_acc.set_ylabel("Acurácia")
    ax_acc.legend()
    ax_acc.grid(True, linestyle=":", alpha=0.6)

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
):
    """
    Avalia o modelo no conjunto de teste, plota a Matriz de Confusão das 33 marcas
    e salva exemplos de imagens classificadas incorretamente para análise de
    distância da foto e reflexos na lataria.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print("\n[INFO] Avaliando modelo no conjunto de teste...")

    y_true = []
    y_pred = []
    imagens_erro = []

    for batch_imgs, batch_labels in test_ds:
        preds = model.predict(batch_imgs, verbose=0)
        pred_classes = np.argmax(preds, axis=1)
        pred_confs = np.max(preds, axis=1)

        y_true.extend(batch_labels.numpy())
        y_pred.extend(pred_classes)

        # Coleta até 12 exemplos de erros para inspeção visual (distância / reflexos)
        if len(imagens_erro) < 12:
            for img, real_idx, pred_idx, conf in zip(
                batch_imgs.numpy(), batch_labels.numpy(), pred_classes, pred_confs
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

    # Relatório de métricas por marca
    relatorio = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0
    )
    caminho_relatorio = output_dir / "relatorio_classificacao.txt"
    with open(caminho_relatorio, "w", encoding="utf-8") as f:
        f.write("RELATÓRIO DE CLASSIFICAÇÃO - 33 MARCAS DE AUTOMÓVEIS\n")
        f.write("=" * 60 + "\n\n")
        f.write(relatorio)
    print(f"[INFO] Relatório de classificação salvo em: {caminho_relatorio.resolve()}")

    # Matriz de Confusão (33 x 33)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(18, 16))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Matriz de Confusão - Identificação de Marcas de Automóveis (33 Classes)",
        ylabel="Marca Real (Ground Truth)",
        xlabel="Marca Prevista pelo Modelo",
    )
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)

    limiar = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                ax.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if cm[i, j] > limiar else "black",
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


def executar_experimentos_treinamento(
    epochs: int = 15,
    usar_transfer_learning: bool = True,
):
    """
    Executa os 3 experimentos variando o Dropout na camada densa (0.0, 0.2 e 0.5),
    compara as curvas de perda (Loss) e salva o melhor modelo para inferência.
    """
    train_ds, val_ds, test_ds, class_names = carregar_datasets()
    num_classes = len(class_names)

    historicos: dict[float, dict] = {}
    melhor_val_loss = float("inf")
    melhor_modelo = None
    melhor_dropout = None

    for rate in DROPOUT_RATES:
        print("\n" + "=" * 70)
        print(f"[EXPERIMENTO] Treinando modelo com Dropout = {rate} na camada densa")
        print("=" * 70)

        model = construir_modelo(
            num_classes=num_classes,
            dropout_rate=rate,
            usar_transfer_learning=usar_transfer_learning,
        )

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            verbose=1,
        )

        historicos[rate] = history.history
        menor_val_loss_atual = min(history.history["val_loss"])
        print(
            f"[RESULTADO Dropout={rate}] Menor Val Loss: {menor_val_loss_atual:.4f} | "
            f"Maior Val Acc: {max(history.history['val_accuracy']):.4f}"
        )

        if menor_val_loss_atual < melhor_val_loss:
            melhor_val_loss = menor_val_loss_atual
            melhor_modelo = model
            melhor_dropout = rate

    # Salva os gráficos comparativos de Loss (Dropout 0.0 vs 0.2 vs 0.5)
    plotar_curvas_loss_dropout(historicos, OUTPUT_DIR)

    # Salva o melhor modelo para predição futura
    if melhor_modelo is not None:
        melhor_modelo.save(MODEL_PATH)
        print(
            f"\n[SUCESSO] Melhor modelo (Dropout = {melhor_dropout}) salvo em: {MODEL_PATH.resolve()}"
        )
        avaliar_e_gerar_matriz_confusao(melhor_modelo, test_ds, class_names, OUTPUT_DIR)


def prever_imagem(caminho_imagem: str, top_k: int = 3) -> str:
    """
    Recebe o caminho de uma imagem de veículo/emblema, carrega o modelo treinado
    e retorna o rótulo da marca identificada.
    """
    img_path = Path(caminho_imagem)
    if not img_path.exists():
        raise FileNotFoundError(f"Imagem não encontrada: {img_path.resolve()}")

    if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
        raise FileNotFoundError(
            f"Modelo treinado ({MODEL_PATH}) ou arquivo de classes ({CLASSES_PATH}) não encontrado.\n"
            "Execute primeiro o treinamento com: python car-detection.py --mode train"
        )

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        class_names: list[str] = json.load(f)

    model = tf.keras.models.load_model(MODEL_PATH)

    # Carrega e prepara a imagem na resolução esperada pelo modelo (224x224)
    img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_batch = tf.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)

    probabilidades = model.predict(img_batch, verbose=0)[0]
    indices_ordenados = np.argsort(probabilidades)[::-1]

    indice_vencedor = int(indices_ordenados[0])
    rotulo_previsto = class_names[indice_vencedor]
    confianca = float(probabilidades[indice_vencedor]) * 100.0

    print("\n" + "=" * 50)
    print(f"Imagem analisada : {img_path.name}")
    print(f"Marca prevista   : {rotulo_previsto} ({confianca:.2f}% de confiança)")
    print("-" * 50)
    print(f"Top-{top_k} probabilidades:")
    for pos, idx in enumerate(indices_ordenados[:top_k], start=1):
        print(f"  {pos}. {class_names[int(idx)]:<18} -> {probabilidades[int(idx)] * 100:.2f}%")
    print("=" * 50)

    return rotulo_previsto


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
        "--epochs",
        type=int,
        default=15,
        help="Número de épocas para cada experimento de Dropout no treinamento (padrão: 15).",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Se informado, treina uma CNN do zero em vez de usar Transfer Learning (MobileNetV2).",
    )

    args = parser.parse_args()

    # Se o usuário passar --image, executa diretamente a inferência retornando o rótulo
    if args.image:
        rotulo = prever_imagem(args.image)
        print(f"Rótulo de saída: {rotulo}")
        return

    if args.mode == "train":
        executar_experimentos_treinamento(
            epochs=args.epochs,
            usar_transfer_learning=not args.from_scratch,
        )
    else:
        parser.print_help()
        print(
            "\n[DICA] Para treinar o modelo e rodar os experimentos de Dropout (0.0, 0.2, 0.5):\n"
            "  python car-detection.py --mode train\n\n"
            "Para prever a marca de um carro a partir de uma imagem (após treinado):\n"
            '  python car-detection.py --image "dataset/Car Brand Classification Dataset/test/Volkswagen/1.jpg"'
        )


if __name__ == "__main__":
    main()
