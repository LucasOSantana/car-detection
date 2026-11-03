"""
Preparação do Dataset (DVM-CAR - Confirmed Fronts)
==================================================

Gera `dataset/marcas/{train,val,test}/<Marca>/` a partir do arquivo
`dataset/Confirmed_fronts.zip` (subconjunto de dianteiras verificadas do DVM-CAR):
   https://figshare.com/articles/figure/DVM-CAR_Dataset/19586296/1

Etapas:
1. Seleciona as 15 marcas do projeto (MARCAS).
2. Remove duplicatas: imagens idênticas que aparecem com marcas diferentes são
   descartadas (rótulo ambíguo); repetidas dentro da mesma marca ficam com 1 cópia.
3. Divide em treino/validação/teste (70/15/15) por grupo "modelo + ano", para que
   fotos quase iguais do mesmo carro não caiam em conjuntos diferentes.
4. Limita cada marca a LIMITE_POR_MARCA imagens, mantendo a proporção 70/15/15.
5. Compacta o resultado em `dataset_marcas.zip` para envio ao Google Drive (Colab).

Uso:
   python preparar_dataset.py
"""

import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

ZIP_ORIGEM = Path("dataset") / "Confirmed_fronts.zip"
DESTINO = Path("dataset") / "marcas"
ZIP_COLAB = Path("dataset_marcas.zip")

MARCAS = [
    "Audi", "BMW", "Citroen", "Fiat", "Ford",
    "Honda", "Hyundai", "Jeep", "Kia", "Mercedes-Benz",
    "Mitsubishi", "Nissan", "Peugeot", "Toyota", "Volkswagen",
]
LIMITE_POR_MARCA = 1500
PROPORCOES = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42


def listar_imagens(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    """Retorna as imagens do zip, já sem as duplicatas e sem as ambíguas."""
    por_conteudo: dict[tuple[int, int], list[zipfile.ZipInfo]] = defaultdict(list)
    for info in zf.infolist():
        if info.filename.endswith(".jpg"):
            por_conteudo[(info.CRC, info.file_size)].append(info)

    imagens = []
    descartadas = 0
    for grupo in por_conteudo.values():
        marcas_no_grupo = {info.filename.split("/")[1] for info in grupo}
        if len(marcas_no_grupo) > 1:
            descartadas += len(grupo)
            continue
        imagens.append(grupo[0])
    print(f"[INFO] {descartadas} imagens com rótulo ambíguo (mesma foto em marcas diferentes) descartadas.")
    return imagens


def dividir_por_grupo(imagens: list[zipfile.ZipInfo], rng: random.Random) -> dict[str, list[zipfile.ZipInfo]]:
    """Divide as imagens de uma marca em train/val/test sem separar um mesmo modelo+ano."""
    grupos: dict[tuple[str, str], list[zipfile.ZipInfo]] = defaultdict(list)
    for info in imagens:
        # Nome do arquivo: Marca$$Modelo$$Ano$$Cor$$IdModelo$$IdAnuncio$$image_N.jpg
        partes = info.filename.split("/")[-1].split("$$")
        grupos[(partes[1], partes[2])].append(info)

    # Embaralha e ordena do maior para o menor grupo: os grandes são distribuídos
    # primeiro e os pequenos completam as cotas com mais precisão
    chaves = sorted(grupos)
    rng.shuffle(chaves)
    chaves.sort(key=lambda chave: len(grupos[chave]), reverse=True)

    total = len(imagens)
    divisao: dict[str, list[zipfile.ZipInfo]] = {split: [] for split in PROPORCOES}
    # Cada grupo vai para o conjunto que está mais abaixo da sua cota
    for chave in chaves:
        split = min(PROPORCOES, key=lambda s: len(divisao[s]) / (PROPORCOES[s] * total))
        divisao[split].extend(grupos[chave])
    return divisao


def main():
    if not ZIP_ORIGEM.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {ZIP_ORIGEM.resolve()}\n"
            "Baixe em https://ndownloader.figshare.com/files/34792480 e salve em dataset/."
        )

    rng = random.Random(SEED)
    if DESTINO.exists():
        shutil.rmtree(DESTINO)

    with zipfile.ZipFile(ZIP_ORIGEM) as zf:
        por_marca: dict[str, list[zipfile.ZipInfo]] = defaultdict(list)
        for info in listar_imagens(zf):
            marca = info.filename.split("/")[1]
            if marca in MARCAS:
                por_marca[marca].append(info)

        faltando = set(MARCAS) - set(por_marca)
        if faltando:
            raise ValueError(f"Marcas não encontradas no zip: {sorted(faltando)}")

        print(f"\n{'Marca':<15}{'train':>7}{'val':>7}{'test':>7}")
        for marca in MARCAS:
            divisao = dividir_por_grupo(por_marca[marca], rng)
            contagens = []
            for split, imagens in divisao.items():
                limite = int(LIMITE_POR_MARCA * PROPORCOES[split])
                if len(imagens) > limite:
                    imagens = rng.sample(imagens, limite)
                pasta = DESTINO / split / marca
                pasta.mkdir(parents=True, exist_ok=True)
                for info in imagens:
                    nome = info.filename.split("/")[-1].replace("$$", "_")
                    (pasta / nome).write_bytes(zf.read(info))
                contagens.append(len(imagens))
            print(f"{marca:<15}" + "".join(f"{c:>7}" for c in contagens))

    print(f"\n[INFO] Dataset gerado em: {DESTINO.resolve()}")

    with zipfile.ZipFile(ZIP_COLAB, "w", zipfile.ZIP_STORED) as zf:
        for arquivo in sorted(DESTINO.rglob("*.jpg")):
            zf.write(arquivo, arquivo.as_posix())
    print(f"[INFO] Zip para o Colab gerado em: {ZIP_COLAB.resolve()}")


if __name__ == "__main__":
    main()
