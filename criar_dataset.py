import os
from osgeo import gdal
import numpy as np
import random
from sklearn.model_selection import train_test_split
import shutil

# ==============================================================================
# CONFIGURAÇÃO DO SCRIPT
# ==============================================================================
# Caminhos para os arquivos originais
ORIGINAL_IMAGEM_PATH = r'C:\ortofoto02\orto_quadra_treino\treino.tif'
ORIGINAL_MASCARA_PATH = r'C:\ortofoto02\mascaras_binarias_treino\treino.tif'

# Pasta de saída para os tiles
OUTPUT_DIR = r'C:\ortofoto02\dataset_unet'
TILE_SIZE = 512
TEST_SIZE = 0.2  # 20% para o conjunto de teste

# ==============================================================================
# INÍCIO DO PROCESSO DE CRIAÇÃO E DIVISÃO
# ==============================================================================
print("Iniciando a criação de tiles sincronizados...")

# Cria as pastas de saída temporárias
TEMP_IMAGENS_DIR = os.path.join(OUTPUT_DIR, 'temp_imagens')
TEMP_MASCARAS_DIR = os.path.join(OUTPUT_DIR, 'temp_mascaras')
os.makedirs(TEMP_IMAGENS_DIR, exist_ok=True)
os.makedirs(TEMP_MASCARAS_DIR, exist_ok=True)

# Abre os arquivos originais com GDAL
try:
    imagem_ds = gdal.Open(ORIGINAL_IMAGEM_PATH)
    mascara_ds = gdal.Open(ORIGINAL_MASCARA_PATH)
    if imagem_ds is None or mascara_ds is None:
        raise FileNotFoundError("Não foi possível abrir os arquivos originais. Verifique os caminhos.")
except Exception as e:
    print(f"Erro: {e}")
    exit()

# Obtém as dimensões dos arquivos
cols = imagem_ds.RasterXSize
rows = imagem_ds.RasterYSize
driver = gdal.GetDriverByName('GTiff')

print(f"Dimensões da Imagem Original: {cols}x{rows}")
print(f"Dimensões da Máscara Original: {mascara_ds.RasterXSize}x{mascara_ds.RasterYSize}")

total_tiles_criados = 0
tile_filenames = []

# Laço de corte usa a dimensão original
for y_offset in range(0, rows, TILE_SIZE):
    for x_offset in range(0, cols, TILE_SIZE):
        current_width = min(TILE_SIZE, cols - x_offset)
        current_height = min(TILE_SIZE, rows - y_offset)

        imagem_tile = imagem_ds.ReadAsArray(x_offset, y_offset, current_width, current_height)
        mascara_tile = mascara_ds.ReadAsArray(x_offset, y_offset, current_width, current_height)

        if imagem_tile is None or mascara_tile is None:
            continue

        if len(imagem_tile.shape) == 3:
            imagem_tile = np.transpose(imagem_tile, (1, 2, 0))

        if np.sum(imagem_tile) == 0:
            print(f"  > Pulando tile ({x_offset}, {y_offset}) - Imagem totalmente preta.")
            continue
        
        # Cria um novo array de 512x512 preenchido com zeros
        padded_imagem = np.zeros((TILE_SIZE, TILE_SIZE, imagem_tile.shape[2]), dtype=imagem_tile.dtype)
        padded_mascara = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)
        
        # Copia o tile lido para o novo array
        padded_imagem[0:current_height, 0:current_width, :] = imagem_tile
        
        # --- CORREÇÃO: CONVERSÃO DE 0/1 PARA 0/255 PARA QUE A MÁSCARA SEJA VISÍVEL ---
        padded_mascara[0:current_height, 0:current_width] = (mascara_tile * 255).astype(np.uint8)
        # -----------------------------------------------------------------------------
        
        filename = f'tile_{x_offset}_{y_offset}.tif'
        imagem_output_path = os.path.join(TEMP_IMAGENS_DIR, filename)
        mascara_output_path = os.path.join(TEMP_MASCARAS_DIR, filename)

        # Salva o tile de imagem
        imagem_output_ds = driver.Create(imagem_output_path, TILE_SIZE, TILE_SIZE, padded_imagem.shape[2], gdal.GDT_Byte)
        for i in range(padded_imagem.shape[2]):
            imagem_output_ds.GetRasterBand(i + 1).WriteArray(padded_imagem[:, :, i])
        imagem_output_ds = None
        
        # Salva o tile de máscara
        mascara_output_ds = driver.Create(mascara_output_path, TILE_SIZE, TILE_SIZE, 1, gdal.GDT_Byte)
        mascara_output_ds.GetRasterBand(1).WriteArray(padded_mascara)
        mascara_output_ds = None
        
        tile_filenames.append(filename)
        total_tiles_criados += 1
        print(f"  > Tiles ({x_offset}, {y_offset}) criados.")

print(f"\nFinalizado a criação. Total de {total_tiles_criados} tiles criados.")

# ==============================================================================
# DIVISÃO EM TREINO E TESTE
# ==============================================================================
print("\nDividindo tiles em conjuntos de treino e teste...")

train_files, test_files = train_test_split(tile_filenames, test_size=TEST_SIZE, random_state=42)

TRAIN_IMAGENS_DIR = os.path.join(OUTPUT_DIR, 'treino', 'imagens')
TRAIN_MASCARAS_DIR = os.path.join(OUTPUT_DIR, 'treino', 'mascaras')
TEST_IMAGENS_DIR = os.path.join(OUTPUT_DIR, 'teste', 'imagens')
TEST_MASCARAS_DIR = os.path.join(OUTPUT_DIR, 'teste', 'mascaras')

os.makedirs(TRAIN_IMAGENS_DIR, exist_ok=True)
os.makedirs(TRAIN_MASCARAS_DIR, exist_ok=True)
os.makedirs(TEST_IMAGENS_DIR, exist_ok=True)
os.makedirs(TEST_MASCARAS_DIR, exist_ok=True)

for filename in train_files:
    shutil.move(os.path.join(TEMP_IMAGENS_DIR, filename), TRAIN_IMAGENS_DIR)
    shutil.move(os.path.join(TEMP_MASCARAS_DIR, filename), TRAIN_MASCARAS_DIR)

for filename in test_files:
    shutil.move(os.path.join(TEMP_IMAGENS_DIR, filename), TEST_IMAGENS_DIR)
    shutil.move(os.path.join(TEMP_MASCARAS_DIR, filename), TEST_MASCARAS_DIR)

# Remove as pastas temporárias
shutil.rmtree(TEMP_IMAGENS_DIR)
shutil.rmtree(TEMP_MASCARAS_DIR)

print(f"Divisão concluída: {len(train_files)} tiles para treino e {len(test_files)} para teste.")
print(f"Dataset pronto em: {OUTPUT_DIR}")
