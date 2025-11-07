import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from osgeo import gdal
from modelo_unet import UNet
from skimage.morphology import remove_small_objects

# ==============================================================================
# CONFIGURAÇÃO DO SCRIPT
# ==============================================================================
dataset_folder = r'C:\ortofoto02\dataset_unet'
model_filename = 'modelo_treinado.pth'
model_path = os.path.join(dataset_folder, model_filename)
output_folder = os.path.join(dataset_folder, 'mascaras_previstas')
os.makedirs(output_folder, exist_ok=True)
TILE_SIZE = 512
THRESHOLD = 0.4 # Ajustado o limiar de binarização
MIN_SIZE_POST_PROCESSING = 50 # Tamanho mínimo para remover pequenos objetos

print(f"DEBUG: THRESHOLD={THRESHOLD}, MIN_SIZE_POST_PROCESSING={MIN_SIZE_POST_PROCESSING}")

# ==============================================================================
# 1. DEFINIÇÃO DA CLASSE DATASET E CARREGAMENTO DOS DADOS DE TESTE
# ==============================================================================
class TileDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_filenames = sorted(os.listdir(img_dir))
        self.img_filenames = [f for f in self.img_filenames if not f.endswith('.xml')]

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        img_filename = self.img_filenames[idx]
        img_path = os.path.join(self.img_dir, img_filename)
        mask_path = os.path.join(self.mask_dir, img_filename)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image_np, mask_np = np.array(image), np.array(mask)
        h, w, _ = image_np.shape
        pad_h, pad_w = TILE_SIZE - h, TILE_SIZE - w
        
        if pad_h > 0 or pad_w > 0:
            image_np = np.pad(image_np, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
            mask_np = np.pad(mask_np, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            
        image, mask = Image.fromarray(image_np), Image.fromarray(mask_np)
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            
            mask = (mask > 0.5).float()
            
        return image, mask

data_transform = transforms.ToTensor()
test_dataset = TileDataset(
    img_dir=os.path.join(dataset_folder, 'teste', 'imagens'),
    mask_dir=os.path.join(dataset_folder, 'teste', 'mascaras'),
    transform=data_transform
)
test_loader = DataLoader(test_dataset, batch_size=1)

# ==============================================================================
# 2. AVALIAÇÃO DO MODELO
# ==============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=3, out_channels=1).to(device)

print(f"Carregando '{model_filename}' para avaliação...")
if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    print(f"Erro: O arquivo de modelo '{model_filename}' não foi encontrado. Por favor, rode o script 'treinar_modelo.py' primeiro.")
    exit()

print("\nIniciando a avaliação no conjunto de teste...")

total_tp = 0
total_fp = 0
total_fn = 0

model.eval()
with torch.no_grad():
    for i, (images, masks) in enumerate(test_loader):
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        
        # APLICAÇÃO DA SIGMOID ANTES DO THRESHOLDING
        predicted_masks = (torch.sigmoid(outputs) > THRESHOLD).float()
        
        # Pós-processamento: remover pequenos objetos
        predicted_masks_np = predicted_masks.squeeze(0).cpu().numpy()[0]
        predicted_masks_np = remove_small_objects(predicted_masks_np.astype(bool), min_size=MIN_SIZE_POST_PROCESSING).astype(np.float32)
        predicted_masks = torch.from_numpy(predicted_masks_np).unsqueeze(0).unsqueeze(0).to(device)

        original_img_path = os.path.join(test_dataset.img_dir, test_dataset.img_filenames[i])
        
        if not os.path.exists(original_img_path):
            print(f"Aviso: Imagem original não encontrada em {original_img_path}. Pulando a gravação da máscara prevista para esta amostra.")
            continue

        original_img_ds = gdal.Open(original_img_path)
        
        if original_img_ds is None:
            print(f"Aviso: Não foi possível abrir a imagem original {original_img_path} com GDAL. Pulando a gravação da máscara prevista para esta amostra.")
            continue

        output_mask_np = predicted_masks.squeeze(0).cpu().numpy()[0]
        
        output_mask_np = (output_mask_np * 255).astype(np.uint8)

        output_filename = test_dataset.img_filenames[i].replace('.tif', '_predicted.tif')
        output_path = os.path.join(output_folder, output_filename)
        
        driver = gdal.GetDriverByName('GTiff')
        output_ds = driver.Create(output_path, TILE_SIZE, TILE_SIZE, 1, gdal.GDT_Byte)
        output_ds.SetGeoTransform(original_img_ds.GetGeoTransform())
        output_ds.SetProjection(original_img_ds.GetProjection())
        output_ds.GetRasterBand(1).WriteArray(output_mask_np)
        
        output_ds = None
        original_img_ds = None
        
        tp = (predicted_masks * masks).sum()
        fp = ((1 - masks) * predicted_masks).sum()
        fn = (masks * (1 - predicted_masks)).sum()
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        print(f"  > Processando imagem {i+1}/{len(test_dataset)}...")

iou = total_tp / (total_tp + total_fp + total_fn + 1e-6)
dice = (2 * total_tp) / (2 * total_tp + total_fp + total_fn + 1e-6)
precision = total_tp / (total_tp + total_fp + 1e-6) # Cálculo da precisão

results_file = os.path.join(dataset_folder, 'resultados_avaliacao.txt')
with open(results_file, 'w') as f:
    f.write(f"--- Resultados da Avaliação ---\n")
    f.write(f"Número total de tiles de teste: {len(test_dataset)}\n")
    f.write(f"IoU (Intersection over Union): {iou.item():.4f}\n")
    f.write(f"Coeficiente de Dice: {dice.item():.4f}\n")
    f.write(f"Precisão: {precision.item():.4f}\n") # Adicionando precisão ao arquivo

print(f"--- Resultados da Avaliação ---\n")
print(f"IoU (Intersection over Union): {iou.item():.4f}\n")
print(f"Coeficiente de Dice: {dice.item():.4f}\n")
print(f"Precisão: {precision.item():.4f}\n") # Adicionando precisão ao console

print(f"\nResultados da avaliação salvos em: {results_file}")
print(f"As máscaras previstas foram salvas em: {output_folder}")



