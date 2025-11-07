import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from modelo_unet import UNet

# ==============================================================================
# CONFIGURAÇÃO DO SCRIPT
# ==============================================================================
dataset_folder = r'C:\ortofoto02\dataset_unet' # Ajuste o caminho conforme necessário
model_filename = 'modelo_treinado.pth'
model_path = os.path.join(dataset_folder, model_filename)

NUM_EPOCHS = 100 # Aumentado o número de épocas para um treinamento mais robusto
BATCH_SIZE = 8 # Aumentado o batch size para potencialmente acelerar o treinamento e ter gradientes mais estáveis
LEARNING_RATE = 0.00005 # Reduzido ainda mais o learning rate para um treinamento mais fino
TILE_SIZE = 512

# Early Stopping Parameters
PATIENCE = 10 # Número de épocas para esperar por melhoria antes de parar
MIN_DELTA = 0.0001 # Mudança mínima na perda para ser considerada uma melhoria

# ==============================================================================
# 1. FUNÇÃO DE PERDA (BCELOSS COM LOGITS PARA ESTABILIDADE NUMÉRICA)
# ==============================================================================
# Usamos BCEWithLogitsLoss para combinar sigmoid e BCELoss em uma única etapa,
# o que é mais numericamente estável e evita problemas de underflow/overflow.
criterion = nn.BCEWithLogitsLoss()

# ==============================================================================
# 2. DEFINIÇÃO DA CLASSE DATASET E CARREGAMENTO DOS DADOS
# ==============================================================================
class TileDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_filenames = sorted(os.listdir(img_dir))

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
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            mask = self.transform(mask)
            
            # Assegura que a máscara é binária (0 ou 1) após a transformação
            # e que está no tipo float para a função de perda.
            mask = (mask > 0.5).float() # Converte para 0.0 ou 1.0
            
        return image, mask

# Definindo as transformações de aumento de dados
data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Adicionado ColorJitter
    transforms.ToTensor()
])

train_dataset = TileDataset(
    img_dir=os.path.join(dataset_folder, 'treino', 'imagens'),
    mask_dir=os.path.join(dataset_folder, 'treino', 'mascaras'),
    transform=data_transform
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ==============================================================================
# 3. DEFINIÇÃO DO MODELO, OTIMIZADOR E TREINAMENTO
# ==============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=3, out_channels=1).to(device)
# Usando Adam para um otimizador mais robusto
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Verificando se já existe um modelo treinado
start_epoch = 0
best_loss = float('inf')
epochs_no_improve = 0

if os.path.exists(model_path):
    print(f"Modelo encontrado em '{model_path}'. Carregando pesos para continuar o treinamento...")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    if 'best_loss' in checkpoint: # Carrega best_loss se disponível
        best_loss = checkpoint['best_loss']
    if 'epochs_no_improve' in checkpoint: # Carrega epochs_no_improve se disponível
        epochs_no_improve = checkpoint['epochs_no_improve']
    print(f"Reiniciando o treinamento a partir da Época {start_epoch}.")
else:
    print("Nenhum modelo treinado encontrado. Iniciando novo treinamento.")

print(f"\nIniciando o treinamento para um total de {NUM_EPOCHS} épocas.")

start_time = datetime.now()
print(f"Treinamento iniciado em: {start_time.strftime('%d/%m/%Y às %H:%M:%S')}")

for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for i, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        
        # A função de ativação sigmoid foi removida da camada final do modelo UNet
        # porque BCEWithLogitsLoss já aplica sigmoid internamente.
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Época [{epoch+1}/{NUM_EPOCHS}], Perda: {avg_loss:.4f}")

    # Early Stopping
    if avg_loss < best_loss - MIN_DELTA:
        best_loss = avg_loss
        epochs_no_improve = 0
        # Salva o modelo se houver melhoria
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
            'epochs_no_improve': epochs_no_improve,
        }, model_path)
        print(f"Melhoria na perda. Modelo salvo na Época {epoch+1}.")
    else:
        epochs_no_improve += 1
        print(f"Perda não melhorou por {epochs_no_improve} épocas.")
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1} as loss did not improve for {PATIENCE} consecutive epochs.")
            break

# ==============================================================================
# 4. SALVANDO O MODELO (se não foi salvo pelo Early Stopping na última época)
# ==============================================================================
# Este bloco só será executado se o loop terminar sem early stopping
# ou se o modelo não foi salvo na última época devido a não melhoria.
# No entanto, com o early stopping, o melhor modelo já estará salvo.
# Mantido para garantir que um modelo seja salvo no final, se necessário.
if epochs_no_improve < PATIENCE and epoch == NUM_EPOCHS - 1:
    torch.save({
        'epoch': NUM_EPOCHS - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
        'epochs_no_improve': epochs_no_improve,
    }, model_path)
    print(f"Modelo final salvo em: {model_path}")

end_time = datetime.now()
duration = end_time - start_time
print("\nTreinamento concluído!")
print(f"Treinamento finalizado em: {end_time.strftime('%d/%m/%Y às %H:%M:%S')}")
print(f"Duração total do treinamento: {duration}")
