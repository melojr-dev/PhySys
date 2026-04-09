import torch
import torch.nn as nn

# 1. Definição da Arquitetura (CNN 1D)
class ModeloFisiorai(nn.Module):
    def __init__(self):
        super(ModeloFisiorai, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16, 1) # Saída única (probabilidade de fluidez)

    def forward(self, x):
        # x formato: (Batch, Channels, Seq_Len)
        x = x.transpose(1, 2) 
        x = self.conv1(x)
        x = self.relu(x)
        x = torch.mean(x, dim=2) # Global Average Pooling
        x = self.fc(x)
        return x

# 2. Gerar e salvar um modelo "virgem" para teste
modelo = ModeloFisiorai()
# Convertendo para TorchScript (O formato que o seu engine.py já espera)
modelo_scripted = torch.jit.script(modelo)
modelo_scripted.save("src/models/modelo_pibic.pt")

print("✅ Modelo base gerado em src/models/modelo_pibic.pt")