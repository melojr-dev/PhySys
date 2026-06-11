# PhySys — Deploy no Railway

## Estrutura do projeto

```
PhySys/
├── app.py
├── requirements.txt
├── Procfile
├── runtime.txt
├── railway.toml
├── .streamlit/
│   └── config.toml
└── src/
    ├── core/
    │   └── engine.py
    ├── models/
    │   └── modelo_pibic.pt
    └── utils/
        └── dataset_pibic.csv
```

## Como fazer o deploy

### 1. Suba o código no GitHub

```bash
git init
git add .
git commit -m "primeiro deploy"
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/physys.git
git push -u origin main
```

### 2. Deploy no Railway

1. Acesse [railway.app](https://railway.app) e faça login com GitHub
2. Clique em **New Project → Deploy from GitHub repo**
3. Selecione o repositório `physys`
4. Railway detecta automaticamente o `Procfile` e sobe o app
5. Vá em **Settings → Networking → Generate Domain** para obter a URL pública

### 3. Variáveis de ambiente (se necessário)

No painel do Railway, vá em **Variables** e adicione caso precise de alguma chave.

## Observações

- O `requirements.txt` usa `torch+cpu` (sem CUDA) para reduzir o tamanho do build (~400MB vs ~2GB com GPU)
- O Railway reinicia o app automaticamente se ele cair (`restartPolicyType = "always"`)
- Plano Hobby (~$5/mês) garante uptime 24/7 sem dormir
