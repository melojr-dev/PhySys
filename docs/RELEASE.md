# Processo de release

1. Atualize `VERSION`, `frontend/package.json` e `CHANGELOG.md`.
2. Execute `python scripts/verify_phase23.py`.
3. Quando Docker estiver disponível, execute `./scripts/e2e_stack.sh` ou `./scripts/e2e_stack.ps1`.
4. Gere/verifique o manifesto com `python scripts/build_release.py --manifest-only` e `python scripts/verify_release.py`.
5. Para build de imagens: `python scripts/build_release.py`.
6. Crie a tag exatamente como `v$(cat VERSION)`.

O manifesto contém SHA-256 dos arquivos distribuídos e deve acompanhar o artefato de release.

## Observação sobre frontend

As dependências diretas estão fixadas por versão, mas enquanto não houver `package-lock.json`, a resolução transitiva do npm não é bit-a-bit reproduzível. CI e Docker sempre executam o build do frontend para detectar incompatibilidades.
