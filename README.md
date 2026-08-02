# Nullain Agent SDK

`nullain-agent-sdk` é o cérebro agêntico da Nullain em Python.

## Recursos Principais
- Arquitetura Hexagonal (Ports & Adapters)
- Event Sourcing na conversa com histórico imutável
- Roteador de Modelos por Tiers (`fast`, `balanced`, `deep`)
- Gerenciador de Contexto com Compaction e re-injeção de instruções
- Protocolo NDJSON sobre stdio para consumo por CLI Go
- Modelo Plan/Act híbrido com validação de specs

## Desenvolvimento

```bash
# Executar todas as verificações de qualidade
make check

# Executar testes
make test

# Executar linter e formatação
make lint

# Executar checagem de tipos
make typecheck
```
