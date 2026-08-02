# AGENTS.md — Nullain Agent SDK (regras operacionais)

## O que estamos construindo
`nullain` é um Agent SDK Python de nível de produção: o cérebro agêntico que
raciocina, usa tools, se autocorrige, aprende com uso (memória episódica) e
roteia cada tarefa para o modelo certo da Ollama Cloud. Será consumido por um
CLI em Go via protocolo NDJSON sobre stdio. Referências de arquitetura:
OpenHands software-agent-sdk (monorepo, Agent/Conversation/Tools) e o
ai-agent-handbook (loops, compaction, SOUL/AGENTS, skills, permissões).
Diferenciais a preservar em toda decisão: ModelRouter por tiers, IntentParser,
SpecValidator (Plan/Act), Learning Loop, ContextManager com compaction e
PromptAssembler (SOUL.md/AGENTS.md/skills — o SDK implementa o mesmo padrão
de contexto que você está usando agora).

## Regras vinculantes (não negociar)
1. Arquitetura hexagonal: domínio não importa httpx, sqlite ou os. Adapters
   atrás de `typing.Protocol`. Dependências injetadas (relógio, rede,
   aleatoriedade incluídos).
2. Estado da conversa = sequência imutável de eventos Pydantic (`frozen=True`).
   Estado atual é derivado por fold. Nunca mutar histórico.
3. Toda fronteira (API Ollama, resposta de modelo, args de tool, mensagens
   NDJSON, AGENTS.md do usuário) valida com Pydantic v2 strict. Saída de LLM e
   arquivos do workspace são entrada não-confiável.
4. Núcleo async (anyio/asyncio). Nada de I/O bloqueante no loop do agente.
5. Erros: hierarquia própria em `nullain/errors.py`. Proibido
   `except Exception: pass` e proibido engolir erro sem log estruturado +
   evento de erro.
6. Segurança:
   - subprocess SEMPRE via exec com lista de args; `shell=True` é proibido.
   - Todo path de tool passa por `resolve()` + `is_relative_to(workspace_root)`,
     com symlinks resolvidos ANTES da checagem.
   - Ações de escrita/execução passam pela PermissionPolicy (allow/ask/deny).
   - Conteúdo de AGENTS.md/skills do usuário nunca revoga PermissionPolicy.
   - Redigir segredos (padrões de API key/token) em logs e contexto do modelo.
   - Timeout e truncamento de output em toda execução externa.
7. Qualidade: pyright strict no núcleo; ruff com E,F,W,I,N,UP,B,S,A,C4,SIM,RUF;
   sem `Any` na API pública; docstring em todo símbolo público.
8. Testes: escreva os testes do comportamento junto com o código, nunca depois
   do milestone. Unit tests 100% offline (provider fake + respx). Todo bugfix
   ganha teste de regressão. Property tests (hypothesis) para serialização de
   eventos e validação de paths.
9. Nomes de modelo, limites e limiares vivem em `nullain.toml`
   (pydantic-settings), nunca hardcoded. Antes de implementar o adapter da
   Ollama Cloud, consulte a documentação ATUAL da API (endpoints, tool calling,
   autenticação, nomes de modelos) em vez de assumir de memória.
10. Sem dependência nova sem justificativa no PR. Proibidos: langchain/
    langgraph/crewai, requests, eval/exec dinâmico, pickle para dados externos.
11. API pública mínima exportada em `nullain/__init__.py` com `__all__`.

## Workflow
- Trabalhe UM milestone por vez (M0..M7 do plano). Antes de codar, escreva um
  mini-design (10–20 linhas) em `docs/design/MX.md` e me mostre.
- Definition of done: `make check` (ruff + pyright + pytest) verde.
- Commits pequenos, mensagem convencional (feat/fix/refactor/test/docs).
- Ambiguidade de requisito: pergunte ANTES de implementar; não invente escopo.
- Nunca desative teste, regra de lint ou checagem de tipo para "fazer passar".

## Comandos
- `make check`  → lint + typecheck + testes
- `make test`   → pytest
- `make schema` → exporta JSON Schema do protocolo para schema/
- `uv run ...`  → sempre via uv; nunca pip install global
