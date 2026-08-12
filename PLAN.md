# Plano de Engenharia — Ecossistema Nullain

> Documento de planejamento para evolução do `nullain-agent` + `nullain-agent-sdk`.
> Pensado para ser usado como contexto no Claude Code (pode virar um `PLAN.md` na raiz do repo ou alimentar o `CLAUDE.md`).

---

## 1. Estado atual (o que já existe — não recriar)

O ecossistema hoje é composto por dois repositórios:

**`nullain-agent-sdk`** (monorepo, Python 3.12+, uv, pyright strict, ruff) é o motor. Ele já implementa arquitetura hexagonal com event sourcing, e contém três pacotes: `nullain-sdk` (core: ModelRouter com circuit breaker, AgentLoop Plan/Act/Verify, ContextManager com compaction, EpisodicMemory, MCP client, PermissionPolicy, sandbox fail-closed de SO, plugins assinados Ed25519, Workflow determinístico, PostgresEventStore, OTLP tracing), `nullain-tools` (arquivo, shell, ripgrep, git, checkpoints) e `nullain-agentd` (daemon NDJSON stdio com schema exportado — a ponte para outras linguagens).

**`nullain-agent`** é o produto de referência sobre o SDK: FastAPI + SSE, AgentBridge, ferramentas de domínio (ZuckPay, Wavespeed, RAG com Qdrant multi-tenant, sandbox e2b/docker/aio, Composio MCP), identidade via SOUL.md, deploy em Docker/Coolify.

**Conclusão estratégica:** a maioria dos "SDKs essenciais" do diagrama (Memory, Tools, Planning, Code, Security) **já existe como módulo interno do SDK**. O trabalho agora não é criar — é decidir o que extrair, o que formalizar e o que apenas evoluir onde está.

---

## 2. Decisão de arquitetura: extrair SDK ou manter módulo?

Regra de ouro para não cair em over-engineering — **só extrai um módulo para SDK/repo separado se pelo menos um destes for verdadeiro:**

1. **Linguagem diferente** do core (ex: Rust) — obrigatório extrair.
2. **Ciclo de release próprio** com consumidores fora do ecossistema Nullain.
3. **Dependências pesadas** que inflam a instalação de quem não usa (ex: OCR/torch da visão).

Aplicando a regra:

| Candidato | Decisão | Justificativa |
|---|---|---|
| Memory, Planning, Tools, Security | **Manter como módulos** do `nullain-sdk` | Nenhum critério atende; extrair só adiciona overhead de versionamento |
| **SDK Search (Rust)** | **Extrair — novo repo `nullain-sdk-search`** | Critério 1: linguagem diferente, hot path de performance |
| **SDK Vision (VLM)** | **Manter como módulo** do `nullain-sdk` | Nenhum critério atende: a implementação real (VLM hospedado via `ModelRouter`/`LLMProvider`) não traz dependência pesada — revisado em 2026-08-12, ver Fase 2 |
| Vision local (OCR/CV, pasta `vision/` do `nullain-agent`) | **Decisão adiada** | Critério 3 se aplicaria (≈600MB, majoritariamente sem uso em produção hoje) — extrair só se/quando essa via for de fato adotada |
| Voice, Emotion (futuros) | Módulo primeiro, extrair depois se crescer | Começar dentro do SDK, promover se justificar |

Resultado: o diagrama vira realidade com **3 repositórios**, não 8 (vision
VLM vira módulo do core, não repo novo — ver Fase 2):

```
nullain-agent            (produto)
  └── nullain-agent-sdk  (facade + core, Python)
        │     ├── port VisionProvider → adapter ModelRouterVisionProvider (módulo, sem deps novas)
        └── nullain-sdk-search   (Rust + bindings PyO3)   ← novo
```

---

## 3. Stack definida

| Camada | Escolha | Observação |
|---|---|---|
| Core / Facade | Python 3.12+, uv, pyright strict, ruff | Já é o padrão do SDK — manter |
| Search engine | **Rust** (edition 2021+), crate `nullain-search-core` | tantivy (full-text) + opcionalmente reqwest para fetch paralelo |
| Bindings Rust→Python | **PyO3 + maturin** | Publica wheel no PyPI como `nullain-search`; abi3 para compatibilidade |
| Vision | Python: OCR (rapidocr/tesseract), VLM via ModelRouter | Reusa o roteamento de modelos do core em vez de cliente próprio |
| Contrato entre SDKs | Interface Python tipada (Protocol/ABC) + fallback | Search: se wheel Rust indisponível, cai no SearXNG/ripgrep atual |
| Contrato cross-linguagem | NDJSON `nullain-agentd` (já existe) | Rust também pode falar direto com o daemon no futuro |
| Estado | Postgres/Supabase (prod) · SQLite (dev) — já existe | Sem mudança |
| Vetores | Qdrant multi-tenant — já existe | Search Rust pode indexar local; Qdrant continua para memória semântica |

---

## 4. Como tudo se interliga (contratos)

O princípio central: **o Agent nunca conhece os sub-SDKs; só o SDK Facade os conhece.** E o SDK Facade só conhece **interfaces**, nunca implementações.

```
nullain-agent
    │  (import nullain — API pública SemVer)
    ▼
nullain-sdk (Facade)
    │
    ├── port SearchProvider (Protocol)
    │       ├── adapter RustSearch  → wheel nullain-search (PyO3)
    │       └── adapter WebSearch   → SearXNG/DDG (fallback, já existe)
    │
    └── port VisionProvider (Protocol)
            └── adapter ModelRouterVisionProvider → módulo do core, VLM via ModelRouter/LLMProvider (sem deps novas)
```

Regras do contrato:

1. Cada sub-SDK expõe um **Protocol** definido NO CORE (`nullain-sdk/ports/`), não no sub-SDK — o core é dono da interface (Dependency Inversion).
2. Sub-SDKs são **extras opcionais**: `pip install nullain-sdk[search-rust,vision]`. Instalação base continua leve.
3. Todo sub-SDK entra no agente **como ferramenta no ToolRegistry**, herdando PermissionPolicy, sandbox e Authority automaticamente — segurança de graça.
4. **Fail-open para capacidade, fail-closed para segurança**: sub-SDK ausente degrada para fallback; permissão ausente nega.
5. Versionamento: SemVer nos três pacotes; o Facade declara ranges compatíveis (`nullain-search >=0.1,<0.2`).

---

## 5. Roadmap de execução (fases para o Claude Code)

### Fase 0 — Fundação dos contratos (no `nullain-agent-sdk`) ✅ concluída em 2026-08-11
- Criar `nullain-sdk/ports/search.py` com `SearchProvider` Protocol (métodos: `index`, `query`, `fetch`) tipado e documentado.
- Criar `nullain-sdk/ports/vision.py` com `VisionProvider` Protocol (`describe_image`, `ocr`, `analyze_screenshot`).
- Refatorar o `web_search` atual para ser o **adapter default** do port de Search (sem mudança de comportamento).
- Testes de contrato: uma suíte que qualquer adapter deve passar (contract tests).
- **Gate:** `make check` verde; nenhuma quebra de API pública.
- **Entregue:** `SearchProvider`/`VisionProvider` Protocols criados; `web_search` refatorado em `WebSearchProvider` (comportamento idêntico, 16 testes pré-existentes sem alteração); `test_search_provider_contract.py`/`test_vision_provider_contract.py` criados como suítes de contrato reutilizáveis; zero mudança em `nullain/__init__.py::__all__`.

### Fase 1 — `nullain-sdk-search` (Rust) ✅ concluída em 2026-08-11
- Novo repo: workspace Cargo com `nullain-search-core` (lib pura Rust) + `nullain-search-py` (PyO3).
- Escopo v0.1: indexação local com tantivy (arquivos do workspace) + busca híbrida (BM25). NÃO reimplementar web search — isso fica no fallback Python.
- Build/CI: maturin + GitHub Actions com wheels para linux/macos/windows (abi3-py312).
- No SDK: adapter `RustSearchAdapter` implementando o Protocol, registrado se o wheel importar com sucesso.
- **Gate:** contract tests do core passando contra o adapter Rust; benchmark comparativo vs ripgrep documentado.
- **Entregue:** `nullain-sdk-search` publicado no GitHub com a API `SearchIndex` (index/index_directory/query/fetch); `RustSearchAdapter` no SDK traduzindo exceções PyO3 para `SearchError` e envolvendo as chamadas bloqueantes em `asyncio.to_thread`; `fetch` composto com `WebSearchProvider` injetado (o índice Rust não busca URLs); contract tests passando contra a wheel real, com skip automático quando ausente; extra `search-rust` no `pyproject.toml`.
- **Pendente:** wheel ainda não publicada no PyPI (issue #77 — `tool.uv.sources` aponta para o path local do repo irmão como solução temporária); benchmark comparativo vs ripgrep não feito nesta fase.

### Fase 2 — `VisionProvider`: adapter VLM dentro do `nullain-sdk` (reescopada em 2026-08-12)
- **Reescopo:** o plano original assumia que `VisionProvider` precisaria de
  extração para um pacote `nullain-vision` separado, pelo critério 3 da
  seção 2 (dependências pesadas — OCR/Chromium, ~600MB). Um diagnóstico da
  pasta `vision/` do `nullain-agent` mostrou que essa suposição não
  corresponde ao que roda em produção: a pasta `vision/` (4.400 LOC,
  OpenCV/Tesseract/Playwright, ~600MB de deps) está **majoritariamente sem
  uso** — só 2 call-sites reais, nenhuma rota de API dedicada, e as 14 tools
  que ela expõe nunca são registradas no agente rodando. A implementação de
  visão **realmente usada** é `agent/vision_groq.py`: uma chamada VLM
  hospedada (Groq, Llama 3.2 vision), sem dependência pesada nenhuma — só
  `httpx`, que o SDK já tem. Essa é a peça que encaixa naturalmente no
  Protocol `VisionProvider` (`bytes` + `mime_type` → `str`), então ela vira
  o Fase 2, e o critério 3 simplesmente não se aplica a ela: nenhuma
  dependência nova, nenhum peso de instalação. A pasta `vision/`
  local (OCR/CV) fica para uma decisão futura separada, não bloqueia esta
  fase.
- Estender `nullain.llm.types.ChatMessage.content` para aceitar uma lista
  de blocos (`TextPart`/`ImagePart`), preservando o caminho `str`/`None`
  existente sem nenhuma mudança de payload — pré-requisito para qualquer
  request multimodal, feito como commit separado do adapter.
- Implementar `ModelRouterVisionProvider` (`nullain.ports.vision`) —
  `describe_image`/`ocr`/`analyze_screenshot` delegando a um `ModelRouter` +
  `LLMProvider` **injetados** (nunca instanciados pelo adapter); os três
  métodos diferem só no prompt, não no caminho de código.
- `VisionError` (`nullain.errors`), seguindo o padrão de `SearchError`:
  qualquer falha do provider subjacente (incluindo rejeição de conteúdo
  multimodal por um modelo sem suporte) vira `VisionError`, nunca falha
  silenciosa.
- `ModelRouterVisionProvider` plugado em `_ADAPTER_FACTORIES` de
  `test_vision_provider_contract.py`, validado pela mesma suíte de
  contrato usada por qualquer adapter futuro.
- No `nullain-agent`: substituir o import direto de `agent/vision_groq.py`
  por consumo via port do SDK fica para a Fase 3 (Integração e promoção no
  produto) — não faz parte do escopo desta fase.
- **Gate:** `make check` verde; nenhuma dependência nova no `pyproject.toml`;
  nenhuma quebra de API pública.

### Fase 3 — Integração e promoção no produto
- `nullain-agent`: expor as novas capacidades como ferramentas com `PermissionLevel` adequado.
- Atualizar SOUL.md/AGENTS.md com regras de escolha de ferramenta (quando usar search local vs web, quando acionar vision).
- Evals: adicionar casos em `evals/` cobrindo search e vision.
- Docs: atualizar diagrama de arquitetura nos dois READMEs.

### Fase 4 — Próximos SDKs (backlog, só após 0–3)
- `voice` (STT/TTS) como módulo do core primeiro.
- Multi-agent collab: já existe base (`Workflow` + subagent Authority) — evoluir antes de criar repo novo.
- Telemetry/self-reflection: construir sobre EpisodicMemory + OTLP existentes.

---

## 6. Setup do Claude Code (por repositório)

Cada repo deve ter um `CLAUDE.md` na raiz com:

**`nullain-agent-sdk/CLAUDE.md`** — comandos (`make check`, `make test`, `make typecheck`, `make schema`), regra de que TODA mudança passa por `make check` antes de commit, aviso de que a API pública segue `docs/api-stability.md`, e a convenção de que ports vivem no core e adapters nunca vazam tipos internos.

**`nullain-sdk-search/CLAUDE.md`** — comandos cargo (`cargo test`, `cargo clippy -- -D warnings`, `maturin develop`), regra de que a lib core não conhece Python (PyO3 só no crate de bindings), e que o contrato de referência são os contract tests do repo do SDK.

**`nullain-agent/CLAUDE.md`** — como rodar (`docker compose up --build`, `python main.py --web`), regra de que o repo NÃO implementa loop de agente (tudo via SDK), e que ferramentas de domínio sempre entram com PermissionLevel explícito.

Fluxo de trabalho sugerido com o Claude Code: uma fase por sessão, começando sempre por "leia o PLAN.md e o CLAUDE.md, e me proponha os passos da Fase N antes de codar" — plan mode primeiro, execução depois, `make check`/`cargo clippy` como gate de cada entrega.

---

## 7. Riscos e mitigações

| Risco | Mitigação |
|---|---|
| Complexidade de build Rust+Python (maturin, wheels multiplataforma) | abi3, CI com matriz mínima, e fallback Python garante que nada quebra sem o wheel |
| Drift de contrato entre facade e sub-SDKs | Contract tests no core rodando contra cada adapter no CI |
| Fragmentação prematura (repos demais) | Regra dos 3 critérios da seção 2 — módulo por padrão, repo por exceção |
| Instalação pesada para usuários casuais | Extras opcionais (`[search-rust]`, `[vision]`); base sempre leve |
| Segurança de novas ferramentas | Tudo entra pelo ToolRegistry — herda PermissionPolicy/sandbox; nada de caminho paralelo |
