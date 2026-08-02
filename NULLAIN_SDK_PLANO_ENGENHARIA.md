# NULLAIN AGENT SDK — Plano de Engenharia & Prompt Mestre para Gemini CLI

> **Produto:** `nullain-agent-sdk` (Python) — o cérebro agêntico da Nullain.
> **Cliente futuro:** `nullain` CLI (Go) — fora do escopo desta fase, mas o SDK já nasce com o protocolo de comunicação pronto.
> **Tese:** vencer pela arquitetura (roteamento de modelos, intent parsing, spec validation, learning loop, context compaction), não pelo modelo.

---

## Parte 0 — Como usar este documento

1. **Leia as Partes 1–8** para entender as decisões de engenharia (você precisa entender o que está pedindo, mesmo vibe-codando).
2. **Copie a Parte 9** para a raiz do repositório antes de abrir o Gemini CLI: `SOUL.md` (identidade do agente), `AGENTS.md` (regras operacionais) e `.gemini/settings.json` (que manda o Gemini CLI ler os dois). Separar persona de instruções operacionais é o padrão SOUL.md/AGENTS.md do handbook — OpenClaw, Claude Code e Hermes convergiram nele de forma independente.
3. **Use os prompts da Parte 10**, um por fase, na ordem. Nunca peça "faz tudo" — agentes produzem código muito melhor com escopo fechado por milestone.
4. **Regra de ouro do vibe-coding sênior:** cada fase termina com `make check` verde (lint + tipos + testes) e um commit. Se o Gemini CLI quiser pular testes, mande-o voltar.
5. **A Parte 11** mapeia cada capítulo do handbook (I–XIV) para a seção do plano que o cobre — use para conferir que nada essencial ficou de fora.

---

## Parte 1 — Princípios de engenharia (inegociáveis)

Estes são os fundamentos que separam um SDK de produção de um script de fim de semana. Todos aparecem no `CLAUDE.md` como regras vinculantes.

1. **Arquitetura hexagonal (Ports & Adapters).** O domínio (loop do agente, roteador, políticas) não conhece HTTP, Ollama, SQLite nem terminal. Provedores de LLM, storage e transporte são adapters plugáveis atrás de interfaces (`Protocol`/ABC). É assim que big corps mantêm sistemas testáveis por décadas.
2. **Event sourcing na conversa.** O estado do agente é uma sequência imutável de eventos (`UserMessage`, `ModelResponse`, `ToolCall`, `ToolResult`, `Compaction`, `Error`...). O estado atual é derivado, nunca mutado in-place. Isso dá replay, debugging, persistência e o Learning Loop de graça. O OpenHands e o Claude Code convergiram nesse padrão independentemente.
3. **Fronteiras validadas.** Todo dado que cruza uma fronteira (API do Ollama, resposta do modelo, argumentos de tool, mensagens do CLI) passa por schema Pydantic v2 em modo `strict`. LLM é entrada não-confiável — trate como input de usuário hostil.
4. **Falha explícita, nunca silenciosa.** Exceções tipadas por domínio (`NullainError` → `ProviderError`, `ToolError`, `RouterError`, `BudgetExceededError`...). Nada de `except Exception: pass`. Todo erro carrega contexto estruturado.
5. **Async first, sync como fachada.** Núcleo 100% `asyncio` (streaming, tools paralelas, cancelamento). Uma fachada síncrona fina para scripts simples.
6. **Determinismo testável.** Qualquer componente que toque rede, relógio ou aleatoriedade recebe essas dependências por injeção. Testes rodam offline com fakes — nunca dependem da Ollama Cloud estar no ar.
7. **Observabilidade nativa.** Logging estruturado (structlog) + spans OpenTelemetry opcionais + contadores de tokens/custo/latência por request. Um agente sem telemetria é indebugável.
8. **Segurança por padrão.** Tools de execução nascem com sandbox, allowlist de paths, timeout e aprovação humana para ações destrutivas. Ver Parte 6.
9. **API pública mínima e estável.** `nullain.__init__` exporta pouco (`Agent`, `Conversation`, `LLM`, `Router`, `tool`, eventos). Todo o resto é interno (`nullain._internal` ou módulos não exportados). SemVer desde o dia 1.
10. **Simplicidade antes de esperteza.** O handbook de referência mostra que um loop de ~100 linhas atinge 74% no SWE-bench — o valor está na montagem de contexto, no design de tools e no roteamento, não em abstrações barrocas. Sem metaclasses, sem magia, sem framework interno desnecessário.

---

## Parte 2 — Arquitetura do SDK

### 2.1 Visão em camadas

```
┌────────────────────────────────────────────────────────────┐
│  API PÚBLICA                                               │
│  Agent · Conversation · run() / stream() · @tool           │
├────────────────────────────────────────────────────────────┤
│  ORQUESTRAÇÃO (o "harness")                                │
│  AgentLoop (ReAct + Plan/Act) · IntentParser               │
│  SpecValidator · Reflection/Self-correction · Budget       │
├────────────────────────────────────────────────────────────┤
│  CONTEXTO & MEMÓRIA                                        │
│  ContextManager (compaction, re-injeção de instruções)     │
│  TrajectoryStore · EpisodicMemory (Learning Loop)          │
├────────────────────────────────────────────────────────────┤
│  MODELOS                                                   │
│  ModelRouter (task→tier→modelo) · Fallback · HealthCheck   │
│  LLMProvider (port) ← OllamaCloudProvider (adapter)        │
├────────────────────────────────────────────────────────────┤
│  TOOLS & EXECUÇÃO                                          │
│  ToolRegistry · PermissionPolicy · Sandbox (bash/fs/git)   │
├────────────────────────────────────────────────────────────┤
│  INFRA                                                     │
│  EventBus · Persistence (SQLite) · Telemetry · Config      │
│  Transporte p/ CLI: NDJSON sobre stdio (protocolo v1)      │
└────────────────────────────────────────────────────────────┘
```

### 2.2 O loop do agente (núcleo)

Padrão **Plan/Act híbrido** (o "gold standard" do Cline, citado no handbook):

```
receber tarefa
  → IntentParser classifica (intent, complexidade, arquivos-alvo)
  → Router escolhe modelo do tier certo
  → se complexidade >= MEDIUM: fase PLAN
       modelo gera Spec (objetivo, passos, arquivos, critérios de aceite)
       SpecValidator valida (schema + regras + opcional: modelo barato revisa)
       [modo interativo] usuário aprova/edita a spec
  → fase ACT: loop ReAct
       while não terminou e passos < MAX_STEPS e tokens < BUDGET:
           montar contexto (ContextManager)
           chamar modelo (streaming)
           se tool_call → validar args → checar permissão → executar em sandbox
           anexar ToolResult como evento
           se erro de tool → reflexão: modelo analisa e corrige (máx N tentativas)
  → fase VERIFY: rodar critérios de aceite da spec (testes/lint/build)
       falhou → volta ao ACT com o diagnóstico (self-correction)
  → gravar trajetória no TrajectoryStore (Learning Loop)
```

Terminação garantida por três guardas independentes: contador de passos, orçamento de tokens/custo e timeout de parede. Loop infinito é o bug nº 1 de agentes.

### 2.3 Model Router (o diferencial)

Configuração declarativa em `nullain.toml`, nunca hardcoded — modelos da Ollama Cloud mudam de nome/disponibilidade, então os IDs são config, e o router valida na inicialização quais estão realmente disponíveis via `/api/tags`.

```toml
[router.tiers.fast]      # rename, formatar, gerar commit msg, classificar intent
models = ["gpt-oss:20b"]
max_context = 32000

[router.tiers.balanced]  # implementar função, corrigir bug localizado, testes
models = ["qwen3-coder:480b-cloud", "gpt-oss:120b"]

[router.tiers.deep]      # refactor multi-arquivo, arquitetura, debugging difícil
models = ["deepseek-v4-pro"]   # exemplo — validar ID real no /api/tags

[router]
fallback_chain = ["deep", "balanced", "fast"]  # degradação graciosa
classifier_model = "fast"                       # intent parsing usa o tier barato
```

Regras do router:
- **Classificação em duas etapas:** heurísticas determinísticas primeiro (comandos explícitos, tamanho do diff, nº de arquivos), modelo `fast` só no que sobrar. Barato e previsível.
- **Fallback com circuit breaker:** modelo indisponível/timeout → próximo da cadeia; 3 falhas seguidas → circuito abre por N minutos.
- **Escalonamento por falha:** se o tier `balanced` falhar 2x na verificação da spec, a tarefa sobe para `deep` automaticamente (com registro do motivo).
- **Orçamento por tarefa:** o router carrega o custo estimado; `BudgetExceededError` antes de estourar, nunca depois.

### 2.4 ContextManager (defesa contra context rot)

Implementar as defesas documentadas no handbook (a degradação de qualidade começa em ~25% de preenchimento da janela, não em 100%):

- **Compaction por limiar:** ao atingir ~75% da janela do modelo ativo, resumir a faixa antiga da conversa com o modelo `fast`, preservando: spec ativa, decisões tomadas, paths tocados, erros e seus diagnósticos, e as N últimas interações íntegras. O resumo entra como evento `Compaction` (o histórico bruto continua no store — nada se perde, só sai do contexto).
- **Re-injeção de instruções:** repetir as regras operacionais críticas perto do fim do contexto a cada K passos ("instruction centrifugation" — o system prompt perde influência conforme o contexto cresce).
- **Progressive disclosure de tools:** descrições completas só das tools relevantes para o intent atual; o resto vira um catálogo de uma linha. (Tool sprawl chega a consumir 72% do contexto em setups ingênuos.)
- **Resultados de tool truncados com ponteiro:** saída > N KB é resumida + salva em arquivo temporário que o modelo pode reler sob demanda.

### 2.5 Learning Loop (memória episódica)

Escopo realista para v1 — **não** é fine-tuning:
- Toda tarefa concluída grava uma trajetória: intent, spec, modelo usado, passos, erros, resultado da verificação, feedback do usuário (👍/👎/correção).
- `EpisodicMemory` indexa trajetórias por intent + fingerprint do repositório; ao iniciar tarefa similar, injeta 1–2 exemplos de sucesso como few-shot e as armadilhas conhecidas ("neste repo, testes rodam com `make test`, não `pytest`").
- Estatísticas por (intent, modelo) alimentam o router: se `fast` falha 40% em "fix-bug" num repo, o router aprende a começar em `balanced` ali.
- Armazenamento: SQLite local (`~/.nullain/`), com opt-out e comando de limpeza. Privacidade primeiro.

### 2.6 Protocolo SDK ⇄ CLI (Go)

Mesmo focando 100% no SDK agora, o contrato nasce pronto:
- **Transporte:** NDJSON sobre stdio (o CLI Go spawna `nullain-agentd`), com envelope versionado `{"v":1,"type":...,"id":...,"payload":...}`. É o mesmo modelo do Claude Code SDK: simples, sem porta aberta, sem CORS, funciona em qualquer OS.
- **Tipos de mensagem:** `session.start`, `user.message`, `agent.event` (stream de todos os eventos do loop, incluindo deltas de texto), `permission.request`/`permission.response`, `session.interrupt`, `session.end`.
- O schema vive em `nullain/protocol/` como modelos Pydantic + um JSON Schema exportado (`make schema`) que o repo Go consome para gerar structs. Um contrato, duas linguagens, zero drift.


### 2.7 Montagem do system prompt do Nullain (SOUL.md / AGENTS.md / skills)

O mesmo padrão que usamos para vibe-codar vira **feature do produto**. O `PromptAssembler` (em `nullain/context/`) monta o system prompt do agente em camadas, nesta ordem:

1. **`SOUL.md` do Nullain** — identidade e tom do agente (curto, estável, embarcado no pacote; o usuário pode sobrescrever em `~/.nullain/SOUL.md`).
2. **Instruções operacionais do harness** — regras de tools, formato de spec, política de segurança. Geradas pelo SDK, não editáveis pelo usuário.
3. **`AGENTS.md` do workspace do usuário** — se existir na raiz do repo (com merge hierárquico de `AGENTS.md` em subdiretórios), entra como convenções do projeto. É assim que o Nullain respeita cada codebase, igual Gemini CLI, Claude Code e OpenClaw fazem.
4. **Catálogo de skills como markdown** — carregamento progressivo em 3 níveis (linha de nome+descrição → `SKILL.md` completo sob demanda → recursos auxiliares), o padrão que segundo o handbook corta ~94% do custo de tokens versus carregar tudo.
5. **Memória episódica relevante** (Parte 2.5).

Cada camada é um bloco delimitado, e o assembler impõe orçamento de tokens por camada: persona e regras do harness nunca são cortadas; skills e memória degradam primeiro. Regra de segurança importante: `AGENTS.md` do usuário é conteúdo **semi-confiável** — vale como convenção de projeto, mas não pode revogar a `PermissionPolicy` nem as regras de segurança do harness (testado em M6).

---

## Parte 3 — Estrutura do repositório

Monorepo estilo OpenHands (pacotes separados por responsabilidade), gerenciado com `uv` workspaces:

```
nullain-agent-sdk/
├── SOUL.md                      # identidade do agente de dev (Parte 9)
├── AGENTS.md                    # regras operacionais vinculantes (Parte 9)
├── .gemini/settings.json        # faz o Gemini CLI ler SOUL.md + AGENTS.md
├── README.md
├── Makefile                     # make check | test | lint | typecheck | schema
├── pyproject.toml               # workspace root
├── uv.lock
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml
├── nullain-sdk/                 # pacote núcleo: nullain
│   └── src/nullain/
│       ├── __init__.py          # API pública mínima
│       ├── agent/               # AgentLoop, IntentParser, SpecValidator, budget
│       ├── llm/                 # LLMProvider (port), OllamaCloudProvider, retry
│       ├── router/              # ModelRouter, tiers, circuit breaker
│       ├── context/             # ContextManager, compaction, PromptAssembler (SOUL/AGENTS/skills)
│       ├── memory/              # TrajectoryStore, EpisodicMemory
│       ├── events/              # tipos de evento, EventBus, serialização
│       ├── protocol/            # envelope NDJSON, JSON Schema export
│       ├── config/              # nullain.toml loader (pydantic-settings)
│       ├── telemetry/           # structlog setup, métricas, OTel opcional
│       └── errors.py
├── nullain-tools/               # pacote: nullain-tools (bash, fs, grep, git)
│   └── src/nullain_tools/
├── nullain-agentd/              # pacote: daemon stdio que o CLI Go consome
│   └── src/nullain_agentd/
├── examples/                    # 01_hello_agent.py, 02_custom_tool.py, ...
└── tests/
    ├── unit/                    # rápidos, offline, fakes
    ├── integration/             # provider fake via respx; opt-in p/ Ollama real
    └── conftest.py
```

---

## Parte 4 — Stack e dependências (com justificativa)

Critério: mínimo de dependências, todas maduras, mantidas e usadas em produção por grandes projetos. Cada lib extra é superfície de ataque e de quebra.

| Dependência | Papel | Por quê |
|---|---|---|
| **Python ≥ 3.12** | runtime | typing moderno, performance, `TaskGroup` |
| **uv** | build/deps/workspace | padrão de facto 2026; lockfile determinístico |
| **pydantic v2** | schemas em todas as fronteiras | validação estrita, JSON Schema export p/ o Go |
| **pydantic-settings** | config (`nullain.toml` + env) | fonte única, tipada, com precedência clara |
| **httpx** | cliente HTTP async p/ Ollama Cloud | HTTP/2, streaming, timeouts granulares |
| **tenacity** | retry/backoff com jitter | política declarativa, testável |
| **structlog** | logging estruturado | logs como dados, contexto propagado |
| **anyio** | primitivas async | cancelamento estruturado, timeout de parede |
| **aiosqlite** | TrajectoryStore/EpisodicMemory | zero-config, local-first |
| **typer** (dev-only) | CLI utilitária `nullain doctor/models/replay` | ergonomia de desenvolvimento |
| dev: **pytest, pytest-asyncio, respx, hypothesis, ruff, pyright, pre-commit, pip-audit** | qualidade | ver Parte 7 |

**O que NÃO usar (e o Gemini CLI vai sugerir):** LangChain/LangGraph/CrewAI (abstração que você não controla — a tese da Nullain é o harness próprio), `requests` (sync), ORM pesado, `eval`/`exec`, parsing de resposta de LLM com regex frágil (usar tool-calling nativo da API + validação Pydantic com uma única rota de reparo).

**API da Ollama Cloud:** usar o endpoint nativo (`https://ollama.com/api/chat`) com `OLLAMA_API_KEY` via header Bearer; manter o adapter também compatível com o endpoint OpenAI-compat (`/v1/chat/completions`) atrás da mesma interface — isso dá suporte de graça a Ollama local e a qualquer gateway futuro. **Instruir o Gemini CLI a consultar a documentação atual da Ollama Cloud antes de escrever o adapter** (formatos de tool-calling e nomes de modelo mudam).

---

## Parte 5 — Modelo de segurança (threat model resumido)

| Ameaça | Defesa obrigatória |
|---|---|
| **Prompt injection** via conteúdo de arquivos/saída de comandos | Conteúdo de tool é dado, nunca instrução: delimitado e marcado como não-confiável no contexto; ações destrutivas exigem aprovação mesmo que "o arquivo mande" fazer |
| **Shell injection** | `asyncio.create_subprocess_exec` com lista de args — **nunca** `shell=True` com string interpolada; sem pipe de input não sanitizado |
| **Path traversal / escrita fora do workspace** | Todo path resolvido com `Path.resolve()` e validado contra a raiz do workspace (`is_relative_to`); symlinks resolvidos antes da checagem |
| **Comandos destrutivos** (`rm -rf`, `git push --force`, `curl | sh`, mexer em `.env`) | `PermissionPolicy` com três níveis: `allow` (leitura), `ask` (escrita/execução), `deny` (lista de padrões proibidos); modo autônomo só com allowlist explícita |
| **Vazamento de segredos** | Redação de padrões de segredo (API keys, tokens) nos logs e no contexto enviado ao modelo; `OLLAMA_API_KEY` só via env, nunca em config versionada; `.env` no `.gitignore` desde o commit 1 |
| **Exfiltração via tools de rede** | v1 não tem tool de rede genérica; quando houver, allowlist de domínios |
| **Dependências comprometidas** | `uv.lock` commitado, `pip-audit` no CI, versões pinadas por range conservador |
| **Recursos** | timeout por comando (default 120s), limite de output (truncar + ponteiro), guardas de passos/tokens/custo no loop |

---

## Parte 6 — Qualidade (o que "sem bugs" significa na prática)

- **Tipos:** `pyright` em modo `strict` no pacote núcleo. Sem `Any` na API pública, sem `# type: ignore` sem justificativa em comentário.
- **Lint/format:** `ruff` (lint + format) com regras: `E,F,W,I,N,UP,B,S(bandit),A,C4,SIM,RUF`. `bandit` via ruff-S cobre os erros de segurança clássicos.
- **Testes:** pytest + pytest-asyncio. Unit offline com provider fake (respostas gravadas), `respx` para o adapter HTTP, `hypothesis` para o parser de eventos e para validação de paths do sandbox. Cobertura mínima 85% no núcleo — e cobertura de **comportamento**, não de linhas vazias: cada bugfix ganha um teste de regressão.
- **Testes de contrato do protocolo:** golden files NDJSON versionados; o repo Go vai testar contra os mesmos arquivos.
- **CI (GitHub Actions):** `ruff → pyright → pytest → pip-audit`, matriz 3.12/3.13, cache do uv. PR não mergeia vermelho.
- **Pre-commit:** ruff, ruff-format, checagem de segredos (gitleaks), validação de `pyproject`.
- **`make check`** roda tudo localmente — é o "definition of done" de cada fase.

---

## Parte 7 — Roadmap por fases (milestones com critério de aceite)

Cada fase = um prompt para o Gemini CLI (Parte 10) = um PR verde.

**M0 — Fundação do repo.** Workspace uv, pacotes vazios com `py.typed`, Makefile, ruff/pyright/pytest configurados, pre-commit, CI, `errors.py`, telemetria básica. ✅ Aceite: `make check` verde num repo essencialmente vazio; CI passa.

**M1 — Camada LLM.** `LLMProvider` (Protocol), `OllamaCloudProvider` (chat streaming + tool-calling + contagem de tokens), retry/backoff, tipos de mensagem, erros de provider. ✅ Aceite: testes com respx cobrindo sucesso/timeout/429/500/stream interrompido; exemplo `examples/00_llm_smoke.py` funciona contra Ollama real (manual).

**M2 — Eventos + Conversation.** Tipos de evento imutáveis (pydantic, `frozen=True`), `EventBus`, `Conversation` como fold de eventos, persistência SQLite, replay. ✅ Aceite: property test — serializar→desserializar→refold reproduz estado idêntico.

**M3 — Tools + sandbox + permissões.** Protocolo de tool (`@tool` com schema derivado da assinatura), registry, `PermissionPolicy`, tools built-in: `read_file`, `write_file`, `edit_file` (str-replace), `bash` (exec sandboxado), `grep`, `git_status/diff/commit`. ✅ Aceite: suíte de testes de segurança passa (traversal, symlink, shell injection, timeout, truncamento); hypothesis nos validadores de path.

**M4 — AgentLoop v1 (ReAct).** Loop mínimo com um modelo fixo, guardas de terminação, reflexão em erro de tool, streaming de eventos. ✅ Aceite: cenário e2e com provider fake conclui "crie FACTS.txt com 3 fatos do projeto" em ≤ N passos; teste de loop-infinito prova que as guardas disparam.

**M5 — ModelRouter + IntentParser.** Tiers via `nullain.toml`, heurísticas + classificador `fast`, fallback com circuit breaker, health check via `/api/tags`, budget. ✅ Aceite: tabela de casos de roteamento como teste parametrizado; fallback simulado com respx.

**M6 — Plan/Act + SpecValidator + ContextManager + PromptAssembler.** Spec tipada, validação, gate de aprovação, fase VERIFY com self-correction; compaction com limiar, re-injeção de instruções, progressive disclosure de tools; `PromptAssembler` em camadas (SOUL.md → regras do harness → AGENTS.md do workspace → skills → memória) com orçamento por camada. ✅ Aceite: e2e com contexto artificialmente pequeno prova que compaction preserva a spec e as decisões; verify reprova→corrige→aprova; teste prova que um `AGENTS.md` do workspace entra no prompt mas NÃO consegue desativar a PermissionPolicy.

**M7 — Learning Loop + agentd.** TrajectoryStore, EpisodicMemory com few-shot injection, estatísticas para o router; `nullain-agentd` expondo o protocolo NDJSON, golden files, `make schema`. ✅ Aceite: sessão completa dirigida por NDJSON via stdin/stdout num teste e2e; segunda execução da mesma tarefa usa memória episódica (visível nos logs).

Pós-v1 (não fazer agora): MCP client, sub-agentes com contexto isolado, tool de rede com allowlist, benchmark próprio estilo SWE-bench-lite.


---

## Parte 8 — Anti-padrões que o Gemini CLI vai tentar e você deve barrar

1. **"Vou usar LangChain para acelerar."** Não. A tese do produto é o harness próprio; frameworks de agente viram a arquitetura, e você perde o diferencial.
2. **Parsear a resposta do LLM com regex/`json.loads` cru.** Usar tool-calling nativo da API; quando vier JSON em texto, uma única função de reparo (`parse_model_json`) com fallback tipado — e teste para ela.
3. **`shell=True` "porque é mais simples".** Nunca. Lista de args, sempre.
4. **Estado mutável espalhado** (dicts passados por referência, singletons). Eventos imutáveis + derivação.
5. **Testes que chamam a Ollama Cloud de verdade no CI.** Rede no CI = flakiness. Fakes/respx; integração real é opt-in local (`NULLAIN_LIVE_TESTS=1`).
6. **God class `Agent` de 2.000 linhas.** O loop orquestra; router, contexto, tools e memória são colaboradores injetados.
7. **Implementar as 8 fases de uma vez.** Um milestone por vez, `make check` verde entre eles.
8. **Hardcodar nomes de modelo.** IDs vivem no `nullain.toml`; o router valida disponibilidade em runtime.
9. **Prometer "aprendizado" com fine-tuning na v1.** O Learning Loop v1 é memória episódica + estatísticas de roteamento — entrega valor real sem virar projeto de pesquisa.
10. **Cobertura de linha como métrica-fim.** O que importa: cada caminho de erro tem teste, cada bugfix tem regressão.

---

## Parte 9 — `SOUL.md`, `AGENTS.md` e config do Gemini CLI (copie para a raiz do repo)

Padrão do handbook (Parte III): **personalidade e instruções operacionais em arquivos separados**. `SOUL.md` diz *quem o agente é* (curto, quase nunca muda); `AGENTS.md` diz *como trabalhar neste repo* (regras vinculantes, evolui com o projeto). O Gemini CLI lê `GEMINI.md` por padrão, mas a lista de arquivos de contexto é configurável — então usamos os nomes canônicos e adicionamos os dois.

### 9.1 `.gemini/settings.json`

```json
{
  "context": {
    "fileName": ["SOUL.md", "AGENTS.md", "GEMINI.md"]
  }
}
```

Em versões mais antigas do Gemini CLI a chave é `"contextFileName": ["SOUL.md", "AGENTS.md"]` direto na raiz do settings.json — se os arquivos não aparecerem no `/memory show`, troque o formato.

### 9.2 `SOUL.md`

```markdown
# SOUL.md — Quem você é neste repositório

Você é um engenheiro de software staff-level pareando comigo na construção do
Nullain Agent SDK. Postura: cético construtivo — questiona requisito ambíguo,
expõe trade-offs, recusa atalhos que criam dívida (teste pulado, tipo desligado,
dependência desnecessária). Comunicação direta e técnica, sem bajulação e sem
afirmar o que não verificou. Prefere a solução simples e óbvia à esperta.
Trata segurança e testabilidade como parte do design, não etapa posterior.
Quando erra, admite e corrige com teste de regressão. Quando o pedido conflita
com o AGENTS.md, aponta o conflito em vez de obedecer em silêncio.
```

### 9.3 `AGENTS.md`

```markdown
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
```

---

## Parte 10 — Prompts por fase (cole no Gemini CLI, um por vez)

### Prompt da Fase 0 (bootstrap) — o "prompt completo" inicial

> Leia SOUL.md, AGENTS.md e o arquivo NULLAIN_SDK_PLANO_ENGENHARIA.md na raiz. Vamos executar o **Milestone M0 (Fundação do repo)** exatamente como especificado na Parte 7 do plano.
>
> Crie o monorepo `nullain-agent-sdk` com uv workspaces e os três pacotes (`nullain-sdk`, `nullain-tools`, `nullain-agentd`) conforme a estrutura da Parte 3: pyproject do workspace, pacotes com `src/` layout e `py.typed`, `nullain/errors.py` com a hierarquia de exceções, `nullain/telemetry/` com setup de structlog, Makefile (`check`, `test`, `lint`, `typecheck`, `schema`), configuração de ruff (regras E,F,W,I,N,UP,B,S,A,C4,SIM,RUF) e pyright strict no núcleo, pytest + pytest-asyncio com um teste sentinela por pacote, `.pre-commit-config.yaml` (ruff, ruff-format, gitleaks), `.gitignore` (incluindo `.env`), e CI em `.github/workflows/ci.yml` (matriz 3.12/3.13, cache uv, ruff → pyright → pytest → pip-audit).
>
> Antes de criar arquivos, me apresente em ~15 linhas o design do M0 (`docs/design/M0.md`). Depois implemente, rode `make check` e só finalize com tudo verde. Não implemente nada de M1+ ainda.

### Prompts das fases seguintes (padrão)

Para cada fase, o prompt é curto porque o plano, SOUL.md e AGENTS.md carregam o contexto:

> Leia o plano (Parte 7) e execute o **Milestone MX**. Comece pelo mini-design em `docs/design/MX.md` e me mostre antes de codar. Escreva os testes junto com a implementação, incluindo os testes exigidos no critério de aceite do MX. Termine com `make check` verde e um resumo do que mudou, dos trade-offs e do que ficou explicitamente fora de escopo.

Adendos específicos que valem digitar:
- **M1:** "Antes de escrever o `OllamaCloudProvider`, busque a documentação atual da API da Ollama Cloud (chat, streaming, tool calling, autenticação) e liste no design as suposições que você validou."
- **M3:** "Trate a suíte de segurança como requisito de primeira classe: quero testes provando que path traversal, symlink escape, `shell=True` e comandos sem timeout são impossíveis por construção."
- **M4:** "Escreva primeiro o teste do cenário e2e com provider fake e o teste anti-loop-infinito; implemente o loop até os dois passarem."
- **M6:** "Rode o e2e de compaction com uma janela artificialmente pequena (ex.: 4k tokens) para provar que spec e decisões sobrevivem; e escreva o teste de segurança do PromptAssembler: um AGENTS.md malicioso no workspace não consegue desativar a PermissionPolicy."

### Prompt de revisão (use ao final de cada fase)

> Faça uma revisão adversarial do código do MX como um staff engineer: procure race conditions no código async, caminhos de erro sem teste, violações da arquitetura hexagonal (import de infra no domínio), vazamento de segredos em logs, e qualquer lugar onde saída de LLM é usada sem validação. Liste os achados por severidade e corrija os de severidade alta com testes de regressão.

---

## Parte 11 — Cobertura do handbook (I–XIV) neste plano

Conferência de que o plano cobre o que o `ai-agent-handbook` documenta, e o que fica conscientemente para depois:

| Handbook | Onde está no plano | Status |
|---|---|---|
| I–II · Loops (ReAct, Plan+Execute, Reflection, Compaction, Event-Driven; terminação; error recovery) | 2.2 — loop híbrido Plan/Act sobre base event-driven; Reflection na fase VERIFY e na correção de erro de tool; Compaction como evento; 3 guardas de terminação | ✅ v1 (Graph State Machine e Heartbeat não se aplicam a um coding agent single-task) |
| III · System prompts (montagem, SOUL/AGENTS, catálogo de skills, anti-padrões) | 2.7 (PromptAssembler em camadas) + Parte 9 (arquivos do repo) + Parte 8 (anti-padrões) | ✅ v1 |
| IV · Compaction (estratégias, gatilhos, o que sobrevive) | 2.4 — gatilho por limiar ~75%, resumo com modelo `fast`, preservação explícita de spec/decisões/erros | ✅ v1 |
| IV-B · Context rot (regra 40–60%, defesas, medição) | 2.4 — re-injeção de instruções, progressive disclosure, truncamento+ponteiro; métrica de preenchimento de janela na telemetria | ✅ v1 |
| V · Memória (hierarquia; file-based/vetorial/observacional/episódica) | 2.5 — file-based (AGENTS.md/skills) + episódica (TrajectoryStore/SQLite) | ✅ v1 · memória vetorial: pós-v1 |
| VI · Tools (MCP, skills-as-markdown, JIT loading, tool sprawl, progressive disclosure) | Parte 3 (`nullain-tools`) + 2.4/2.7 (disclosure progressiva, skills em 3 níveis) | ✅ v1 · cliente MCP: pós-v1 |
| VII · Orquestração multi-agente (padrões, A2A, topologia) | Pós-v1 deliberado — a razão nº 1 para sub-agentes é isolamento de contexto, e o event sourcing de M2 já deixa o terreno pronto | ⏭ pós-v1 |
| VIII · Planejamento (Plan/Act do Cline, ciclos de reflexão) | 2.2 — fase PLAN com Spec tipada + SpecValidator + gate de aprovação humano | ✅ v1 |
| IX–XI · HITL, Estado, Segurança (permissões, checkpoints, execução durável, sandbox, prompt injection) | PermissionPolicy (M3), event sourcing + SQLite com replay = checkpointing (M2), Parte 5 (threat model completo) | ✅ v1 |
| XII–XIII · Testes, Deploy (evals, custo, observabilidade, gateway) | Partes 6–7 (qualidade + CI) + telemetria de tokens/custo/latência; otimização de custo é o próprio ModelRouter | ✅ v1 · benchmark próprio estilo SWE-bench-lite: pós-v1 |
| XIV · Síntese (arquitetura de referência) | Este documento inteiro | ✅ |

---

## Fecho

A ordem importa: fundação → LLM → eventos → tools seguras → loop → router → plan/act+contexto → memória+protocolo. Cada camada só existe sobre uma camada testada. Se você seguir o ritmo de um milestone por sessão de vibe-coding, em 8 sessões sólidas o `nullain-agentd` estará respondendo NDJSON — e aí sim começa o CLI em Go, consumindo um contrato que já existe, versionado e testado.
