# Nullain-Code-SDK vs Claude Code / Grok Build / Gemini CLI — Relatório + Plano de Correção

Data: 2026-08-02 · Base: leitura real do código (mapa arquitetural) + docs oficiais dos 3 concorrentes.

---

## PARTE 1 — Onde você já ganha e onde perde

### Diferenciais reais do Nullain hoje (não têm nos 3 concorrentes)

1. **Backend open-source first-class.** Claude/Grok/Gemini são casados com uma API fechada (Anthropic/xAI/Google). Nullain roda Ollama nativo (`/api/chat`) + compat OpenAI (`/v1`). **Este é o maior fosso explorável** — privacidade, air-gap, custo zero, portabilidade de modelo.
2. **Plan/Act/Verify tipado com `TaskSpec`/`SpecValidator`.** Os 3 têm plan mode raso. Nullain já prototipa um contrato estruturado (M6). Nenhum concorrente expõe `TaskSpec` validável.
3. **Compaction plugável + `EpisodicMemory`.** Gemini emite evento `ChatCompressed` mas o sumarizador é opaco; Grok só expõe hook; Claude é undocumented. Nullain tem `ContextManager` + `EpisodicMemory` (SQLite) — base pra sumarização inspecionável e few-shot recall.
4. **`ModelRouter` + `CircuitBreaker`.** Gemini tem fallback interno; Grok tem `--effort`; Claude tem seleção de modelo. Nenhum expõe router explícito por-intent com circuit breaker e fallback chain. Você tem.
5. **`Clock` port (DI de tempo determinístico).** Permite testar timeout/recovery sem sleep. Coisa fina que os outros não ostentam.
6. **Segurança real no tools layer:** subprocess arg-list (sem `shell=True`), validação de path com `is_relative_to` + symlink-resolve, deny patterns, redação de secrets, truncagem de output, timeouts. Grok é fail-open e sandbox off por padrão — você é mais defensivo.

### Onde você perde feio hoje (gaps vs concorrentes)

| Gap | Concorrente que faz | Severidade |
|---|---|---|
| `nullain.toml` **não é lido** pelo daemon (`main.py` chama `load_settings()` sem path) | Todos carregam config | Crítica — sua config documentada é morta |
| Fluxo de permissão **ASK não implementado** (ASK vira ALLOW silencioso) | Claude (6 modos), Gemini (PolicyEngine), Grok (allow/deny) | Alta — furo de segurança |
| `SpecValidator.verify` = coincidência de keyword, não verificação real | — | Alta — Verify é cosmético |
| `ContextManager.compact` = bookkeeping textual, **sem sumarização LLM** | Gemini (sumariza), Claude (sumariza + reinjeta CLAUDE.md) | Alta — contexto degrada em trajetórias longas |
| Sem **subagentes / multi-agent** | Claude (subagent bg + worktree), Gemini (LocalAgentExecutor), Grok (spawn_subagent) | Alta — perde isolamento de contexto |
| Sem **hooks** (PreToolUse/Stop/PreCompact...) | Claude (5 tipos de handler, exit-2-blocks), Gemini, Grok | Média-alta — perde extensibilidade |
| Sem **MCP** | Todos os 3 | Média-alta — perde ecossistema de tools |
| `IntentParser` = heurística keyword; `classifier_model` **morto** | Grok `--effort`, Gemini routing | Média — router não cumpre o que promete |
| Sem **plan mode como modo de permissão** | Claude (`plan` mode) | Média |
| Sem **skills / slash commands / plugins** | Claude (skills+plugins+marketplace), Gemini (extensions), Grok (plugin marketplace) | Média |
| Sem **memória persistente** (`CLAUDE.md`/`MEMORY.md`/`AGENTS.md` ingestion) | Claude (CLAUDE.md + auto memory), Grok (lê AGENTS.md + CLAUDE.md), Gemini (GEMINI.md) | Média — você lê AGENTS.md/SOUL mas sem re-injeção pós-compaction |
| Telemetry = só structlog logs | Claude (OTEL spans: interaction→llm_request→tool), Gemini (18 eventos tipados) | Média |
| Sem **dispatch paralelo vs sequencial** por anotação readOnly | Claude (readOnlyHint → concorrente) | Média |
| `EventStore.EVENT_CLASS_MAP` **falta `StreamDeltaEvent`** | — | Baixa |
| `escalate_tier` = **dead code** | — | Baixa |
| Sem teste E2E do daemon NDJSON (stdin→stdout) | — | Média — cobertura falsa |

---

## PARTE 2 — Comparativo direto por dimensão

### Loop do agente
- **Claude:** turn-based, sem yield mid-cycle; termina em no-tool-call ou `max_turns`/`max_budget_usd`; subtipos de fim definidos (`success`/`error_max_turns`/...).
- **Gemini:** `GeminiClient.sendMessageStream` + `Turn` + `Scheduler` state machine (parallel/sequential via `wait_for_previous`); loop detection.
- **Grok:** Rust, `xai-grok-shell` runtime; max-turns/`complete_task`.
- **Nullain:** ReAct loop c/ Plan→Act→Verify, budget tokens, timeout via Clock, self-correction (3 retries), max_steps=25. **Paridade razoável no loop básico.** Faltam: subtipos de fim padronizados, loop detection, dispatch paralelo/seq.

### Toolset
- **Claude:** Read/Edit/Write/Glob/Grep/Bash/PowerShell/Monitor/WebFetch/WebSearch/Agent/Task*/Skill/Workflow/LSP/Plan/Worktree/Cron + MCP.
- **Gemini:** run_shell_command, glob, grep_search, list_directory, read_file, read_many_files, replace, write_file, google_web_search, web_fetch, save_memory, ask_user, write_todos, plan_mode, complete_task.
- **Grok:** terminal, file edit, search, VCS, checkpoints, web search.
- **Nullain:** bash, git_status, git_diff, git_commit, read_file, write_file, edit_file, grep. **Faltam:** glob/list_directory, web, ask_user, todo, monitor, LSP, plan-mode tool, subagent tool. Glob sozinho já é gap grande — Grep sem Glob custa caro.

### Contexto / compaction
- **Claude:** auto-sumariza, emite `compact_boundary`, **reinjeta CLAUDE.md pós-compaction**, thrashing protection (para após N tentativas).
- **Gemini:** `ChatCompressed` event, sumariza turno antigos.
- **Grok:** compaction via hook.
- **Nullain:** `ContextManager` mantém últimos 4 eventos + "summary" textual (não-sumarizado); `reinject_instructions` a cada 5 passos. **Perde:** sumarização real, reinjeção de regras pós-compaction, thrashing protection.

### Subagentes
- **Claude:** subagent = contexto fresco, tools scoped, summary-only, **bg por padrão**, fork mode, `disallowedTools` vence `tools`. Agent teams com `SendMessage`.
- **Gemini:** `LocalAgentExecutor` até `complete_task`, grace-period de recovery.
- **Grok:** `spawn_subagent`, depth cap 1, isolation worktree, `resume_from`.
- **Nullain:** **nenhum.** Gap estrutural.

### Permissões / sandbox
- **Claude:** 6 modos, regra uniforme `ToolName(spec)`, deny>ask>allow, `canUseTool` callback, sandbox OS-level separado de permissão.
- **Gemini:** PolicyEngine + sandbox Seatbelt/Podman/runsc/LXC/icacls.
- **Grok:** `--allow`/`--deny`, `--yolo`, sandbox profiles; **fail-open, sandbox off por padrão**.
- **Nullain:** `PermissionPolicy`/`PermissionLevel` (DENY/ASK/ALLOW) mas **ASK vira ALLOW** no `ToolRegistry.execute`. Sem sandbox OS. **Crítico.**

### Hooks / extensibilidade
- **Claude:** 5 handler types (command/http/mcp_tool/prompt/agent), exit-2-blocks, `additionalContext` (10k cap), `if` filter.
- **Gemini:** hooks lifecycle.
- **Grok:** before/after tool/prompt/permission/subagent/compaction.
- **Nullain:** **nenhum hook.** `EventBus` pub/sub existe mas é interno, não é hook lifecycle configurável.

### MCP
- **Claude:** stdio/http/sse/ws, **tool search (schemas deferidos)**, scopes project/user/managed.
- **Gemini:** `McpClientManager`.
- **Grok:** `grok mcp list/add/remove/doctor`.
- **Nullain:** **nenhum.**

### Memória
- **Claude:** CLAUDE.md (hierárquico, re-injetado pós-compaction) + auto-memory (`MEMORY.md` 200 linhas/25KB + topic files).
- **Grok:** lê `AGENTS.md`/`CLAUDE.md`/`.claude/rules/` com zero config.
- **Gemini:** `GEMINI.md` + `.gemini/skills/`.
- **Nullain:** lê `AGENTS.md`/`SOUL.md` no `PromptAssembler`, mas **sem reinjeção pós-compaction** e sem auto-memory persistente. `EpisodicMemory` é few-shot, não regras.

### Telemetria
- **Claude:** OTEL spans hierárquicos (interaction→llm_request/tool→execution), `ttft_ms`, cost attribution (model×query_source×effort×agent), correlation IDs.
- **Gemini:** 18 eventos tipados, dois event buses.
- **Nullain:** structlog + EventBus. **Sem spans, sem cost tracking, sem ttft.**

---

## PARTE 3 — Plano de Correção (nível sênior)

Priorizado por impacto/risco. Cada item: o quê, onde, tempo estimado.

### P0 — Correções de corretude/segurança (semana 1, ~2 dias)

1. **Carregar `nullain.toml` no daemon.** `nullain-agentd/src/nullain_agentd/main.py` chama `load_settings()` sem path → passa `load_settings("nullain.toml")` (ou path via env `NULLAIN_CONFIG`). Senão sua config é mentira. **~20 min.**
2. **Implementar fluxo ASK ou remover a mentira.** `tools/registry.py:execute` só bloqueia em DENY. Opção A (recomendada): daemon emite `PermissionRequestPayload` no protocolo e aguarda `PermissionResponsePayload` antes de executar. Opção B: remove `ASK` do `PermissionLevel`, documenta ALLOW-only, mantém DENY. **A: ~3-4h · B: ~30 min.**
3. **Adicionar `StreamDeltaEvent` ao `EVENT_CLASS_MAP`** em `events/store.py` ou documentar como não-persistível com `assert`. **~15 min.**
4. **Deletar `Router.escalate_tier`** (dead code) ou wire num retry-after-failure. **~10 min deletar.**
5. **Deletar `raise ProviderError("Failed to complete request")`** inalcançável em `ollama.py:generate`. **~5 min.**
6. **Corrigir contagem de `step` off-by-one** em `loop.py` (incrementado antes da chamada). **~20 min.**
7. **Parar de engolir erro sem `ErrorEvent`** em `_retrieve_few_shot` e `_record_trajectory` (viola AGENTS.md regra 5). **~30 min.**

### P1 — Honestidade arquitetural (semana 1-2, ~3 dias)

8. **Tornar `SpecValidator.verify` real.** Substitua coincidência de keyword por: (a) verificação só via `verification_commands` (saída de comando é verdade), e (b) critérios opcionais via LLM-judge (chamada isolada ao provider). Remove o "tudo passa" atual. **~4-6h.**
9. **Tornar `ContextManager.compact` real OU renomear.** Ou implementa sumarização LLM (chamada ao provider com prompt de sumário) ou documenta como "compaction estrutural apenas" e adiciona `summarize_with(provider)` como passo seguinte. **Sumarização: ~1 dia.**
10. **Reinjeção de regras pós-compaction.** `Conversation.fold` já insere summary como system msg — adicione re-injeção de `AGENTS.md`/`SOUL.md`/spec ativa após `CompactionEvent` (igual Claude reinjeta CLAUDE.md). **~3h.**
11. **Thrashing protection.** Se compaction não reduz tokens abaixo do threshold após 2-3 tentativas, levanta `ContextWindowExhaustedError` em vez de loopar. **~1h.**
12. **Decidir `classifier_model`: usar ou matar.** Ou `IntentParser` chama o classifier_model (LLM leve) para intents ambíguas, ou remove o campo morto de `RouterConfig`/`nullain.toml.example`. **~2h implementar / 5 min remover.**

### P2 — Features de paridade (semana 2-3, ~1 semana)

13. **Tools faltantes:** `glob`, `list_directory`, `web_fetch` (sem extraction model — retorna markdown cru), `ask_user` (interativo no daemon). **~1 dia.**
14. **Dispatch paralelo vs sequencial.** Adicione `read_only: bool` no `ToolSpec`/`@tool`; no `AgentLoop._execute_tools`, rode read-only em `asyncio.gather`. **~3h.**
15. **Subtipos de fim padronizados.** `ResultMessage`-equivalente: `success`/`max_steps`/`budget`/`timeout`/`error`. Hoje você emite `MaxStepsExceeded` mas sem contrato. Defina um `RunResult` Pydantic. **~2h.**
16. **Loop detection.** Hash do (tool_name, args) por passo; se repetir 3x consecutivas, injeta correção ou para. Gemini faz. **~3h.**
17. **Subagentes (mínimo viável).** `Agent.spawn(prompt, tools, model)` = novo `AgentLoop` com contexto fresco, tools scoped, retorna só texto. Sem worktree/bg no v1. **~1 dia.**

### P3 — Plataforma (semana 3-4, contínuo)

18. **Hooks.** Lifecycle `pre_tool`/`post_tool`/`stop`/`pre_compact` no EventBus, configurável via `nullain.toml`, handler `command` (exit 2 bloqueia). Espelha convenção do Claude. **~2 dias.**
19. **MCP client.** `stdio` + `http` transport, `McpClient` que registra tools no `ToolRegistry`. Schema deferido (tool search) é P4. **~2 dias.**
20. **Telemetria OTEL.** Spans: `interaction → llm_request (model, tokens, ttft_ms, cost) → tool (execution/blocked)`. Cost tracking por (model, tier, agent). `opentelemetry-sdk` dep. **~1 dia.**
21. **Memória persistente.** `AGENTS.md` re-injetado pós-compaction (cobre no item 9/10); auto-memory `MEMORY.md` com cap 25KB + topic files. **~1 dia.**
22. **Plano de testes:** (a) teste E2E do daemon NDJSON (pipe stdin→stdout), (b) teste do endpoint nativo Ollama (`/api/chat`), (c) teste `load_settings` com TOML real, (d) teste do `IntentParser` HIGH/LOW edge cases. **~1 dia.**

### P4 — Diferenciação (mês 2+, após paridade)

23. **Sandbox fail-closed por padrão** (landlock/Seatbelt/firejail) — ganha onde Grok é fail-open.
24. **Authority-intersection law p/ subagentes** (child authority = intersection(parent-delegation, child-def, context, policy)) — nenhum concorrente prova isso.
25. **Plugin signing + SBOM + capability manifests.**
26. **Tool search (schemas deferidos) p/ escalar MCP.**
27. **Workflow orchestrator** (script JS determinístico sobre subagentes) — equivalente ao `Workflow` do Claude.

---

## PARTE 4 — Roadmap resumido (para colar no wall)

```
Semana 1  (P0+P1 parcial):  config lê nullain.toml · fluxo ASK · dead code · off-by-one ·
                           SpecValidator real · compaction real · reinjeção pós-compaction
Semana 2  (P1+P2):         thrashing protection · classifier decision · tools faltantes ·
                           dispatch paralelo · RunResult · loop detection
Semana 3  (P2+P3):         subagentes MVP · hooks · MCP client
Semana 4  (P3):            telemetria OTEL · memória persistente · plano de testes
Mês 2     (P4):            sandbox fail-closed · authority law · plugins · tool search · workflow
```

**Tempo total até paridade competitiva (P0-P3):** ~4 semanas para 1 dev sênior full-time.

---

## Anexo A — Itens de dead code / bugs confirmados no código

- `router/router.py`: `escalate_tier` sem callers.
- `llm/ollama.py`: `raise ProviderError("Failed to complete request")` inalcançável.
- `llm/response_models.py`: `OpenAIChoice` não exportado (usado interno — não é dead, só não está em `__all__`).
- `router/intent.py` + `config/settings.py` + `nullain.toml.example`: `classifier_model` configurado mas nunca usado.
- `agent/loop.py`: `step` incrementado antes da chamada → contagem off-by-one em paths de timeout/budget.
- `events/store.py`: `EVENT_CLASS_MAP` omite `StreamDeltaEvent`.
- `protocol/types.py`: `PermissionRequestPayload`/`PermissionResponsePayload` definidos mas daemon não emite/consome.
- `nullain-agentd/main.py`: `load_settings()` sem path → `nullain.toml` ignorado.
- `context/manager.py`: `compact` produz string de bookkeeping, não sumário.
- `agent/spec.py`: `verify` aceitação = coincidência de substring "error"/"fail".
- `agent/loop.py`: `_retrieve_few_shot` e `_record_trajectory` engolem `Exception` sem `ErrorEvent`.
- `agent/loop.py`: `_execute_tools` classifica erro por prefixo string (`startswith("Error:")`) — frágil.
- `agent/loop.py`: `_check_budget_and_compact` refaz fold completo a cada iteração → O(n²) em trajetória longa.

## Anexo B — Onde o Nullain pode GANHAR dos 3 (não só empatar)

1. **Único harness open-source-native com Ollama** — privacidade/air-gap/custo zero. Ninguém mais entrega.
2. **Fail-closed sandbox default** — Grok é fail-open; você pode ser o seguro por padrão.
3. **Compaction inspecionável + episodic memory** — concorrentes são opacos.
4. **Plan/Act/Verify com TaskSpec validável** — contrato tipado que ninguém tem.
5. **ModelRouter + CircuitBreaker explícito** — roteamento por intent documentado.
6. **Leitura multi-vendor de memória** (`AGENTS.md` + `CLAUDE.md` + `GEMINI.md`) — compat universal.

Fontes: docs oficiais Claude Code (code.claude.com/docs), Gemini CLI (github.com/google-gemini/gemini-cli), Grok Build (github.com/xai-org/grok-build + teardown OpenAgentsInc).