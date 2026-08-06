# PROMPT MESTRE — Fases 1 a 4 de Correção do Nullain Agent SDK

> **Como usar:** este documento é o prompt de trabalho para o agente de codificação.
> Entregue **uma fase por sessão**, na ordem M8 → M9 → M10 → M11. Nunca peça "faz tudo".
> Cada fase termina com `make check` verde + commit convencional.

---

## Parte 0 — Contexto obrigatório (leia antes de codar)

Você está trabalhando no `nullain-agent-sdk`, um monorepo Python (uv workspace) com três pacotes:

- `nullain-sdk/src/nullain/` — o núcleo (loop, router, contexto, tools, memória, mcp, plugins, workflow)
- `nullain-tools/src/nullain_tools/` — as tools embutidas (filesystem, bash, git, web, memory, search, ask_user)
- `nullain-agentd/src/nullain_agentd/` — o daemon NDJSON sobre stdio

**Leia obrigatoriamente antes de escrever qualquer linha:**

1. `AGENTS.md` — as 11 regras vinculantes. Elas não são sugestões.
2. `SOUL.md` — identidade do agente.
3. `docs/design/COMPARATIVO_E_PLANO_CORRECAO.md` — o que já foi feito (P0–P4 concluídos) e o Anexo A de dívida técnica.
4. `NULLAIN_SDK_PLANO_ENGENHARIA.md` — princípios de arquitetura.

**Estado atual verificado (não re-investigue, é fato):**

- 197 testes passam, 10 skipped (`uv run pytest`).
- P0 a P4 do comparativo estão **realmente implementados**: sandbox fail-closed, Authority-intersection law, plugin signing/SBOM, tool search deferido, workflow orchestrator, hooks, MCP, telemetria OTEL, memória persistente.
- O que falta **não é** mais arquitetura de agente. É: (a) camada de edição de código, (b) API pública usável, (c) robustez do núcleo, (d) diferenciação.

**Regras de processo (de `AGENTS.md`, reforçadas):**

- Antes de codar cada fase, escreva um mini-design de 10–20 linhas em `docs/design/MX.md` e **mostre para revisão antes de implementar**.
- Testes junto com o código, nunca depois. 100% offline (fakes + respx). Todo bugfix ganha teste de regressão.
- `pyright` strict no núcleo. `ruff` com `E,F,W,I,N,UP,B,S,A,C4,SIM,RUF`. Sem `Any` na API pública. Docstring em todo símbolo público.
- Subprocess **sempre** por lista de args. `shell=True` é proibido.
- Todo path passa por `resolve()` + `is_relative_to(workspace_root)` com symlinks resolvidos **antes** da checagem.
- Proibido `except Exception: pass`. Todo erro engolido precisa de log estruturado + evento de erro.
- Sem dependência nova sem justificativa explícita. Proibidos: langchain/langgraph/crewai, requests, eval/exec dinâmico, pickle para dados externos.
- Ambiguidade de requisito: **pergunte antes**, não invente escopo.
- `make check` verde é a definition of done. Nunca desative teste, regra de lint ou checagem de tipo para "fazer passar".

**Comandos:**

```
make check    # ruff + pyright + pytest  ← definition of done
make test     # pytest
make format   # ruff --fix + format
make schema   # exporta JSON Schema do protocolo
uv run ...    # sempre via uv, nunca pip global
```

---

## FASE 1 — M8: Camada de edição de código

> **Objetivo:** transformar o toolset de "agente genérico" em "agente de codificação".
> Hoje as tools de arquivo não escalam para arquivos reais nem editam com segurança.
> **Estimativa:** ~1 semana. **Design doc:** `docs/design/M8.md`.

### Contexto do problema

Arquivo alvo principal: `nullain-tools/src/nullain_tools/filesystem.py`.

Problemas concretos, verificados no código:

- `read_file` lê o arquivo inteiro (`target.read_text()`), sem `offset`/`limit` e sem números de linha. Um arquivo de 3000 linhas estoura a janela de contexto e o modelo não consegue referenciar linhas.
- `edit_file` faz `content.replace(old_str, new_str, 1)` — substitui a **primeira** ocorrência sem verificar se `old_str` é único no arquivo. Se aparecer 5 vezes, edita silenciosamente a errada. É a fonte de bug número 1 em agentes de código.
- `grep` percorre `rglob("*")` em Python puro, lê cada arquivo inteiro, e corta em 50 matches sem avisar que truncou. Ordens de magnitude mais lento que ripgrep e sem contexto de linhas vizinhas.
- Não existe edição multi-hunk atômica: um refactor com 4 alterações vira 4 chamadas independentes, e falhar na terceira deixa o arquivo em estado inconsistente.
- Não existe rastreio de plano multi-passo. Em tarefa longa o modelo perde o fio do que já fez.

### Tarefas

**1.1 — `read_file` com paginação e numeração**

- Adicionar parâmetros `offset: int = 0` e `limit: int = 2000`.
- Retornar as linhas no formato `cat -n` (número de linha, tab, conteúdo), para que o modelo possa referenciar `arquivo.py:42`.
- Quando o arquivo excede `limit`, informar explicitamente no retorno quantas linhas restam e como pedir a próxima fatia. Truncamento silencioso é proibido.
- Truncar linhas individuais absurdamente longas (ex.: minificados) com marcador explícito.
- Manter `read_only=True` e `requires={Capability.READ}`.

**1.2 — `edit_file` seguro**

- Se `old_str` aparecer mais de uma vez e `replace_all` for `False`, **falhar** com mensagem que diz quantas ocorrências foram encontradas e sugere ampliar o contexto do `old_str`. Nunca adivinhar.
- Adicionar parâmetro `replace_all: bool = False`.
- Se `old_str == new_str`, falhar (edição no-op é quase sempre bug do modelo).
- Rastrear quais arquivos foram lidos na sessão e exigir leitura prévia antes de editar. Onde guardar esse estado é decisão sua — proponha no design doc. Sugestão: um `FileAccessTracker` injetado nas tools, criado por sessão no daemon.
- Retornar um trecho do resultado (algumas linhas ao redor da edição, numeradas) para o modelo confirmar visualmente o que mudou.

**1.3 — `multi_edit` atômico**

- Nova tool: recebe `path` e uma lista de edições `[{old_str, new_str, replace_all?}]`.
- Aplica **todas em sequência sobre o conteúdo em memória**; só escreve no disco se todas passarem. Se qualquer uma falhar, nada é escrito e o erro diz qual edição falhou e por quê.
- Cada edição opera sobre o resultado da anterior (permite edições encadeadas).
- `requires={Capability.WRITE}`.

**1.4 — `grep` com ripgrep**

- Detectar `rg` no PATH e usá-lo quando disponível; fallback para a implementação Python atual quando ausente. O comportamento observável (formato de saída) deve ser o mesmo nos dois caminhos.
- Chamar `rg` **por lista de args** (regra de segurança 6 do AGENTS.md), nunca por shell.
- Novos parâmetros: `output_mode` (`content` | `files_with_matches` | `count`), `context_lines` (equivalente a `-C`), `glob_filter`, `case_insensitive`, `head_limit`.
- Quando truncar, dizer explicitamente que truncou e quantos resultados existiam.

**1.5 — `todo_write` e evento de progresso**

- Nova tool que recebe uma lista de itens `{content, status}` com status em `pending` | `in_progress` | `completed`.
- Validar: no máximo **um** item `in_progress` por vez. Mais de um é erro.
- Emitir um novo `TodoEvent` (frozen, Pydantic) no `EventBus` a cada atualização, para que o daemon possa espelhar o progresso no cliente.
- Registrar o evento em `events/types.py`, exportar em `events/__init__.py`, e **adicionar ao `EVENT_CLASS_MAP` em `events/store.py`** (o Anexo A já registra que `StreamDeltaEvent` foi esquecido lá — não repita o erro).
- `requires=frozenset()` (capability-neutral) e `read_only=False`.

### Definition of done da Fase 1

- Todas as tools novas/alteradas registradas em `register_default_tools`.
- Testes offline cobrindo: paginação e numeração, match ambíguo rejeitado, edição no-op rejeitada, rollback do `multi_edit`, fallback do grep quando `rg` ausente, invariante de um único `in_progress`.
- `TodoEvent` com teste de round-trip de serialização no `EventStore`.
- `make check` verde.
- Commit: `feat(m8): coding-grade file tools (paged read, safe edit, multi_edit, rg grep, todo)`.

---

## FASE 2 — M9: API pública e CLI

> **Objetivo:** tornar o SDK utilizável por alguém que não escreveu o SDK.
> Hoje montar um agente exige instanciar `AgentLoop` + `ToolRegistry` + `Provider` + `PermissionPolicy` na mão.
> **Estimativa:** ~4 dias. **Design doc:** `docs/design/M9.md`.

### Contexto do problema

- `nullain-sdk/src/nullain/cli.py` tem 42 linhas e implementa apenas `version` e `doctor`. Não roda um agente.
- `nullain-sdk/src/nullain/__init__.py` exporta **106 símbolos** em `__all__`. Isso viola frontalmente a regra 11 do AGENTS.md ("API pública mínima") e congela toda a superfície interna sob SemVer.
- Existe um único exemplo executável (`examples/00_llm_smoke.py`).
- Não há `docs/` de uso nem quickstart.

### Tarefas

**2.1 — Fachada `Agent`**

- Nova classe `Agent` em `nullain/agent/facade.py`, a porta de entrada de 90% dos usuários.
- Construtor com defaults seguros: monta `OllamaCloudProvider`, `ToolRegistry` com as tools padrão, `PermissionPolicy` fail-closed, `ModelRouter` a partir de `nullain.toml`, sandbox selecionado pela plataforma.
- Métodos: `async run(prompt) -> RunResult`, `async stream(prompt)` (async iterator de eventos), e uma fachada **síncrona** fina `run_sync(prompt)` para scripts (princípio 5 do plano de engenharia: "async first, sync como fachada").
- `Agent.from_settings(settings)` e `Agent.from_config(path)` como construtores alternativos.
- Não duplicar lógica: a fachada **monta** colaboradores, nunca reimplementa o loop.

**2.2 — CLI real**

Substituir o `argparse` atual. Escolha de framework: `typer` ou `argparse` estruturado — decida no design doc justificando (regra 10: sem dependência nova sem justificativa).

Comandos mínimos:

- `nullain run "<prompt>"` — execução única, imprime o resultado. Flags: `--model`, `--workspace`, `--max-steps`, `--json` (saída estruturada para pipe).
- `nullain chat` — sessão interativa multi-turno com streaming e aprovação de permissão no terminal (o callback `ASK` conectado ao TTY).
- `nullain doctor` — expandir o atual: checar provider alcançável, disponibilidade do sandbox na plataforma, `nullain.toml` válido, servidores MCP inicializáveis, `rg` presente.
- `nullain mcp list|add|remove` — gestão dos servidores MCP declarados no `nullain.toml`.
- `nullain version` — manter.

Requisitos: exit codes corretos (0 sucesso, não-zero por classe de falha), `--json` produzindo NDJSON consumível por script, e nenhum segredo impresso (regra 6: redação de secrets).

**2.3 — Enxugar a API pública**

- Reduzir `__all__` de 106 para **~25 símbolos**: `Agent`, `AgentLoop`, `RunResult`, `Conversation`, `EventBus`, `tool`, `ToolRegistry`, `LLMProvider`, `OllamaCloudProvider`, `NullainSettings`, `load_settings`, a hierarquia de erros de topo, e os eventos que o consumidor realmente observa.
- Tudo o mais continua **importável pelo caminho completo do módulo** (`from nullain.plugins import PluginLoader`), apenas sai do `__all__` de topo. Não quebre imports existentes: isto é redução de superfície declarada, não remoção de código.
- Documentar a política em `docs/api-stability.md`: o que está sob SemVer e o que é interno.

**2.4 — Documentação e exemplos**

- `docs/quickstart.md`: do zero a um agente rodando, com comandos copiáveis que funcionam de verdade.
- `docs/configuration.md`: `nullain.toml` explicado seção por seção.
- `docs/tools.md`: catálogo das tools embutidas com assinatura e capabilities.
- `docs/architecture.md`: o diagrama de camadas e a explicação do pipeline Intent → Route → Plan → Act → Verify → Memory.
- Exemplos executáveis em `examples/`: `01_basic_agent.py`, `02_custom_tool.py`, `03_subagent_authority.py`, `04_workflow.py`, `05_mcp_server.py`.
- Atualizar o `README.md` da raiz para apontar para eles.

### Definition of done da Fase 2

- `uv run nullain run "liste os arquivos python"` funciona de ponta a ponta contra um provider real.
- Testes de CLI com provider fake (sem rede), cobrindo exit codes e saída `--json`.
- Todos os exemplos rodam (adicionar um teste de smoke que os importa/executa em modo dry-run).
- `make check` verde.
- Commit: `feat(m9): Agent facade, real CLI, minimal public API, docs`.

---

## FASE 3 — M10: Robustez do núcleo

> **Objetivo:** eliminar as fragilidades estruturais que só aparecem em produção.
> **Estimativa:** ~4 dias. **Design doc:** `docs/design/M10.md`.

### Contexto do problema

Cinco defeitos concretos, todos verificados no código:

1. **Detecção de erro por prefixo de string.** `agent/loop.py:60-65` define `ERROR_OUTPUT_PREFIXES` e classifica o resultado de uma tool como falha via `res_output.startswith(prefix)`. Consequência: ler um arquivo cujo conteúdo começa com `Error:` marca a leitura como falhada e dispara auto-correção sem motivo. O Anexo A já flagra isso como frágil.
2. **Sem cancelamento.** Não existe `CancellationToken` em lugar nenhum. Um run em andamento não pode ser abortado — nem pelo usuário no CLI, nem pelo daemon quando o cliente desconecta.
3. **O(n²) na trajetória.** `_check_budget_and_compact` chama `Conversation.fold(sess_id, accumulated_events)` sobre a lista inteira a cada passo, e `_build_messages` faz o mesmo logo em seguida. Dois folds completos por passo. Anexo A já registra.
4. **Streaming de tool-call sem merge de deltas.** Em `llm/ollama.py:stream`, cada chunk é parseado isoladamente e os `tool_calls` são acumulados por `extend`. Provedores OpenAI-compat fragmentam os argumentos de uma tool call entre múltiplos chunks (`arguments` chega em pedaços). Hoje isso produz tool calls truncadas ou duplicadas.
5. **Estimativa de token por `len/4`.** `context/manager.py:54` usa `len(text) // 4`. Em código o erro passa de 30%, então a compaction dispara cedo demais (desperdiça janela) ou tarde demais (estoura).

### Tarefas

**3.1 — `ToolResult` tipado**

- Modelo Pydantic frozen `ToolResult` com pelo menos `output: str`, `is_error: bool`, `error_type: str | None`, e metadados opcionais.
- `RegisteredTool.execute` e `ToolRegistry.execute` passam a retornar `ToolResult`, não `str`.
- As tools embutidas retornam erro **estruturalmente** (levantando `ToolError` ou retornando `ToolResult(is_error=True)`), não por prefixo de string.
- Remover `ERROR_OUTPUT_PREFIXES` do loop.
- **Compatibilidade:** manter um caminho que aceita tools legadas que retornam `str` (envolver em `ToolResult(output=..., is_error=False)`), para não quebrar plugins e MCP. Documente essa ponte.
- `SpecValidator.verify` usa `BASH_NONZERO_PREFIX` para detectar falha de comando — migrar também para o campo estruturado.

**3.2 — Cancelamento**

- `CancellationToken` (ou `anyio.CancelScope` — decida e justifique no design doc) propagado do `AgentLoop.run*` para: o provider (abortar request HTTP em voo), a execução de tools (matar subprocess), e o `spawn` de subagentes (cancelar filhos em cascata).
- O daemon cancela o run quando o cliente envia `session.cancel` ou fecha o stdin.
- Um run cancelado retorna `RunResult` com status `cancelled` (novo membro de `RunStatus`), não uma exceção crua.
- Garantir limpeza: subprocess mortos, streams fechados, sem tasks órfãs.

**3.3 — Fold incremental**

- Cachear o `ConversationState` derivado e atualizá-lo incrementalmente conforme eventos são anexados, em vez de refazer o fold do zero.
- Invalidar o cache corretamente em `CompactionEvent` (que remove eventos do histórico efetivo).
- Preservar a semântica pura: `fold` continua sendo uma função pura sobre a sequência; o cache é uma otimização **verificável**. Adicionar um teste de propriedade (hypothesis) provando que o resultado incremental é idêntico ao fold completo para qualquer sequência de eventos.
- Eliminar o fold duplicado entre `_check_budget_and_compact` e `_build_messages` no mesmo passo.

**3.4 — Merge de deltas de tool call no streaming**

- Implementar acumulação por índice/id: fragmentos de `arguments` da mesma tool call são concatenados, e a call só é considerada completa quando o parse do JSON tem sucesso ou o stream sinaliza fim.
- Cobrir os dois `endpoint_type` (`v1` OpenAI-compat e `native` Ollama).
- Testes com um stream fake que fragmenta os argumentos em 3+ chunks no meio de um token JSON.

**3.5 — Contagem de tokens real**

- Substituir `len//4` por contagem real. Ordem de preferência: (a) usar o `usage` que o provider devolve quando disponível; (b) um tokenizer local. Se optar por `tiktoken`, justifique a dependência no design doc conforme a regra 10 — e mantenha a heurística como fallback quando o tokenizer não estiver instalado.
- Manter a interface `estimate_tokens` / `estimate_context_tokens` estável; a mudança é de implementação.
- Teste comparando a estimativa contra o `usage` real de uma resposta fake, garantindo margem de erro aceitável.

### Definition of done da Fase 3

- Testes de regressão para cada um dos 5 defeitos (o teste deve falhar no código antigo).
- Property test do fold incremental.
- `make check` verde.
- Commit: `fix(m10): typed ToolResult, cancellation, incremental fold, stream delta merge, real token count`.

---

## FASE 4 — M11: Diferenciação

> **Objetivo:** entregar o que os concorrentes não fazem bem.
> **Estimativa:** ~1 semana. **Design doc:** `docs/design/M11.md`.

### Tarefas

**4.1 — Checkpoints e undo**

- Antes de cada operação de escrita (`write_file`, `edit_file`, `multi_edit`), tirar um snapshot restaurável do estado do arquivo.
- Mecanismo: preferir git (stash/objeto/índice separado) quando o workspace é um repositório; fallback para cópia em `.nullain/checkpoints/` quando não é. **Não** poluir o histórico de commits do usuário nem alterar o índice dele — decida a abordagem exata no design doc e justifique.
- Tool `undo` (ou comando `nullain undo`) que restaura o checkpoint anterior.
- Retenção limitada (últimos N checkpoints ou N MB), com eviction dos mais antigos.
- Grok e Claude têm isso; um agente que escreve sem rollback não é usável em produção.

**4.2 — Cliente LSP**

- Cliente LSP mínimo sobre stdio (o padrão é JSON-RPC — a mesma família do MCP que já está implementado; reaproveite o que fizer sentido de `mcp/transport.py`, sem acoplar os dois).
- Tools: `diagnostics(path)` (erros/avisos do arquivo), `goto_definition(path, line, col)`, `find_references(path, line, col)`, `hover(path, line, col)`.
- Configuração dos servidores por linguagem em `nullain.toml` (seção `[lsp.servers.<lang>]`), no mesmo formato do `[mcp.servers.*]`.
- Fail-soft: servidor LSP indisponível é logado e ignorado, nunca derruba a sessão (mesmo padrão de `_load_mcp_clients`).
- **Este é o maior diferencial técnico da lista** — nenhum dos três concorrentes expõe diagnósticos ao modelo de forma decente.

**4.3 — `IntentParser` com classifier real**

- O campo `classifier_model` existe no `RouterConfig` e no `nullain.toml.example` mas nunca é usado — é dead code desde o comparativo original.
- Manter as heurísticas determinísticas como primeiro estágio (barato e previsível). Quando o prompt não bate nenhuma heurística com confiança, chamar o `classifier_model` (tier `fast`) para classificar intent + complexidade.
- O classificador é uma chamada isolada e barata, com timeout curto e fallback para a heurística atual em qualquer falha.
- Cachear classificações por hash do prompt dentro da sessão.
- Teste: prompt ambíguo aciona o classificador; classificador indisponível cai na heurística sem erro.

**4.4 — Subagente em worktree isolado**

- `AgentLoop.spawn` hoje é síncrono in-place e o filho escreve no mesmo diretório do pai. Dois subagentes concorrentes corrompem o workspace.
- Adicionar modo `isolation="worktree"`: cria um `git worktree` temporário, o filho trabalha lá, e ao terminar o resultado é integrado (ou descartado se nada mudou).
- Limpeza garantida do worktree mesmo em falha ou cancelamento (use o `CancellationToken` da Fase 3).
- Preservar integralmente a lei de intersecção de autoridade do P4.24 — o isolamento é **adicional** ao gate de capability, nunca substituto.
- Manter `isolation=None` (in-place) como default, backward-compatible.

### Definition of done da Fase 4

- Testes offline: checkpoint/restore round-trip, LSP com servidor fake, classificador com provider fake + fallback, worktree criado e limpo (incluindo no caminho de falha).
- Documentação de cada feature nova em `docs/`.
- `make check` verde.
- Commit: `feat(m11): checkpoints, LSP client, real intent classifier, worktree subagents`.

---

## Parte final — Ordem de execução e critérios de aceite globais

```
M8  (Fase 1) → toolset de codificação      ~1 semana
M9  (Fase 2) → API pública + CLI           ~4 dias
M10 (Fase 3) → robustez do núcleo          ~4 dias
M11 (Fase 4) → diferenciação               ~1 semana
                                    total: ~4 semanas
```

**Por que esta ordem:** M8 antes de M9 porque não faz sentido expor uma CLI polida de um agente que ainda não edita código com segurança. M10 depois de M9 porque o `CancellationToken` precisa da CLI e do daemon como consumidores reais para ser desenhado certo. M11 por último porque checkpoints dependem das tools de escrita da M8 e worktrees dependem do cancelamento da M10.

**Critérios de aceite que valem para todas as fases:**

1. `make check` verde ao final de cada fase — sem exceções, sem `# type: ignore` novo sem comentário justificando, sem `noqa` sem justificativa.
2. Nenhuma regressão: os 197 testes atuais continuam passando.
3. Cada símbolo público novo tem docstring.
4. Cada fase entrega seu `docs/design/MX.md` **antes** da implementação, para revisão.
5. Nenhuma dependência nova sem justificativa escrita no design doc.
6. Nada de escopo inventado. Se um requisito estiver ambíguo, pergunte antes de implementar.
7. Commits pequenos e convencionais dentro de cada fase; o commit final da fase usa a mensagem especificada acima.

**O que explicitamente NÃO está no escopo destas 4 fases:** o CLI em Go (cliente do daemon), publicação em PyPI, e qualquer reescrita da arquitetura já validada em P0–P4. Se você achar que algo aí é bloqueante, levante a questão em vez de implementar.
