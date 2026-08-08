<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# @GitHubNão abre Issue, so prepara a resposta pro claude code com essa nova resposta do bge-m3 via FgastEmbed no qdrant free

Plano de Ação — Fusão Nullain Agent SDK ⇄ Nullain Agent (App)
0. Veredito em uma frase
O Nullain Agent (app) tem um AgentLoop completo escondido dentro do LangChain/LangGraph — chat.py:build_agent() reimplementa, com pior isolamento, quase tudo que o SDK já tem: loop de tool-calling, roteamento de modelo, memória, checkpoint, resiliência. A ponte certa é fazer o SDK virar o cérebro (AgentLoop, ToolRegistry, EventBus, ModelRouter, EventStore) e o App virar apenas transporte + tools de domínio (FastAPI/SSE, Composio, ZuckPay, Wavespeed, sandbox, RAG) — plugadas no SDK como RegisteredTools, não como um segundo agente paralelo.

1. O que descobrimos (mapa das duas bases)
SDK (nullain-sdk) — já pronto e reaproveitável
Agent (facade) → AgentLoop — ReAct loop hexagonal, já com max_steps/max_tokens/timeout, EventBus pub/sub, EventStore (SQLite, resume de sessão, repair automático de sessões corrompidas), ModelRouter, PersistentMemory + EpisodicMemory, HookManager.
ToolRegistry + PermissionPolicy (ASK/ALLOW/DENY, fail-closed) + Authority/Capability (isolamento de subagentes).
LLMProvider Protocol — generate/stream/health_check. Adapters hoje: OllamaCloudProvider, OpenAICompatibleProvider (issue \#40) — ou seja, já é multi-provider de verdade, ao contrário do app que só sabe falar ChatOpenAI.
nullain.mcp — client MCP próprio (protocol.py/transport.py/client.py).
nullain.tools.sandbox — sandbox multi-adapter (Landlock, seatbelt, Windows Job, none) já dentro do SDK.
Nada de RAG, nada de vetor, nada de Qdrant/Supabase ainda — área nova.
App (nullain-agent) — 11.665 linhas, LangChain/LangGraph
agent/chat.py (1142L): build_agent() monta um create_agent do LangChain com AgentMiddlewares próprios, ChatOpenAI genérico (qualquer provider "OpenAI-compatible"), checkpointer LangGraph (InMemorySaver/AsyncPostgresSaver). Esse é o verdadeiro concorrente do AgentLoop do SDK.
web/server.py (1917L): FastAPI, mas com agente global singleton por processo (_agent/_settings/_agent_lock, linha 332) — o maior obstáculo estrutural pra injetar um Agent do SDK de forma limpa. Sem DI, sem isolamento multi-tenant real no nível do agente.
Tool ecosystem: Composio (hub de ação — Gmail/Slack/GitHub/Notion/Discord etc.), AnythingLLM (RAG — a ser substituído), Wavespeed (imagem), ZuckPay (pagamento, credenciais por usuário via RunnableConfig do LangChain — mecanismo específico do LangChain que a ponte precisa replicar), Higgsfield/Semrush (MCPs externos com OAuth PKCE). Todas as tools já são StructuredTool + Pydantic args_schema → equivalente funcional a JSON Schema OpenAI-style, compatível de cara com ToolSpec/RegisteredTool do SDK.
agent/security.py: puro, sem acoplamento a framework — sanitização de prompt injection, denylist de bash, criptografia Fernet de segredo por usuário, sanitização de erro. Reaproveitável quase 1:1.
agent/supervisor.py + agent/resilience.py: scaffolding parcialmente morto — supervisor só é chamado 1x (chat.py:936), circuit breaker/retry sem call site confirmado. Fortes candidatos a remoção total, substituídos pelos mecanismos do SDK.
agent/metering.py: quotas definidas mas não aplicadas em nenhum lugar — SDK deveria assumir isso se quisermos quotas reais.
Redis: opcional/fail-open em todo lugar (metering, rate limit, login lockout) — não é dependência dura.
2. Decisão arquitetural — o padrão Bridge
┌─────────────────────────── nullain-agent (App) ───────────────────────────┐
│  web/server.py  →  FastAPI fino: HTTP/SSE, auth, upload, VNC proxy        │
│       │                                                                     │
│       ▼                                                                     │
│  AgentBridge (NOVO — camada de adaptação, vive no app)                    │
│    - 1 instância de nullain.Agent por (tenant/thread), não 1 global        │
│    - traduz SSE do app ⇄ AsyncIterator[BaseEvent] do SDK                  │
│    - injeta user_email/tenant_id como contexto de execução da tool         │
└──────────────────────────────┬──────────────────────────────────────────┘
│ usa
▼
┌─────────────────────────── nullain-sdk (Core) ────────────────────────────┐
│  Agent → AgentLoop → ToolRegistry, ModelRouter, EventBus, EventStore      │
│  ToolRegistry recebe as tools de domínio do App como RegisteredTool:       │
│    - composio_bridge_tools   (via nullain.mcp client, não langchain_mcp)   │
│    - wavespeed_tool, zuckpay_tools, sandbox_tools                          │
│    - rag_search_tool  (NOVO: Cuckoo Filter + RAG Tree, dentro do SDK)      │
│  LLMProvider: OpenAICompatibleProvider apontando pro Ollama Cloud/xAI/etc │
└─────────────────────────────────────────────────────────────────────────┘

Por que Bridge (não "reescrever tudo", não "so importar o SDK direto no chat.py"):
O App mantém sua própria identidade de transporte (FastAPI, SSE, auth, VNC, upload) — coisas de UI/infra que não pertencem ao SDK.
O SDK nunca importa nada do App — a dependência é unidirecional (App depende do SDK, não o contrário), preservando o SDK como produto standalone/publicável.
AgentBridge é a única camada nova que conhece os dois lados. Se o LangGraph for removido amanhã, só o AgentBridge muda.
3. O que sai da stack do App (retirar / substituir)
Componente atualAçãoMotivo
langchain.agents.create_agent + AgentMiddlewares (chat.py)
Remover, substituído por nullain.Agent.stream()
É o próprio ReAct loop — duplicado pelo AgentLoop do SDK, com pior isolamento (1 agente global/processo)
langgraph.checkpoint.* (InMemorySaver/AsyncPostgresSaver)
Remover, substituído por EventStore do SDK (SQLite hoje; ver §5 pra Supabase)
Mesmo propósito (persistir trajetória), mecanismo já existe no SDK com resume + repair automático
agent/mcp_client.py (MultiServerMCPClient do langchain_mcp_adapters)
Adaptar, não remover — vira um nullain.mcp client para Composio, exposto como RegisteredTools
Consolida em um client MCP só; hoje há client MCP duplicado (langchain_mcp_adapters vs nullain.mcp)
agent/anythingllm_tools.py + AnythingLLM MCP/REST
Remover, substituído por Cuckoo Filter + RAG Tree (novo, dentro do SDK)
Pedido explícito do usuário
agent/supervisor.py
Remover
Uso confirmado em 1 call site só; SupervisorDecision/HITL guardrail vira uma feature do PermissionPolicy/Authority do SDK se for necessário
agent/resilience.py (CircuitBreaker/retry_with_backoff)
Remover
Sem call site confirmado; SDK deve ter (ou ganhar) seu próprio retry/backoff no LLMProvider/AgentLoop
web/server.py globais (_agent/_settings/_agent_lock)
Substituir por app.state + resolução por request/tenant
Pré-requisito técnico pra multi-tenant real e pra várias sessões concorrentes usando o SDK corretamente
_ModelRouterMiddleware (chat.py)
Remover, substituído por ModelRouter do SDK
Já existe no SDK; o app só precisa mapear string de UI → model id
Credential injection via RunnableConfig (zuckpay_tools.py)
Adaptar: usar o mecanismo de contexto de execução do ToolRegistry/AgentLoop (ex.: Authority/execution context) para carregar user_email por chamada
Sem isso, isolamento de credencial por usuário quebra silenciosamente
agent/metering.py
Manter no App por enquanto, mas considerar mover pra dentro do SDK (AgentLoop já sabe contar tokens) numa fase depois
Quotas não aplicadas hoje; oportunidade de consertar aplicando via max_tokens do próprio Agent por sessão
Mantém no App, sem tocar (por ora): web/auth.py, web/server.py (rotas HTTP puras), sandbox subsystem (Docker/E2B/AIO — infraestrutura de execução, não agente), vision/ subsystem, attachments_store.py/file_parse.py, agent/security.py (reusar como está), Wavespeed/ZuckPay REST clients (viram só tools registradas).
4. Cuckoo Filter + RAG Tree — dentro do SDK
Novo módulo: nullain-sdk/src/nullain/rag/ (nome sugerido, decidir no design doc):
cuckoo_filter.py — filtro probabilístico (suporta delete, ao contrário de Bloom) usado como pré-filtro de existência antes de bater no Qdrant: evita busca vetorial cara quando sabemos de antemão que aquele documento/chunk não está indexado para aquele usuário/coleção.
rag_tree.py — estrutura hierárquica (levels/cluster/parent) para busca em árvore: reduz o espaço de busca vetorial navegando de cluster amplo → específico, em vez de knn flat sobre todo o espaço.
qdrant_store.py — port VectorStore (Protocol, análogo ao LLMProvider) com adapter Qdrant.
metadata.py — contrato de metadados obrigatório por vetor: user_id/tenant_id (obrigatório, sem default), session_id, source, created_at, level, cluster_id, parent_id.
Isolamento multi-tenant (ponto crítico do Raphael)
Toda escrita de vetor passa por um VectorRecord tipado que exige tenant_id — sem tenant_id, TypeError/validação falha (fail-closed, mesmo padrão do resto do SDK).
Toda leitura usa filtro de payload obrigatório (must: [{key: "tenant_id", match: {value: ...}}]) — nunca um search() sem filtro. Isso vira um wrapper único (scoped_search(tenant_id, query, ...)) que é a ÚNICA função pública de busca — não expor client.search() cru em lugar nenhum do código, pra ninguém "esquecer" o filtro.
Teste de isolamento como gate de CI: inserir vetores de 2 tenants sintéticos com embeddings propositalmente próximos, garantir que a busca de um nunca retorna o outro — isso vira um teste obrigatório antes de qualquer merge dessa área.
Qdrant — schema (Free Tier)
Coleção única nullain_rag (Free Tier tem limite de coleções — melhor 1 coleção com payload rico que N coleções):
vector: embedding (dimensão conforme o embedding model escolhido — decidir no design doc, ex. bge-small/nomic-embed-text se quisermos rodar local/barato)
payload: tenant_id, session_id, rag_tree_node_id, level, cluster_id, parent_id, source, text_preview, created_at
Supabase — schema (Free Tier)
Tabelas relacionais (Postgres):
checkpoints — substitui o AsyncPostgresSaver do LangGraph E complementa/espelha o EventStore SQLite do SDK quando quisermos persistência centralizada multi-instância (ver §6, distribuição de serviço)
sessions — thread_id, tenant_id, created_at, last_active_at, model, status
users — id, email, tier (free/pro/enterprise — já existe o conceito em PlanTierLimits)
metadata — chave-valor genérico por sessão/usuário (preferências, projeto ativo etc. — parecido com o que agent/memory.py faz hoje em Redis+local)
traces — log estruturado de execução (tool calls, decisões do router, erros) — observabilidade, não é o EventStore (que é a fonte da verdade replay-able), é uma visão de leitura pra debugging/analytics
Nota de design: o EventStore do SDK (SQLite) continua sendo a fonte da verdade para replay/resume de sessão — é o que dá resume correto e repair automático. Supabase checkpoints/sessions/traces é uma camada de observabilidade e multi-instância por cima, não substitui o EventStore. Isso precisa virar decisão explícita no design doc: ou (a) EventStore grava local E espelha pro Supabase, ou (b) criamos um EventStore adapter Postgres/Supabase nativo no SDK. Recomendo (b) a médio prazo — mais limpo, um dono só da persistência — mas (a) é mais rápido pra sair do zero.
5. Injeção de contexto (SLM-first, per Raphael)
Como o app paga os tokens do próprio bolso e quer usar SLMs ao máximo, o AgentBridge/SDK precisa:
Um ContextAssembler (o SDK já tem nullain/context/assembler.py — verificar se cobre isso ou precisa de extensão) que monta o prompt final priorizando: (1) resultado do RAG Tree relevante, (2) memória de longo prazo relevante (PersistentMemory/EpisodicMemory do SDK, substituindo agent/memory.py do app), (3) histórico truncado — nessa ordem de prioridade de orçamento de tokens.
Cuckoo Filter entra aqui como otimização de custo: evita gastar uma chamada de embedding + busca vetorial quando o pré-filtro já garante "não existe conteúdo relevante indexado".
Isso é a peça que mais precisa de iteração empírica (medir qualidade de resposta do SLM com/sem RAG Tree) — não dá pra planejar 100% no papel, precisa de ciclo de teste.
6. Distribuição de serviço (per Raphael — fase de lançamento, não agora)
Registrar a decisão sem implementar ainda: quando sair do Free Tier, o processo que roda AgentLoop/tool execution (loops de LLM + tools, potencialmente sandbox) deve ser um worker separado do processo HTTP/SSE (web/server.py), escalável independentemente — hoje os dois vivem no mesmo processo Uvicorn. O AgentBridge já force isso a ser uma fronteira limpa (é a interface entre transporte e execução), então adicionar uma fila (Redis/Celery-like, ou HTTP interno) entre eles depois é uma extensão, não uma reescrita.
7. Fases de execução
Fase 0 — Fundação (sem quebrar nada em prod): criar AgentBridge no app, rodando em paralelo ao build_agent() atual atrás de um feature flag. Provar que nullain.Agent consegue: (a) rodar com OpenAICompatibleProvider apontando pro mesmo endpoint Ollama Cloud/xAI que o app já usa, (b) expor pelo menos 1 tool Composio real via RegisteredTool, (c) emitir eventos que o AgentBridge traduz pro formato SSE (status/token/tool/error/done) que o frontend já entende — sem mudar o frontend.
Fase 1 — Paridade de tools: portar Wavespeed, ZuckPay, sandbox tools, Composio (todas) pro ToolRegistry do SDK via AgentBridge. Resolver a injeção de user_email/tenant no contexto de execução da tool.
Fase 2 — Cortar o LangGraph: trocar web/server.py pra usar só AgentBridge, remover agent/chat.py's build_agent/LangChain, remover singleton global por app.state scoped por tenant/sessão.
Fase 3 — RAG novo: construir Cuckoo Filter + RAG Tree + Qdrant no SDK, com o teste de isolamento multi-tenant como gate. Rodar em paralelo ao AnythingLLM (flag), comparar qualidade, então remover AnythingLLM.
Fase 4 — Supabase: schema users/sessions/metadata/traces, decidir EventStore adapter Postgres nativo vs. espelhamento.
Fase 5 — Limpeza final: remover agent/supervisor.py, agent/resilience.py, agent/mcp_client.py (langchain_mcp_adapters), agent/models.py's duplicação de ModelRouter.
Cada fase termina em app funcionando de ponta a ponta antes da próxima começar — nunca um "big bang" de troca total.
8. Perguntas em aberto pra próxima conversa
Qual embedding model pro RAG Tree (local/SLM-friendly vs. hospedado)?
EventStore ganha adapter Postgres/Supabase nativo (opção b do §4) ou espelhamento (opção a)? Recomendo (b), mas é decisão de vocês.
O HITL guardrail do supervisor.py (destructive keywords) tem valor real que deveria virar uma feature do PermissionPolicy do SDK, ou pode simplesmente sumir?
Metering/quotas: mover pra dentro do AgentLoop (via max_tokens por sessão) ou manter camada separada no app?

Aqui está o plano atualizado com todas as decisões resolvidas — seção 8 substituída pelas ADRs, §4 corrigido com a decisão de embedding (bge-m3 via FastEmbed, não via Ollama Cloud), e Fase 4 atualizada. Pode colar direto pro Claude Code:

***

# Plano de Ação — Fusão Nullain Agent SDK ⇄ Nullain Agent (App)

## 0. Veredito em uma frase

O Nullain Agent (app) tem um `AgentLoop` completo escondido dentro do LangChain/LangGraph — `chat.py:build_agent()` reimplementa, com pior isolamento, quase tudo que o SDK já tem: loop de tool-calling, roteamento de modelo, memória, checkpoint, resiliência. A ponte certa é fazer o **SDK virar o cérebro** (`AgentLoop`, `ToolRegistry`, `EventBus`, `ModelRouter`, `EventStore`) e o **App virar apenas transporte + tools de domínio** (FastAPI/SSE, Composio, ZuckPay, Wavespeed, sandbox, RAG) — plugadas no SDK como `RegisteredTools`, não como um segundo agente paralelo.

## 1. O que descobrimos (mapa das duas bases)

### SDK (nullain-sdk) — já pronto e reaproveitável

- `Agent` (facade) → `AgentLoop` — ReAct loop hexagonal, já com `max_steps/max_tokens/timeout`, `EventBus` pub/sub, `EventStore` (SQLite, resume de sessão, repair automático de sessões corrompidas), `ModelRouter`, `PersistentMemory` + `EpisodicMemory`, `HookManager`.
- `ToolRegistry` + `PermissionPolicy` (ASK/ALLOW/DENY, fail-closed) + `Authority/Capability` (isolamento de subagentes).
- `LLMProvider` Protocol — `generate/stream/health_check`. Adapters hoje: `OllamaCloudProvider`, `OpenAICompatibleProvider` (issue \#40) — ou seja, **já é multi-provider de verdade**, ao contrário do app que só sabe falar `ChatOpenAI`.
- `nullain.mcp` — client MCP próprio (protocol.py/transport.py/client.py).
- `nullain.tools.sandbox` — sandbox multi-adapter (Landlock, seatbelt, Windows Job, none) já dentro do SDK.
- Nada de RAG, nada de vetor, nada de Qdrant/Supabase ainda — área nova.


### App (nullain-agent) — 11.665 linhas, LangChain/LangGraph

- `agent/chat.py` (1142L): `build_agent()` monta um `create_agent` do LangChain com `AgentMiddleware` próprios, `ChatOpenAI` genérico (qualquer provider "OpenAI-compatible"), checkpointer LangGraph (`InMemorySaver`/`AsyncPostgresSaver`). Esse é o **verdadeiro concorrente do `AgentLoop` do SDK**.
- `web/server.py` (1917L): FastAPI, mas com **agente global singleton por processo** (`_agent`/`_settings`/`_agent_lock`, linha 332) — o maior obstáculo estrutural pra injetar um `Agent` do SDK de forma limpa. Sem DI, sem isolamento multi-tenant real no nível do agente.
- Tool ecosystem: Composio (hub de ação — Gmail/Slack/GitHub/Notion/Discord etc.), AnythingLLM (RAG — a ser substituído), Wavespeed (imagem), ZuckPay (pagamento, credenciais por usuário via `RunnableConfig` do LangChain — **mecanismo específico do LangChain que a ponte precisa replicar**), Higgsfield/Semrush (MCPs externos com OAuth PKCE). Todas as tools já são `StructuredTool` + Pydantic `args_schema` → **equivalente funcional a JSON Schema OpenAI-style, compatível de cara com `ToolSpec/RegisteredTool` do SDK**.
- `agent/security.py`: puro, sem acoplamento a framework — sanitização de prompt injection, denylist de bash, criptografia Fernet de segredo por usuário, sanitização de erro. **Reaproveitável quase 1:1.**
- `agent/supervisor.py` + `agent/resilience.py`: scaffolding parcialmente morto — supervisor só é chamado 1x (chat.py:936), circuit breaker/retry sem call site confirmado. Fortes candidatos a **remoção total**, substituídos pelos mecanismos do SDK.
- `agent/metering.py`: quotas definidas mas não aplicadas em nenhum lugar — SDK deveria assumir isso se quisermos quotas reais.
- Redis: opcional/fail-open em todo lugar (metering, rate limit, login lockout) — não é dependência dura.


## 2. Decisão arquitetural — o padrão Bridge

```
┌─────────────────────────── nullain-agent (App) ───────────────────────────┐
│  web/server.py  →  FastAPI fino: HTTP/SSE, auth, upload, VNC proxy        │
│       │                                                                   │
│       ▼                                                                   │
│  AgentBridge (NOVO — camada de adaptação, vive no app)                    │
│    - 1 instância de nullain.Agent por (tenant/thread), não 1 global       │
│    - traduz SSE do app ⇄ AsyncIterator[BaseEvent] do SDK                  │
│    - injeta user_email/tenant_id como contexto de execução da tool        │
└──────────────────────────────┬────────────────────────────────────────────┘
                               │ usa
                               ▼
┌─────────────────────────── nullain-sdk (Core) ────────────────────────────┐
│  Agent → AgentLoop → ToolRegistry, ModelRouter, EventBus, EventStore      │
│  ToolRegistry recebe as tools de domínio do App como RegisteredTool:      │
│    - composio_bridge_tools   (via nullain.mcp client, não langchain_mcp)  │
│    - wavespeed_tool, zuckpay_tools, sandbox_tools                         │
│    - rag_search_tool  (NOVO: Cuckoo Filter + RAG Tree, dentro do SDK)     │
│  LLMProvider: OpenAICompatibleProvider apontando pro Ollama Cloud/xAI/etc │
└───────────────────────────────────────────────────────────────────────────┘
```

**Por que Bridge** (não "reescrever tudo", não "só importar o SDK direto no chat.py"):

- O App mantém sua própria identidade de transporte (FastAPI, SSE, auth, VNC, upload) — coisas de UI/infra que **não pertencem ao SDK**.
- O SDK nunca importa nada do App — a dependência é unidirecional (App depende do SDK, não o contrário), preservando o SDK como produto standalone/publicável.
- `AgentBridge` é a única camada nova que conhece os dois lados. Se o LangGraph for removido amanhã, só o `AgentBridge` muda.


## 3. O que sai da stack do App (retirar / substituir)

| Componente atual | Ação | Motivo |
| :-- | :-- | :-- |
| `langchain.agents.create_agent` + `AgentMiddleware` (chat.py) | Remover, substituído por `nullain.Agent.stream()` | É o próprio ReAct loop — duplicado pelo `AgentLoop` do SDK, com pior isolamento (1 agente global/processo) |
| `langgraph.checkpoint.*` (`InMemorySaver`/`AsyncPostgresSaver`) | Remover, substituído por `EventStore` do SDK (SQLite hoje; ver §5 pra Supabase) | Mesmo propósito (persistir trajetória), mecanismo já existe no SDK com resume + repair automático |
| `agent/mcp_client.py` (`MultiServerMCPClient` do langchain_mcp_adapters) | Adaptar, não remover — vira um `nullain.mcp` client para Composio, exposto como `RegisteredTools` | Consolida em um client MCP só; hoje há client MCP duplicado (langchain_mcp_adapters vs nullain.mcp) |
| `agent/anythingllm_tools.py` + AnythingLLM MCP/REST | Remover, substituído por Cuckoo Filter + RAG Tree (novo, dentro do SDK) | Pedido explícito do usuário |
| `agent/supervisor.py` | **Remover sem portar** (ver ADR-3) | Keyword blocklist é frágil (bypass trivial, falsos positivos). A garantia real vive no denylist do `security.py` (bash) e em rules declarativas do `PermissionPolicy` (tools irreversíveis) |
| `agent/resilience.py` (CircuitBreaker/retry_with_backoff) | Remover | Sem call site confirmado; SDK deve ter (ou ganhar) seu próprio retry/backoff no `LLMProvider/AgentLoop` |
| `web/server.py` globais (`_agent`/`_settings`/`_agent_lock`) | Substituir por `app.state` + resolução por request/tenant | Pré-requisito técnico pra multi-tenant real e pra várias sessões concorrentes usando o SDK corretamente |
| `_ModelRouterMiddleware` (chat.py) | Remover, substituído por `ModelRouter` do SDK | Já existe no SDK; o app só precisa mapear string de UI → model id |
| Credential injection via `RunnableConfig` (zuckpay_tools.py) | Adaptar: usar o mecanismo de contexto de execução do `ToolRegistry/AgentLoop` (ex.: `Authority`/execution context) para carregar `user_email` por chamada | Sem isso, isolamento de credencial por usuário quebra silenciosamente |
| `agent/metering.py` | **Manter no App como accounting; enforcement migra pro SDK como `QuotaHook`** (ver ADR-4) | Quotas não aplicadas hoje viram enforcement fail-closed no caminho crítico do AgentLoop |

**Mantém no App, sem tocar (por ora):** `web/auth.py`, `web/server.py` (rotas HTTP puras), sandbox subsystem (Docker/E2B/AIO — infraestrutura de execução, não agente), `vision/` subsystem, `attachments_store.py`/`file_parse.py`, `agent/security.py` (reusar como está), Wavespeed/ZuckPay REST clients (viram só tools registradas).

## 4. Cuckoo Filter + RAG Tree — dentro do SDK

Novo módulo: `nullain-sdk/src/nullain/rag/` (nome sugerido, decidir no design doc):

- `cuckoo_filter.py` — filtro probabilístico (suporta delete, ao contrário de Bloom) usado como **pré-filtro de existência** antes de bater no Qdrant: evita busca vetorial cara quando sabemos de antemão que aquele documento/chunk não está indexado para aquele usuário/coleção.
- `rag_tree.py` — estrutura hierárquica (levels/cluster/parent) para busca em árvore: reduz o espaço de busca vetorial navegando de cluster amplo → específico, em vez de knn flat sobre todo o espaço.
- `embedding.py` — port `EmbeddingProvider` (Protocol, análogo ao `LLMProvider`):

```python
class EmbeddingProvider(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...
```

Adapters: `FastEmbedProvider` (default — ver ADR-1) e `OllamaEmbeddingProvider` (pra quem rodar o SDK standalone com Ollama local).
- `qdrant_store.py` — port `VectorStore` (Protocol, análogo ao `LLMProvider`) com adapter Qdrant.
- `metadata.py` — **contrato de metadados obrigatório por vetor**: `user_id`/`tenant_id` (obrigatório, sem default), `session_id`, `source`, `created_at`, `level`, `cluster_id`, `parent_id`.


### Isolamento multi-tenant (ponto crítico do Raphael)

- Toda escrita de vetor passa por um `VectorRecord` tipado que **exige** `tenant_id` — sem `tenant_id`, `TypeError`/validação falha (fail-closed, mesmo padrão do resto do SDK).
- Toda leitura usa **filtro de payload obrigatório** (`must: [{key: "tenant_id", match: {value: ...}}]`) — nunca um `search()` sem filtro. Isso vira um wrapper único (`scoped_search(tenant_id, query, ...)`) que é a ÚNICA função pública de busca — não expor `client.search()` cru em lugar nenhum do código, pra ninguém "esquecer" o filtro.
- **Teste de isolamento como gate de CI:** inserir vetores de 2 tenants sintéticos com embeddings propositalmente próximos, garantir que a busca de um nunca retorna o outro — isso vira um teste obrigatório antes de qualquer merge dessa área.


### Embedding — decisão: bge-m3 via FastEmbed in-process (ver ADR-1)

**O modelo NÃO roda dentro do Qdrant nem do Supabase** — banco vetorial só armazena e busca vetores prontos. O bge-m3 roda **dentro do processo Python do app** via **FastEmbed** (lib da própria Qdrant, ONNX Runtime em CPU, sem PyTorch, sem serviço extra, sem Ollama). O Ollama Cloud não oferece bge-m3 — descartado pra embeddings. Fluxo: texto → FastEmbed (bge-m3, in-process) → vetor 1024-dim → Qdrant.

```python
from fastembed import TextEmbedding

model = TextEmbedding(model_name="BAAI/bge-m3")  # ONNX, baixa e cacheia na 1ª vez
vector = list(model.embed(["texto em português"]))[0]  # 1024 floats → Qdrant
```

**Modelo por ambiente** (o port `EmbeddingProvider` torna isso uma linha de config):

- **Dev/local + prod pago (≥2 GB RAM):** `BAAI/bge-m3` — 1024 dims, 8K contexto, multilíngue forte (100+ idiomas, PT-BR incluso), MIT. ~2.27 GB de modelo ONNX, ~1.5–2 GB RAM em inferência.
- **Free tier do host (≤512 MB RAM):** `embeddinggemma` (300M, ~600 MB RAM, 768 dims, Matryoshka truncation → 256 dims pra economia) ou `multilingual-e5-small` (~220 MB, 384 dims) — ambos também via FastEmbed, ambos com bom PT-BR.
- O RAG Tree e o Cuckoo Filter **não mudam** com a troca de modelo — a dimensão do vetor é detalhe do adapter, definida na criação da coleção.

**Dockerfile:** `pip install fastembed` + warm-up no build (`python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-m3')"`) pro modelo já vir na imagem e o cold start não pagar o download.

### Qdrant — schema (Free Tier: 1 nó, 1 GiB RAM, 0.5 vCPU, 4 GiB disco)

Coleção única `nullain_rag` (Free Tier tem limite de coleções — melhor 1 coleção com payload rico que N coleções):

- **vector:** dense, dimensão conforme o adapter ativo (1024 bge-m3 / 768 ou 256 embeddinggemma / 384 e5-small)
- **payload:** `tenant_id`, `session_id`, `rag_tree_node_id`, `level`, `cluster_id`, `parent_id`, `source`, `text_preview`, `created_at`

**Capacidade estimada no free tier:**


| Cenário | Config | Capacidade |
| :-- | :-- | :-- |
| Default | Vetores em RAM | ~150–200 mil vetores (~5–20 mil documentos a ~10–50 chunks/doc) |
| Otimizado | Scalar quantization (int8) + `on_disk: true` no HNSW | ~500–800 mil vetores (limite vira o disco) |

Suficiente com folga pro lançamento. Ativar quantização escalar quando passar de ~100 mil vetores — mudança de config da coleção, zero mudança de código. Texto completo dos chunks fica no Supabase; o Qdrant guarda só `text_preview`.

### Supabase — schema (Free Tier)

Tabelas relacionais (Postgres):

- `checkpoints` — substitui o `AsyncPostgresSaver` do LangGraph E complementa o `EventStore` (ver ADR-2)
- `sessions` — thread_id, tenant_id, created_at, last_active_at, model, status
- `users` — id, email, tier (free/pro/enterprise — já existe o conceito em `PlanTierLimits`)
- `metadata` — chave-valor genérico por sessão/usuário (preferências, projeto ativo etc. — parecido com o que `agent/memory.py` faz hoje em Redis+local)
- `traces` — log estruturado de execução (tool calls, decisões do router, erros) — observabilidade/read model derivado do EventBus, **nunca escrita paralela**, não é fonte da verdade


## 5. Injeção de contexto (SLM-first, per Raphael)

Como o app paga os tokens do próprio bolso e quer usar SLMs ao máximo, o `AgentBridge`/SDK precisa:

- Um `ContextAssembler` (o SDK já tem `nullain/context/assembler.py` — verificar se cobre isso ou precisa de extensão) que monta o prompt final priorizando: (1) resultado do RAG Tree relevante, (2) memória de longo prazo relevante (`PersistentMemory`/`EpisodicMemory` do SDK, substituindo `agent/memory.py` do app), (3) histórico truncado — nessa ordem de prioridade de orçamento de tokens.
- Cuckoo Filter entra aqui como otimização de custo: evita gastar uma chamada de embedding + busca vetorial quando o pré-filtro já garante "não existe conteúdo relevante indexado". Bônus do FastEmbed: como o embedding roda in-process em CPU, um "miss" do Cuckoo Filter economiza só CPU local, não dinheiro de API — o filtro continua valendo pelo tempo de resposta.
- Isso é a peça que mais precisa de iteração empírica (medir qualidade de resposta do SLM com/sem RAG Tree) — não dá pra planejar 100% no papel, precisa de ciclo de teste.


## 6. Distribuição de serviço (per Raphael — fase de lançamento, não agora)

Registrar a decisão sem implementar ainda: quando sair do Free Tier, o processo que roda `AgentLoop`/tool execution (loops de LLM + tools, potencialmente sandbox) deve ser um **worker separado** do processo HTTP/SSE (`web/server.py`), escalável independentemente — hoje os dois vivem no mesmo processo Uvicorn. O `AgentBridge` já força isso a ser uma fronteira limpa (é a interface entre transporte e execução), então adicionar uma fila (Redis/Celery-like, ou HTTP interno) entre eles depois é uma extensão, não uma reescrita. Nessa fase, avaliar também extrair o FastEmbed pra um micro-serviço de embedding se a CPU do worker virar gargalo — overkill agora.

## 7. Fases de execução

- **Fase 0 — Fundação (sem quebrar nada em prod):** criar `AgentBridge` no app, rodando em paralelo ao `build_agent()` atual atrás de um feature flag. Provar que `nullain.Agent` consegue: (a) rodar com `OpenAICompatibleProvider` apontando pro mesmo endpoint Ollama Cloud/xAI que o app já usa, (b) expor pelo menos 1 tool Composio real via `RegisteredTool`, (c) emitir eventos que o `AgentBridge` traduz pro formato SSE (status/token/tool/error/done) que o frontend já entende — **sem mudar o frontend**.
- **Fase 1 — Paridade de tools:** portar Wavespeed, ZuckPay, sandbox tools, Composio (todas) pro `ToolRegistry` do SDK via `AgentBridge`. Resolver a injeção de `user_email`/tenant no contexto de execução da tool.
- **Fase 2 — Cortar o LangGraph:** trocar `web/server.py` pra usar só `AgentBridge`, remover `agent/chat.py`'s `build_agent`/LangChain, remover singleton global por `app.state` scoped por tenant/sessão.
- **Fase 3 — RAG novo:** construir Cuckoo Filter + RAG Tree + `EmbeddingProvider` (FastEmbed) + Qdrant no SDK, com o teste de isolamento multi-tenant como gate. Criar a coleção `nullain_rag` com a dimensão do adapter ativo. Rodar em paralelo ao AnythingLLM (flag), comparar qualidade, então remover AnythingLLM.
- **Fase 4 — Supabase:** schema `users/sessions/metadata/traces` + **`PostgresEventStore` nativo no SDK** (ADR-2, opção b). Tabela `events` append-only (`session_id`, `seq`, `event_type`, `payload`, `created_at`), mesma semântica de resume/repair do SQLite — repair automático é feature do port, não do backend. SQLite continua default do SDK standalone e dev local; Postgres/Supabase entra via `nullain.toml` em prod. Um dono só da persistência por ambiente.
- **Fase 5 — Limpeza final:** remover `agent/supervisor.py` (sem portar — ADR-3), `agent/resilience.py`, `agent/mcp_client.py` (langchain_mcp_adapters), duplicação de `ModelRouter` em `agent/models.py`. Implementar `QuotaHook` (ADR-4) se ainda não estiver feito.

**Cada fase termina em app funcionando de ponta a ponta antes da próxima começar** — nunca um "big bang" de troca total.

## 8. Decisões tomadas (ADRs)

### ADR-1 — Embedding: bge-m3 via FastEmbed in-process

**Decisão:** `BAAI/bge-m3` rodando via **FastEmbed** (ONNX, CPU, in-process no app) atrás de um port `EmbeddingProvider` no SDK. **Não** via Ollama Cloud (bge-m3 não está disponível lá — só no Ollama self-hosted) e **não** "dentro" do Qdrant (banco vetorial só armazena vetores prontos, nunca roda modelo).

**Modelo por ambiente:** bge-m3 (1024 dims, ~2 GB RAM) em dev/prod pago; `embeddinggemma` (768 dims, ~600 MB) ou `multilingual-e5-small` (384 dims, ~220 MB) no free tier do host — troca via config, zero mudança de código.

**Alternativas consideradas:** Ollama self-hosted (serviço extra pra manter no deploy), API hospedada Jina/Voyage/OpenAI (custo por token + dependência externa), Qwen3-Embedding-8B (caro demais pro volume).

### ADR-2 — EventStore: adapter Postgres/Supabase nativo (opção b)

**Decisão:** `PostgresEventStore` no SDK implementando a mesma interface do `SQLiteEventStore`, com semântica idêntica de resume/repair (repair automático é feature do port, não do backend). Tabela append-only. SQLite continua default do standalone/dev; Postgres via `nullain.toml` em prod.

**Por que não espelhamento (opção a):** dual-write cria duas fontes de verdade e falha parcial (gravou local, falhou no Supabase → replay divergente). Um dono só da persistência por ambiente.

### ADR-3 — Guardrail do supervisor: remover sem portar

**Decisão:** `supervisor.py` morre na Fase 5. Keyword blocklist é o mecanismo mais frágil que existe (bypass trivial por encoding/sinônimo/indireção, falsos positivos em contexto legítimo, manutenção eterna de lista).

**A garantia real já existe em dois lugares melhores:** comandos bash destrutivos → denylist do `security.py` (fica, é puro e testável); ações irreversíveis (pagamento ZuckPay, delete, envio externo) → rules declarativas no `PermissionPolicy` do SDK (`zuckpay.*` = ASK, `sandbox.bash` = policy por comando, fail-closed por default). Confirmação por tool registrada, não por substring no texto do usuário — mais auditável, sem falso positivo.

### ADR-4 — Metering/quotas: enforcement no SDK (hook), accounting no App

**Decisão:** split enforcement/accounting. Um `QuotaHook` no `HookManager` do SDK (pre-step, fail-closed) consulta um port `QuotaStore` antes de cada chamada de LLM/tool e nega se estourou — quota é hook opcional, SDK permanece standalone. O `metering.py` do app vira subscriber do EventBus que só persiste usage (Redis agora, Supabase `traces` na Fase 4) e aplica regras de `PlanTierLimits`.

**Por que não `max_tokens` por sessão como quota:** ele é circuit breaker de runaway loop — não sabe fazer quota diária/mensal, tier por usuário, nem custo diferenciado por modelo. Os dois coexistem: `max_tokens` protege a sessão, o `QuotaHook` protege o bolso. Isso resolve de verdade o problema atual ("quotas definidas mas não aplicadas") — hoje é código morto, vira enforcement fail-closed no caminho crítico.

***

Uma observação: guarde esse documento no repo — `docs/FUSION_PLAN.md` no `nullain-agent` ou um ADR no SDK. As Fases 0–5 vão gerar várias sessões de trabalho no Claude Code, e ter o plano versionado evita que as decisões (principalmente as 4 ADRs) se percam entre conversas.

