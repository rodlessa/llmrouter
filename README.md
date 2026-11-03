# llmrouter

Proxy OpenAI-compatível que decide, por request, entre **modelo local** (llama.cpp)
e **API na nuvem**. Feito para o OpenClaw, mas serve qualquer cliente que fale
`/v1/chat/completions`.

```
        +-------------------+        simples        +----------------------+
        |                   |---------------------->| llama.cpp (local)    |
cliente |     llmrouter     |                       |  :8080  GPU+CPU/RAM  |
  ----> |      :4000        |        pesado         +----------------------+
        |                   |---------------------->| API OpenAI-compat    |
        +-------------------+                       |  (DeepSeek, etc)     |
                 |                                  +----------------------+
                 |
                 +-- schemas de tools seguros
                 +-- limite de contexto
                 +-- palavras-chave de tarefa pesada
                 +-- Laya (opcional): probabilidades, nao decide
```

## O que ele faz

- **Roteia por request**, não por sessão: a mesma conversa pode alternar entre local e API.
- **Contexto grande, loop de tool e tarefa pesada vão para a API.** Curto e simples vão para o local.
- **Sobe o llama-server sob demanda** e **derruba depois de N segundos ocioso** (libera VRAM para jogos).
- **Guarda de VRAM**: se não houver VRAM livre suficiente, não carrega o modelo — manda para a API.
- **Fallback automático**: local fora do ar ou 5xx → repete na API.
- **Rodapé ASCII** no fim de cada resposta de agente dizendo onde ela rodou.
- Corrige o `gzip` do upstream (`aiter_bytes`), aceita ids de modelo sem prefixo e
  não vaza o `content-encoding`.

## Requisitos

- Linux com systemd `--user` (testado em Arch, AMD RX 9070 XT + Vulkan).
- [`uv`](https://docs.astral.sh/uv/) (o projeto **não usa pip**).
- `llama.cpp` compilado (binário `llama-server`) e um `.gguf`.
- [`rocm-smi`](https://rocm.docs.amd.com/) para a leitura de VRAM (opcional; sem ele a guarda é desativada).

## Instalação

```bash
git clone <repo> ~/ai/router
cd ~/ai/router
./install.sh              # venv + envs + units
./install.sh --with-laya  # + decisor local (torch CPU + laya, ~1.7 GB)
```

Depois edite:

- `router.env` — `API_KEY`, `API_BASE_URL`, `API_MODEL` (nunca commite este arquivo)
- `local.env` — `LLAMA_SERVER` e `MODEL_PATH` do seu `.gguf`

```bash
systemctl --user restart llmrouter
curl -s http://127.0.0.1:4000/local/status
```

Se o `llama-server` estiver em outro caminho, ajuste `LLAMA_SERVER` no `local.env`;
a unit já usa `${LLAMA_SERVER}` em vez de caminho fixo.

### Modelo local sugerido

`unsloth/gpt-oss-20b-GGUF` — `gpt-oss-20b-UD-Q4_K_XL.gguf` (11 GiB, ~180 tok/s,
MoE). Alternativa que ocupa metade da VRAM: `gemma-4-12b-it-UD-Q4_K_XL.gguf`
(6.9 GiB, ~62 tok/s).

## OpenClaw

Provider (merge no `~/.openclaw/openclaw.json`):

```json5
{
  models: {
    mode: "merge",
    providers: {
      router: {
        baseUrl: "http://127.0.0.1:4000/v1",
        apiKey: "router-local",
        api: "openai-completions",
        timeoutSeconds: 600,
        models: [
          { id: "auto",  name: "Router auto (local->API)", reasoning: true,
            input: ["text"], contextWindow: 1000000, maxTokens: 8192,
            cost: { input: 0.14, output: 0.28, cacheRead: 0.028, cacheWrite: 0 } },
          { id: "local", name: "Router local (llama.cpp)", reasoning: true,
            input: ["text"], contextWindow: 32768, maxTokens: 8192,
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } },
          { id: "api",   name: "Router API", reasoning: true,
            input: ["text"], contextWindow: 1000000, maxTokens: 8192,
            cost: { input: 0.14, output: 0.28, cacheRead: 0.028, cacheWrite: 0 } }
        ]
      }
    }
  },
  agents: {
    defaults: {
      model: { primary: "router/auto", fallbacks: ["deepseek/deepseek-v4-flash"] },
      models: {
        "router/auto": { alias: "Router" },
        "router/local": { alias: "Local" },
        "router/api": { alias: "API" }
      }
    }
  }
}
```

`router/auto` decide sozinho; `router/local` e `router/api` forçam o destino
(via `/model` ou `--model`).

> `router/local` declara 32k de contexto. O router escala para a API acima de
> ~60% da janela local, então nunca estoura o `-c` do llama-server.

## Configuração (`router.env`)

| Variável | Padrão | O que faz |
|---|---|---|
| `API_BASE_URL` / `API_KEY` / `API_MODEL` | — | upstream OpenAI-compatível |
| `LOCAL_BASE_URL` | `http://127.0.0.1:8080/v1` | llama-server |
| `LOCAL_CTX` | `32768` | igual ao `-c` do llama-server |
| `LOCAL_TOOL_ALLOWLIST` | `*` | `*` libera todas; lista separa por vírgula |
| `LOCAL_SERVICE` | `llama-local.service` | unit do systemd |
| `LOCAL_AUTOSTART` | `1` | subir sob demanda |
| `LOCAL_IDLE_STOP_SECONDS` | `180` | 0 = nunca parar |
| `LOCAL_MIN_FREE_VRAM_MIB` | `13500` | abaixo disso, vai para a API |
| `FOOTER` | `1` | rodapé de rota |
| `FOOTER_REQUIRE_TOOLS` | `1` | só em requests com tools |
| `LAYA_MODE` | `advise` | `off` / `advise` / `decide` |
| `ROUTER_DEBUG` | `1` | loga decisão, tools e tokens estimados |

## Rodapé de rota

Toda resposta de agente termina com uma caixa ASCII:

```
+---------------------------------------------------+
| ROTA: LOCAL (llama.cpp)                           |
| 14954 prompt tok | 104 gerados | 159 tok/s | 6.1s |
+---------------------------------------------------+
```

Ele é injetado no SSE **antes** do `finish_reason`: o router segura o chunk de
finish, emite um delta com a caixa e só então libera o finish e o usage. No modo
non-stream, é concatenado ao `content`.

## Laya (opcional)

[`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) é um
System One model Apache-2.0 que roda em CPU: state + perguntas tipadas
(`choice`/`score`/`noul`) → probabilidades calibradas, ~130 ms.

Medição neste repo: o checkpoint zero-shot **não** é confiável para decidir a
rota (ex.: "refatore o arquivo X" → `local p=0.80`). Por isso o padrão é
`advise`: ele roda, grava `decisions.jsonl` e **não** muda a rota. Esse arquivo
é o dataset para fine-tunar; depois disso troque para `LAYA_MODE=decide`
(`LAYA_DECIDE_P` define o corte).

Custo em `advise`: ~2,6 s no primeiro request, ~130 ms por request e +1,7 GB RSS.
`LAYA_MODE=off` desliga.

## Ciclo de vida / VRAM

```bash
systemctl --user stop llama-local      # unload imediato (~13 GiB)
systemctl --user start llama-local     # pré-carrega
curl -s localhost:4000/local/status    # estado, VRAM livre, ocioso
```

- Não sobe no boot (`disabled`), sobe no primeiro request local.
- Cai sozinho após `LOCAL_IDLE_STOP_SECONDS`.
- `PartOf=openclaw-gateway.service`: parar o gateway descarrega o modelo.

## API de diagnóstico

```bash
curl -s localhost:4000/v1/models
curl -s localhost:4000/local/status
curl -s -X POST localhost:4000/route/explain -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"..."}]}'
```

`/route/explain` mostra a decisão da heurística e o palpite do Laya sem
encaminhar nada.

## Bench (`bench/`)

`quality_bench.py` mede acerto e tok/s de modelos GGUF locais, salvando um JSON
completo por teste (`prompt`, `request`, `resposta_final`, `reasoning`,
`tool_calls`) em `~/quality-details/<data>/<modelo>-<commit>/`. É a ferramenta que
usamos para escolher o modelo local. Arquivo é um snapshot; a versão de trabalho
fica fora do repo.

## Testes (container)

```bash
tests/run.sh
```

Sobe um Debian limpo, clona o commit num `git bundle` (testa exatamente o que
seria enviado), roda `install.sh --no-systemd`, sobe o router contra dois mocks
(API OpenAI + llama-server) e valida 33 pontos: clone sem segredos, instalação
idempotente, roteamento por contexto/tools/palavra-chave, rodapé em stream e
non-stream, gzip do upstream, ordem do SSE (`conteudo -> rodape -> finish ->
DONE`), fallback local->API, fast-fail sem `systemctl` e degradação do
`install.sh` em ambiente sem systemd.

## Segurança

- `router.env`, `local.env` e qualquer `*.env` estão no `.gitignore`.
- `decisions.jsonl` (contém trechos de prompts) também.
- Nenhum `.gguf`, `.venv`, banco SQLite ou log entra no repo.
- A `API_KEY` fica só no `router.env` local, com `chmod 600`.

## Troubleshooting

| Sintoma | Causa provável |
|---|---|
| `Agent couldn't generate a response` | upstream comprimido sem header: use `aiter_bytes()` |
| Tudo cai na API | `tools` fora da allowlist ou contexto acima de 60% |
| `local indisponivel (VRAM insuficiente)` | outro processo (jogo) está com a GPU |
| 502 em tudo | `API_KEY`/`API_BASE_URL` errados |
| Primeiro request demora ~4 s | cold start do llama-server (normal) |
