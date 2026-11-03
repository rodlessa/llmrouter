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

## Começando do zero (para a próxima máquina)

```bash
git clone git@github.com:rodlessa/llmrouter.git ~/ai/router
cd ~/ai/router
./install.sh                 # venv (uv) + router.env/local.env + units systemd
$EDITOR router.env           # API_KEY, API_BASE_URL, API_MODEL
$EDITOR local.env            # LLAMA_SERVER (binário) e MODEL_PATH (.gguf)
systemctl --user restart llmrouter
./tests/run.sh               # 33 checagens em container (podman/docker)
```

Depois aponte o cliente para `http://127.0.0.1:4000/v1` (seção **OpenClaw**).

### Checklist: está funcionando?

```bash
systemctl --user is-active llmrouter          # -> active
curl -s localhost:4000/v1/models              # -> router/auto, router/local, router/api
curl -s localhost:4000/local/status           # -> "ativo": true depois do 1o request local
curl -s -X POST localhost:4000/route/explain \
  -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"oi"}]}'
```

Se o `/route/explain` diz `destino: local` para uma pergunta curta e `destino: api`
para algo com "refatorar"/"arquitetura", o roteamento está de pé.

### Comandos do dia a dia

| Quero | Comando |
|---|---|
| ver o estado do modelo local | `curl -s localhost:4000/local/status` |
| liberar VRAM agora (jogo) | `systemctl --user stop llama-local` |
| pré-carregar o modelo | `systemctl --user start llama-local` |
| ver a decisão de rota de um prompt | `curl -s -X POST localhost:4000/route/explain -H 'Content-Type: application/json' -d '{...}'` |
| acompanhar o router | `journalctl --user -u llmrouter -f` |
| acompanhar o modelo | `journalctl --user -u llama-local -f` |
| rodar a suíte de testes | `./tests/run.sh` |
| aplicar mudança de env | `systemctl --user restart llmrouter` |

### Onde cada coisa mora

| Arquivo | Papel |
|---|---|
| `router.env` | API key + limites + Laya (nunca versionado) |
| `local.env` | caminho do binário `llama-server` e do `.gguf` |
| `llmrouter.py` | o proxy e as decisões de rota |
| `run-local.sh` | wrapper que sobe o llama-server lendo `local.env` |
| `systemd/` | units instaladas em `~/.config/systemd/user` |
| `bench/` | mede modelos GGUF para escolher o local |
| `tests/` | suíte de validação em container |

## Requisitos

- Linux com systemd `--user` (testado em Arch, AMD RX 9070 XT + Vulkan).
- [`uv`](https://docs.astral.sh/uv/) (o projeto **não usa pip**).
- `llama.cpp` compilado (binário `llama-server`) e um `.gguf`.
- [`rocm-smi`](https://rocm.docs.amd.com/) para a leitura de VRAM (opcional; sem ele a guarda é desativada).

## Instalação (detalhes)

O caminho rápido está em **Começando do zero**. Variações:

```bash
./install.sh --with-laya     # + decisor local (torch CPU + laya, ~1.7 GB)
./install.sh --no-systemd    # só venv + envs, sem units
./install.sh --force         # sobrescreve units existentes (guarda .bak)
```

O `install.sh` nunca sobrescreve `router.env`/`local.env` existentes e degrada com
aviso quando não há `systemctl --user` (container, WSL sem systemd).

Se o `llama-server` estiver em outro caminho, ajuste `LLAMA_SERVER` no `local.env`
(o `run-local.sh` lê esse arquivo e faz `exec` no binário).

### Modelo local sugerido

`unsloth/gpt-oss-20b-GGUF` — `gpt-oss-20b-UD-Q4_K_XL.gguf` (11 GiB, ~180 tok/s,
MoE). Alternativa que ocupa metade da VRAM: `gemma-4-12b-it-UD-Q4_K_XL.gguf`
(6.9 GiB, ~62 tok/s).

## Resultados medidos

Bench de qualidade (`bench/quality_bench.py`), 3 runs por tarefa, temp 0.2,
commit `8268bf3a5678` — 18 tarefas de código/reasoning + 20 de tool calling:
Ryzen 9 9950X, RX 9070 XT 16 GiB (Vulkan), 2026-11-02.

| modelo | VRAM | tok/s | tarefas (18) | tool calling (20) | pior tarefa |
|---|---:|---:|---:|---:|---:|
| **gpt-oss-20b Q4XL** | 11.06 GiB | **182** | **18/18** | 19/20 | 13.1 s |
| gemma-4-12b Q4XL | 6.86 GiB | 62 | 15/18 | 20/20 | 132 s |
| qwen3.5-9b Q4XL | 5.55 GiB | 84 | 13/18 | 20/20 | 97 s |

- **gpt-oss-20b**: único que acerta edição cirúrgica (3/3); o único erro é 1 tool
  call respondida em texto. É o modelo recomendado.
- **gemma-4-12b**: perfeito em tools e metade da VRAM, mas falha edição cirúrgica
  (0/3) e é 3x mais lento.
- **qwen3.5-9b**: entra em loop de raciocínio (8000 tokens, conteúdo vazio) em
  edição cirúrgica e escrita de teste — descartado.
- O tok/s cai muito quando outro processo ocupa a VRAM (jogo): o llama.cpp manda
  as camadas que não couberem para CPU+RAM.

Números do router em uso (este repo): prompt de agente ≈ **15k tokens**,
prefill ≈ **5.1k tok/s**, geração **159–180 tok/s**, primeira resposta local em
**~6 s**; cold start do modelo **~4 s**.

Suíte de testes (`tests/run.sh`): **33 asserts, 0 falhas**, rodando em
`debian:bookworm-slim` limpo via podman rootless — inclui clone sem segredos,
instalação idempotente, roteamento, rodapé em stream/non-stream, gzip do
upstream, ordem do SSE, fallback e fast-fail sem systemd.

### Bench por categoria (suite agnostica)

`bench/run_suite.py` + `bench/suite.json` — 7 categorias, 2 runs, ctx 32768,
gpt-oss-20b / gemma-4-12b / qwen3.5-9b. Score = tarefas aprovadas; tempo = mediana.

| categoria | gpt-oss-20b | gemma-4-12b | qwen3.5-9b | o que mede |
|---|---|---|---|---|
| code_write | **100%** (4 s) | 75% (37 s) | 50% (30 s) | escrever funcao a partir da spec |
| code_fix | **75%** (8 s) | 50% (31 s) | 50% (21 s) | corrigir bug com testes |
| code_review | **50%** (13 s) | 0% (29 s) | 0% (49 s) | achar os defeitos plantados |
| refactor | 100% (7 s) | 100% (17 s) | 100% (30 s) | mover logica sem mudar comportamento |
| explain | 100% (**1 s**) | 100% (13 s) | 100% (17 s) | resposta curta e correta |
| long_context | 100% (1 s) | 50% (8 s) | 100% (5 s) | agulha no palheiro em ~9k tok |
| tool_call | 100% (5 s) | 50% (25 s) | 100% (15 s) | 8 casos get_weather/run_sql |

Em `code_review` o binario esconde o detalhe: contando os defeitos plantados um a
um (4 por caso), gpt-oss acha **23/32 (72%)**, gemma **20/32 (62%)**, qwen **0/32**
(nem devolve JSON, entra em loop de raciocinio).

Conclusao: **um modelo serve para tudo** — gpt-oss-20b. Nao ha ganho em trocar de
modelo por tarefa; gemma e qwen erram mais e sao 3-4x mais lentos. Trocar de
modelo exigiria parar/religar o llama-server (4-10 s) para piorar a resposta.

### Janela de contexto (gpt-oss-20b, 16 GiB)

| ctx | VRAM | agulha em 13.4k tok | geracao |
|---|---|---|---|
| 32768 | 12.91 GiB | ok | 160 tok/s |
| **65536** | **13.71 GiB** | ok | **162 tok/s** |
| 131072 | 14.86 GiB | ok | 131 tok/s |

O KV cache do gpt-oss e barato: dobrar de 32k para 64k custa ~0.8 GiB. O padrao
do repo passou a ser **65536** (`CTX` no local.env e `LOCAL_CTX` no router.env).
131072 funciona, mas deixa ~1 GiB livre na GPU — so use com a maquina so para isso.

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

Quando a rota muda em relação ao request anterior da mesma sessão (local -> API
ou API -> local), a caixa ganha uma linha extra avisando:

```
+------------------------------------+
| ROTA: API (deepseek-v4-flash)      |
| 271 prompt tok | 65 gerados | 1.7s |
| mudou: LOCAL -> API                |
+------------------------------------+
```

O aviso é por sessão quando o cliente manda um header de sessão
(`x-session-id`, `x-openclaw-session`); sem header, vale para o processo.

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
