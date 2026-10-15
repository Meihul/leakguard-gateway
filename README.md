# LeakGuard Gateway

AI security gateway that sits between users and an LLM: inspects prompts and
responses, masks or blocks sensitive data and prompt injection, and enforces
role-based document access. See `CLAUDE.md` for the full project contract.

## Run in one command

Requires [Docker](https://www.docker.com/) and a `.env` file in the project
root (see `gateway/main.py` / `gateway/detectors/injection_ml.py` for the
expected variables: `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`). The `.env`
file is never baked into the image — `docker-compose.yml` reads it at
container start via `env_file`, and `.dockerignore` keeps it out of the
build context entirely.

```powershell
docker compose up --build
```

The gateway is then available at `http://localhost:8000`, with the RAG
index rebuilt from `data/docs` on every start and both the Chroma index and
the SQLite audit DB persisted in a named Docker volume across restarts.

- `POST /chat` — the native LeakGuard API.
- `POST /v1/chat/completions` — an OpenAI-compatible endpoint, so any
  existing chatbot can switch to LeakGuard by changing only its base URL to
  `http://localhost:8000/v1`. Pass `X-LeakGuard-Role` / `X-LeakGuard-User`
  headers to set the role (default `intern`) and user for access control;
  LeakGuard's own decision is returned in an extra `leakguard` field
  (`action`, `findings`, `user_message`) alongside the standard
  `chat.completion` response.

Stop it with `docker compose down` (add `-v` to also drop the persisted
index/audit volume).
