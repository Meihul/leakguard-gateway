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

## UI

With the gateway running (`http://localhost:8000`), start the Streamlit UI
with one command:

```powershell
streamlit run ui/chat_app.py
```

This is a single multipage app — "Chat", "Approvals" and "Dashboard" all
live in the sidebar. You log in first; your role comes from your account
(there is no role picker). Approvals and Dashboard only show their content
to `manager`/`hr` (interns see a notice instead).

## File uploads

`POST /chat` also accepts `multipart/form-data` with `prompt`, optional
`confirm`, and an optional `file` (.txt, .pdf, .docx, .csv; max 5 MB). The
file is read in memory only and never written to disk. Its text goes through
the same detectors as prompts, plus confidential-marker detection. Then:

- hidden instructions (injection) in the file → **block**
- PII / secrets → **masked**, then sent to the LLM as clearly delimited
  reference data alongside the question (the uploader's own values are
  restored in their response, as for prompts)
- "CONFIDENTIAL" / "INTERNAL USE ONLY" style stamps, or a scanned PDF with no
  text layer → **confirm**

A file that gets through returns a `file_id`. Send `{"prompt", "file_id"}`
to `/chat` to keep asking about it: answers come only from that file (no
company-document search), with the same output inspection and audit logging.
The masked file text is kept in memory only, for its uploader only, and
expires 30 minutes after its last use (`410` with a clear message);
`DELETE /files/{file_id}` detaches it early. In the chat UI the file stays
attached (chip above the input) until you click ✕.

Findings from a file carry `"source": "file"` and a `page` (PDF) or `line`
(txt/csv lines, docx paragraphs). The audit log stores only a hash of the
filename, the file type, size and finding locations — never the content.

Demo files are in `demo_files/` (regenerate with
`python demo_files/make_demo_files.py`): a clean policy PDF, an employee
CSV with fake Aadhaar and phone numbers, and a .docx with a hidden injection
line. In the chat UI, use the paperclip next to the message box.

## Authentication (demo-grade)

Set a signing secret in `.env` (see `.env.example`) before starting the gateway:

```
LEAKGUARD_SECRET=<a long random string>
```

Generate one in PowerShell with:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Demo accounts (passwords are stored only as salted PBKDF2 hashes in
`gateway/users.yaml`):

| Username | Password        | Role    |
|----------|-----------------|---------|
| riya     | `riya-demo123`  | intern  |
| aman     | `aman-demo123`  | manager |
| priya    | `priya-demo123` | hr      |

For live demos, set `DEMO_MODE=true` in `.env` to show one-click login
buttons ("Riya (intern)", "Aman (manager)", "Priya (HR)") on the login page.
They read these demo credentials from `ui/demo_accounts.yaml` and call
`/login` exactly like the normal form, which stays available. Leave it
`false` (the default) anywhere else.

`POST /login {"username", "password"}` returns a signed token valid for 8
hours. `/chat`, `/approvals`, `/requests` and `/v1/chat/completions` require
`Authorization: Bearer <token>`; the user and role come only from the token
(any `user`/`role` in the request body is ignored). Only `hr`/`manager`
tokens can list or decide approvals, and the approver name is taken from the
token. Changing `LEAKGUARD_SECRET` invalidates all existing tokens.
