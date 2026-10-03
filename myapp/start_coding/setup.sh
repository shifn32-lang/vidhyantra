#!/usr/bin/env bash
# __BRAND__ - Start coding: one-line setup for the OpenCode terminal agent.
#
#   curl -fsSL __BASE_URL__/start-coding/setup.sh | bash
#
# What it does, in order:
#   1. installs OpenCode if it is missing (the official installer, or npm)
#   2. shows a short code and opens your browser so YOU approve this computer
#   3. saves the key it receives in OpenCode's own credentials file
#   4. adds __MODEL_LABEL__ to OpenCode's config
# It never asks you to paste a key, and it only touches OpenCode's two files:
#   ~/.local/share/opencode/auth.json   and   ~/.config/opencode/opencode.json
# Options:  --project   write opencode.json in the current folder instead
#           --no-browser  print the link instead of opening it

set -u
BASE_URL="__BASE_URL__"
BRAND="__BRAND__"
MODEL_ID="__MODEL_ID__"
MODEL_LABEL="__MODEL_LABEL__"

PROJECT_ONLY=0
OPEN_BROWSER=1
for arg in "$@"; do
  case "$arg" in
    --project) PROJECT_ONLY=1 ;;
    --no-browser) OPEN_BROWSER=0 ;;
  esac
done

if [ -t 1 ]; then BOLD=$'\033[1m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; DIM=$'\033[2m'; OFF=$'\033[0m'; else BOLD=""; GREEN=""; YELLOW=""; RED=""; DIM=""; OFF=""; fi
say()  { printf '%s\n' "$*"; }
step() { printf '\n%s==>%s %s%s%s\n' "$GREEN" "$OFF" "$BOLD" "$*" "$OFF"; }
warn() { printf '%s!%s %s\n' "$YELLOW" "$OFF" "$*"; }
die()  { printf '%sx%s %s\n' "$RED" "$OFF" "$*" >&2; exit 1; }

command -v curl >/dev/null 2>&1 || die "curl is required. Install it and run this again."

# Anything that can read and write JSON: used to MERGE into files you may
# already have, so your other OpenCode providers are never overwritten.
JSON_TOOL=""
for tool in python3 python node; do
  if command -v "$tool" >/dev/null 2>&1; then
    case "$tool" in python*) "$tool" -c 'import json' >/dev/null 2>&1 || continue ;; esac
    JSON_TOOL="$tool"; break
  fi
done

say ""
say "${BOLD}${BRAND} - Start coding${OFF}"
say "${DIM}Sets up OpenCode with ${MODEL_LABEL} on this computer.${OFF}"

# ---------------------------------------------------------------- 1. OpenCode
step "1/4  Checking OpenCode"
OPENCODE_BIN="$(command -v opencode 2>/dev/null || true)"
[ -z "$OPENCODE_BIN" ] && [ -x "$HOME/.opencode/bin/opencode" ] && OPENCODE_BIN="$HOME/.opencode/bin/opencode"
if [ -n "$OPENCODE_BIN" ]; then
  say "OpenCode is already installed ($("$OPENCODE_BIN" --version 2>/dev/null || echo installed))."
else
  say "OpenCode is not installed yet. Installing it..."
  if curl -fsSL https://opencode.ai/install | bash; then :; elif command -v npm >/dev/null 2>&1; then
    warn "The official installer did not finish; trying npm."
    npm install -g opencode-ai || die "Could not install OpenCode. See https://opencode.ai/docs for other ways, then run this again."
  else
    die "Could not install OpenCode. Install it from https://opencode.ai/docs (or install Node.js and run: npm install -g opencode-ai), then run this again."
  fi
  OPENCODE_BIN="$(command -v opencode 2>/dev/null || true)"
  [ -z "$OPENCODE_BIN" ] && [ -x "$HOME/.opencode/bin/opencode" ] && OPENCODE_BIN="$HOME/.opencode/bin/opencode"
fi

# ----------------------------------------------------------- 2. browser approval
step "2/4  Approve this computer in your browser"
MACHINE="$(hostname 2>/dev/null | tr -cd 'A-Za-z0-9._ -' | cut -c1-40)"
[ -z "$MACHINE" ] && MACHINE="my computer"
START="$(curl -fsS -m 30 -X POST "$BASE_URL/api/v1/code/device/start" -H 'Content-Type: application/json' -d "{\"machine\":\"$MACHINE\"}" 2>/dev/null)" || START=""
field() { printf '%s' "$1" | grep -o "\"$2\" *: *\"[^\"]*\"" | head -1 | sed 's/.*: *"\(.*\)"/\1/'; }
number() { printf '%s' "$1" | grep -o "\"$2\" *: *[0-9]*" | head -1 | sed 's/.*: *//'; }
DEVICE_CODE="$(field "$START" device_code)"
USER_CODE="$(field "$START" user_code)"
VERIFY_URL="$(field "$START" verification_url | sed 's#\\/#/#g')"
EXPIRES="$(number "$START" expires_in)"; EXPIRES="${EXPIRES:-600}"
INTERVAL="$(number "$START" interval)"; INTERVAL="${INTERVAL:-3}"
if [ -z "$DEVICE_CODE" ] || [ -z "$USER_CODE" ]; then
  MSG="$(field "$START" message)"
  die "${MSG:-Could not reach $BASE_URL. Check your internet connection and try again.}"
fi

say ""
say "  Your code:  ${BOLD}${USER_CODE}${OFF}"
say "  Open this link, check the code matches, and click Approve:"
say "  ${BOLD}${VERIFY_URL}${OFF}"
say ""
if [ "$OPEN_BROWSER" = 1 ]; then
  if command -v xdg-open >/dev/null 2>&1; then xdg-open "$VERIFY_URL" >/dev/null 2>&1 &
  elif command -v open >/dev/null 2>&1; then open "$VERIFY_URL" >/dev/null 2>&1 &
  elif command -v wslview >/dev/null 2>&1; then wslview "$VERIFY_URL" >/dev/null 2>&1 &
  elif command -v cmd.exe >/dev/null 2>&1; then cmd.exe /c start "" "$VERIFY_URL" >/dev/null 2>&1 &
  fi
fi
say "Waiting for you to approve (this code expires in $((EXPIRES / 60)) minutes)..."

API_KEY=""
DEADLINE=$(( $(date +%s) + EXPIRES ))
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  sleep "$INTERVAL"
  REPLY="$(curl -sS -m 30 -X POST "$BASE_URL/api/v1/code/device/poll" -H 'Content-Type: application/json' -d "{\"device_code\":\"$DEVICE_CODE\"}" 2>/dev/null)" || continue
  case "$(field "$REPLY" status)" in
    approved) API_KEY="$(field "$REPLY" api_key)"; break ;;
    denied)   die "The request was denied in the browser. Nothing was changed." ;;
    expired)  die "The code expired. Run the setup command again." ;;
    invalid)  die "This setup request is no longer valid. Run the setup command again." ;;
  esac
done
[ -z "$API_KEY" ] && die "Timed out waiting for approval. Run the setup command again."
say "${GREEN}Approved.${OFF}"

# ------------------------------------------------------------ 3. save the login
step "3/4  Saving your sign-in"
DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/opencode"
AUTH_FILE="$DATA_DIR/auth.json"
mkdir -p "$DATA_DIR" || die "Could not create $DATA_DIR"

if [ -n "$JSON_TOOL" ]; then
  export VD_KEY="$API_KEY" VD_BASE="$BASE_URL" VD_BRAND="$BRAND" VD_MODEL_ID="$MODEL_ID" VD_MODEL_LABEL="$MODEL_LABEL"
  merge_json() {  # merge_json FILE auth|config
    case "$JSON_TOOL" in
      node) node -e '
const fs=require("fs");const [file,mode]=process.argv.slice(1);let d={};
try{d=JSON.parse(fs.readFileSync(file,"utf8"));if(typeof d!=="object"||d===null||Array.isArray(d))process.exit(3)}catch(e){if(e.code!=="ENOENT")process.exit(3)}
if(mode==="auth"){d.vidhyora={type:"api",key:process.env.VD_KEY}}
else{d.$schema=d.$schema||"https://opencode.ai/config.json";d.provider=d.provider||{};
const m={};m[process.env.VD_MODEL_ID]={name:process.env.VD_MODEL_LABEL,limit:{context:128000,output:8192}};
d.provider.vidhyora={npm:"@ai-sdk/openai-compatible",name:process.env.VD_BRAND,options:{baseURL:process.env.VD_BASE+"/api/v1/code"},models:m};
if(!d.model)d.model="vidhyora/"+process.env.VD_MODEL_ID}
fs.mkdirSync(require("path").dirname(file),{recursive:true});fs.writeFileSync(file,JSON.stringify(d,null,2)+"\n",{mode:0o600});' "$1" "$2" ;;
      *) "$JSON_TOOL" -c '
import json, os, sys
file, mode = sys.argv[1], sys.argv[2]
data = {}
try:
    with open(file, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        sys.exit(3)
except FileNotFoundError:
    pass
except ValueError:
    sys.exit(3)
env = os.environ
if mode == "auth":
    data["vidhyora"] = {"type": "api", "key": env["VD_KEY"]}
else:
    data.setdefault("$schema", "https://opencode.ai/config.json")
    data.setdefault("provider", {})["vidhyora"] = {
        "npm": "@ai-sdk/openai-compatible", "name": env["VD_BRAND"],
        "options": {"baseURL": env["VD_BASE"] + "/api/v1/code"},
        "models": {env["VD_MODEL_ID"]: {"name": env["VD_MODEL_LABEL"], "limit": {"context": 128000, "output": 8192}}},
    }
    data.setdefault("model", "vidhyora/" + env["VD_MODEL_ID"])
os.makedirs(os.path.dirname(os.path.abspath(file)), exist_ok=True)
with open(file, "w", encoding="utf-8") as handle:
    json.dump(data, handle, indent=2)
    handle.write("\n")
' "$1" "$2" ;;
    esac
  }
  merge_json "$AUTH_FILE" auth
  case $? in
    0) chmod 600 "$AUTH_FILE" 2>/dev/null; say "Saved your key in $AUTH_FILE (only you can read it)." ;;
    *) warn "Could not safely edit $AUTH_FILE. Sign in yourself: run  opencode auth login  -> Other -> vidhyora -> paste this key:"; say "   $API_KEY" ;;
  esac
elif [ ! -e "$AUTH_FILE" ]; then
  umask 077
  printf '{"vidhyora":{"type":"api","key":"%s"}}\n' "$API_KEY" > "$AUTH_FILE" && say "Saved your key in $AUTH_FILE (only you can read it)."
else
  warn "Python or Node.js is needed to update your existing $AUTH_FILE safely."
  warn "Sign in yourself: run  opencode auth login  -> Other -> vidhyora -> paste this key:"; say "   $API_KEY"
fi

# ------------------------------------------------------------- 4. the config
step "4/4  Adding ${MODEL_LABEL} to OpenCode"
if [ "$PROJECT_ONLY" = 1 ]; then CONFIG_FILE="$PWD/opencode.json"; else CONFIG_FILE="${XDG_CONFIG_HOME:-$HOME/.config}/opencode/opencode.json"; fi
CONFIG_OK=0
if [ -n "$JSON_TOOL" ]; then
  merge_json "$CONFIG_FILE" config && CONFIG_OK=1
elif [ ! -e "$CONFIG_FILE" ]; then
  mkdir -p "$(dirname "$CONFIG_FILE")" && cat > "$CONFIG_FILE" <<EOF && CONFIG_OK=1
{
  "\$schema": "https://opencode.ai/config.json",
  "provider": {
    "vidhyora": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "$BRAND",
      "options": { "baseURL": "$BASE_URL/api/v1/code" },
      "models": { "$MODEL_ID": { "name": "$MODEL_LABEL", "limit": { "context": 128000, "output": 8192 } } }
    }
  },
  "model": "vidhyora/$MODEL_ID"
}
EOF
fi
if [ "$CONFIG_OK" = 1 ]; then
  say "Updated $CONFIG_FILE"
else
  warn "Your existing $CONFIG_FILE could not be edited automatically (it may contain comments)."
  warn "Add this under \"provider\" in it, and set \"model\": \"vidhyora/$MODEL_ID\":"
  say "   \"vidhyora\": { \"npm\": \"@ai-sdk/openai-compatible\", \"name\": \"$BRAND\", \"options\": { \"baseURL\": \"$BASE_URL/api/v1/code\" }, \"models\": { \"$MODEL_ID\": { \"name\": \"$MODEL_LABEL\" } } }"
fi

say ""
say "${GREEN}${BOLD}All set.${OFF} Open a terminal in your project folder and run:"
say ""
say "    ${BOLD}opencode${OFF}"
say ""
if ! command -v opencode >/dev/null 2>&1; then
  warn "If 'opencode' is not found, open a NEW terminal window first (the installer updates your PATH)."
fi
say "${DIM}First time in a project? Type /init inside OpenCode. Manage connected computers under Start coding in your ${BRAND} account.${OFF}"
